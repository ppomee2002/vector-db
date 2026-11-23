"""1차 C1/C2/C3 통합 프로젝트의 **입출력 계약만** 정의한다. (계약 v1 후보, 2026-11-23 회의 후 고정)

아직 실제 Milvus 연결이나 알고리즘을 구현한 파일이 아니다.
내부 구현은 담당자 자유이고, 이 파일의 입력·출력·ID 규칙·기록 항목만 고정한다.

공통 규칙 (모든 모듈)
1. 공통 chunk_id = 원본 `chunks.jsonl`의 줄 번호(0 ~ 999,999). pk_seq와 같다(항등, CHECK.md 1-4).
2. 레이아웃별 PK 변환은 C1 모듈 내부에서만 한다. 모듈 사이에는 PK를 넘기지 않는다.
   (검색 컬렉션 nq_seq / nq1m_sel은 pk == chunk_id이므로 Search adapter는 항등임을 확인만 한다.)
3. 검색 순위를 보존한다. `QueryResult.hits`가 순위의 유일한 기록이다.
   Fetch용 ID 목록은 정렬될 수 있으므로(기존 C2 plan()은 PK 정렬) 순위를 거기서 복원하지 않는다.
4. 질문 키는 (query_set, q_idx). q_idx만으로는 질문을 식별하지 않는다(번호 체계가 둘 이상).
5. 생성자 규약 (모든 어댑터: Search, C1, C2, C3): `__init__(self, meta: ChunkMeta, cfg: dict)`.
   meta = 실행기가 로드한 ChunkMeta, cfg = config.yaml + 실행 overlay를 합친 dict(전체). 필요한 값은 cfg에서 읽는다.
   추가 인자는 기본값이 있는 키워드 인자만 허용한다(실행기와 계약 테스트는 (meta, cfg)만 넘긴다).
   C1/C2/C3 구현은 config의 c1.impl / c2.impl / c3.impl 이름으로 고른다(modules/registry.py의 @register).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, Mapping, Optional, Protocol, Sequence

if TYPE_CHECKING:  # 계약 표기용. 이 파일은 numpy를 실제로 import하지 않는다.
    import numpy as np

NeighborPolicy = Literal["none", "BC1", "all"]
FetchMode = Literal["individual", "batch", "range"]  # range는 1차 제외(D7). 설정에 넣으면 오류로 처리
BatchImpl = Literal["query_pk_in", "get"]            # 기본 query(filter="pk in [...]") (D6). get(ids)는 C1 비교 항목
SchedulerMode = Literal["m3", "m7"]
Layout = Literal["seq", "doc", "rand"]               # seq = nq_seq(문서 순), doc/rand = jihyeon w13c1_doc/rand
QuerySet = Literal["nq_test_real", "nq_test_ans", "c2x_test"]
QueryKey = tuple[str, int]                           # (query_set, q_idx)


# ---------------------------------------------------------------------------
# stage_ms 키 (단위 ms). 키 이름은 이 상수만 쓴다.
# ---------------------------------------------------------------------------
STAGE_EMBED = "embed"            # 질문 임베딩. 1차는 미리 계산한 query_vecs를 쓰므로 기록하지 않음(D10)
STAGE_SEARCH = "search"          # Milvus ANN 검색 호출
STAGE_C2_SELECT = "c2_select"    # 이웃 선택 + chunk_id↔pk 매핑 포함
STAGE_FETCH = "fetch"            # C1 본문 조회 (Top-context ∪ 이웃 한 번에)
STAGE_QUEUE_WAIT = "queue_wait"  # C3 semaphore 획득 전후로 따로 잰 대기 시간 (m3는 0)
STAGE_TOTAL = "total"            # 큐 진입부터 결과 완료까지 (대기 포함)
STAGE_SERVICE = "service"        # total − queue_wait
STAGE_KEYS = (STAGE_EMBED, STAGE_SEARCH, STAGE_C2_SELECT, STAGE_FETCH,
              STAGE_QUEUE_WAIT, STAGE_TOTAL, STAGE_SERVICE)


@dataclass(frozen=True)
class Settings:
    """실행 설정. 값의 출처는 config.yaml.

    규칙: ann_ef >= search_topk (Milvus HNSW는 ef가 limit보다 작으면 검색할 수 없음).
    1차 (D9):
    - B0와 C3 비교: search_topk 10, context_topk 10, ef 64.
    - C2 비교(대조군 ANN Top-(10+k) 포함): search_topk 40, context_topk 10, ef 64 (ef ≥ search_topk 만족).
      search_topk 40으로 받은 hits의 상위 10개가 search_topk 10일 때와 같은지 첫 실행에서 확인하고 실행 메타에 기록한다.
    2차 선별 대비(참고만): search_topk 100, ef 256, context_topk 10.
    """
    dataset: str = "NQ_1M_64word"
    query_set: QuerySet = "nq_test_ans"
    search_collection: str = "junlim/nq_seq"   # "<db>/<collection>". 소유자 동의 대기(D1)
    fetch_collection: str = "junlim/nq_seq"    # C1 비교 시 jihyeon/w13c1_doc 또는 w13c1_rand
    layout: Layout = "seq"                     # fetch_collection의 물리 배치. PK 매핑 선택에 쓰임
    search_topk: int = 10                      # Milvus에서 받아 오는 후보 수
    context_topk: int = 10                     # 그중 Context에 넣는 상위 개수 (C2 입력)
    ann_ef: int = 64
    neighbor_policy: NeighborPolicy = "none"
    fetch_mode: FetchMode = "batch"
    batch_impl: BatchImpl = "query_pk_in"
    scheduler_mode: SchedulerMode = "m3"
    clients: int = 64
    concurrency_cap: int = 8  # m7에서만 사용


@dataclass(frozen=True)
class ChunkMeta:
    """chunk_id로 색인된 메타데이터. C2 규칙과 C1 PK 변환이 같이 쓴다.

    로드 계약:
    - 모든 배열 길이는 n_chunks(1,000,000)와 같아야 한다. 다르면 실행을 중단한다.
    - config.yaml에 md5가 적힌 원본 파일은 실행 시작 시 md5를 검사하고, 결과를 실행 메타에 기록한다.
      md5가 없는 파일은 첫 실행 때 계산해 기록한다.
    - 원본 파일은 읽기만 한다(복사·수정 금지, D2).
    """
    n_chunks: int
    doc_idx: "np.ndarray"            # int64, chunk_id → 문서 번호 (단조 증가)
    chunk_idx: "np.ndarray"          # int32, 문서 안 순번
    flags: "np.ndarray"              # uint8, bit0 = 문장 중간에서 시작, bit1 = 문장 중간에서 끝남 (규칙 A용)
    nbytes: "np.ndarray"             # chunk 본문 바이트 수
    pk_of_chunk: Mapping[str, "np.ndarray"]   # layout → (chunk_id → pk). seq/doc은 항등, rand는 순열
    chunk_of_pk: Mapping[str, "np.ndarray"]   # layout → (pk → chunk_id) 역매핑
    offset: Optional[Mapping[str, "np.ndarray"]] = None  # 파일 저장소용: layout → 바이트 오프셋
    length: Optional["np.ndarray"] = None                # 파일 저장소용: chunk 바이트 길이


class ChunkMetaLoader(Protocol):
    def load(self, *, layouts: Sequence[Layout]) -> ChunkMeta: ...


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int  # PK가 아니라 원본에 고정된 공통 chunk_id
    rank: int      # 0부터, 검색 순위
    score: float | None = None  # IP 점수. 2차 선별 연결용으로 항상 채움


@dataclass
class FetchStats:
    """질문 하나의 Fetch 기록.

    n_requests, n_pks, bytes_returned는 Milvus 경로에서도 질문 단위로 정확히 셀 수 있으므로 필수.
    - n_requests: 이 질문의 Fetch에서 저장소에 보낸 요청 수(Milvus 호출 수 또는 pread/preadv 수).
    - n_pks: 요청한 PK 수(중복 제거 후).
    - bytes_returned: **반환된 본문 텍스트의 UTF-8 바이트 합** = sum(len(texts[c].encode("utf-8")) for c in 반환 dict).
      응답 메시지 크기, 메타 필드, 읽은 블록 크기는 넣지 않는다(경로와 무관하게 같은 정의).
    blocks, device_bytes는 Optional:
    - Milvus 경로에서는 질문 단위로 얻을 수 없다. 저장소 읽기 메트릭
      (internal_cache_cell_access_miss_bytes_total)은 서버 전체 값이라 동시 요청에서 섞인다.
      → 구간 단위 메트릭 차이로 실행 메타에 따로 기록하고, 여기에는 넣지 않는다(None).
    - 파일 저장소 경로(offset/length로 4KB 블록 계산)에서만 정확하다.
    """
    n_requests: int
    n_pks: int
    bytes_returned: int   # 반환 본문 UTF-8 바이트 합
    blocks: Optional[int] = None
    device_bytes: Optional[int] = None


@dataclass
class QueryResult:
    query_set: str
    q_idx: int
    hits: list[SearchHit]          # 검색 결과 search_topk개, 순위 순 (순위 보존의 기준)
    context_ids: list[int]         # hits 상위 context_topk개의 chunk_id, 순위 순 (C2 입력)
    neighbor_ids: list[int]        # C2가 추가한 이웃, chunk_id 순 정렬
    fetched_ids: list[int]         # 실제로 본문을 받은 chunk_id (context ∪ 이웃), chunk_id 순 정렬
    fetch_stats: FetchStats | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)  # 키는 STAGE_KEYS만
    error: str | None = None
    is_warmup: bool = False        # C3 예열 구간에서 끝난 질문이면 True. 집계(QPS, 지연 백분위, 오류율)에서 뺀다
    # 본문 텍스트는 QueryResult에 넣지 않는다. 품질 평가(정답 문자열 포함률)는 별도 기록으로 연결


class SearchAdapter(Protocol):
    """생성자 규약 (meta, cfg). 검색 컬렉션은 cfg milvus.search. 결과 PK를 chunk_id로 바꿔 SearchHit으로 돌려준다."""
    async def search(self, key: QueryKey, *, topk: int, ef: int) -> list[SearchHit]: ...


class C2NeighborAdapter(Protocol):
    """생성자에서 ChunkMeta를 주입받는다(예: __init__(self, meta: ChunkMeta)).

    select_neighbors 반환값: 입력 R에 없는 이웃 chunk_id, chunk_id 순 정렬 (pf3_rules.select와 같음).
    이웃 = 같은 문서의 ±1 chunk. 정책은 none / BC1 / all.
    대조군(ANN Top-(10+k), R3)은 이 모듈이 아니라 실험 계층에서 만든다(D8).
    """
    def select_neighbors(self, ranked_chunk_ids: Sequence[int], *, policy: NeighborPolicy) -> list[int]: ...


class C1FetchAdapter(Protocol):
    """chunk_id 목록을 받아 본문과 FetchStats를 돌려준다.

    레이아웃별 PK 변환(ChunkMeta.pk_of_chunk[layout])은 이 모듈 내부에서만 한다.
    반환 dict의 키는 chunk_id. 요청한 ID가 빠지면 FetchStats와 별개로 오류로 기록한다.
    fetch_source()는 실제 본문 출처를 밝힌다: "file:<데이터 파일 경로>" 또는 "milvus:<db>/<collection>".
    실행기는 이 값을 실행 메타에 기록하고, Settings.fetch_collection과 다르면 둘 다 남긴다.
    """
    def fetch_source(self) -> str: ...

    async def fetch_text(
        self, chunk_ids: Sequence[int], *, mode: FetchMode, batch_impl: BatchImpl
    ) -> tuple[Mapping[int, str], FetchStats]: ...


class C3SchedulerAdapter(Protocol):
    """질문 실행 전체(검색 → C2 → Fetch)의 동시 실행 수를 제어한다.

    m3: 제한 없음(비동기). m7: run_one 전체를 semaphore(cap)로 감쌈.
    스케줄러는 큐 진입 시각, semaphore 획득 시각, 완료 시각을 재서
    stage_ms의 queue_wait, total, service를 채운다(run_one 안에서는 재지 않음).

    시간 기반 실행 (기존 C3 측정과 같음: 예열 5초 + 측정 20초):
    - query_keys는 고정 seed 순서로 만든 목록이다. 클라이언트들은 이 목록을 순서대로 순환하며 질문을 꺼낸다.
    - 시작 후 warmup_s 동안 끝난 질문은 is_warmup=True로 표시한다.
    - warmup_s + duration_s가 지나면 새 질문을 시작하지 않고 멈춘다.
    - 집계(QPS, 지연 백분위, 오류율)에서는 is_warmup=True인 결과를 뺀다. QPS의 분모는 duration_s 구간이다.

    개수 기반 실행 (품질 평가용, max_queries가 None이 아닐 때):
    - query_keys를 고정 순서로(필요하면 순환) 정확히 max_queries개 실행하고 멈춘다. 결과는 max_queries개.
      max_queries = len(query_keys)면 목록을 한 바퀴 실행한다.
    - warmup_s, duration_s는 쓰지 않는다. 모든 결과는 is_warmup=False.
    - mode/clients/cap(동시 실행 제어)과 queue_wait/total/service 기록은 시간 기반과 같다.
    """
    async def run_many(
        self,
        query_keys: Sequence[QueryKey],
        run_one: Callable[[QueryKey], Awaitable[QueryResult]],
        *,
        mode: SchedulerMode,
        clients: int,
        cap: int,
        warmup_s: float,
        duration_s: float,
        max_queries: Optional[int] = None,
    ) -> list[QueryResult]: ...
