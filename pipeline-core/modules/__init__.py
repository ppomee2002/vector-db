"""pipeline-core 모듈 패키지.

- 공통 코어(총괄): config_loader, registry, chunk_meta, queries, search_milvus, pipeline, evaluate, run_meta
- 자리 표시용 reference 어댑터(담당자 구현으로 교체): c1_file_reference, c2_none_reference, c3_serial_reference
- 담당자 어댑터: 이 폴더에 파일을 추가하고 @register("c1"|"c2"|"c3", "<이름>")로 등록한다 (SMOKE.md 참고).
"""
