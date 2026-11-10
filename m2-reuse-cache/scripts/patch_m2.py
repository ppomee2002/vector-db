#!/usr/bin/env python3
"""M2 trace patch for Qdrant v1.20.0 (on top of the gran patch). Idempotent; keeps *.m2orig backups.
QDRANT_M2TRACE=<file>: append one line per HNSW segment search ("H <graph dir> L<id>.<level> ... <vector ids>")
and one line per Gridstore payload read ("P <page dir> <point_offset> <page_id> <byte_offset> <length>")."""
import shutil, sys
from pathlib import Path
Q = Path(sys.argv[1])

def edit(rel, pairs):
    p = Q / rel
    b = p.with_name(p.name + ".m2orig")
    if not b.exists():
        shutil.copy(p, b)
    s = b.read_text()
    for old, new in pairs:
        assert s.count(old) >= 1, (rel, old[:80])
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)

# 1. blobstore: global trace writer
(Q / "lib/blobstore/src/m2trace.rs").write_text(r'''//! M2 experiment hook: access trace (see QDRANT_M2TRACE).
use std::io::Write;
use std::sync::{LazyLock, Mutex};

static OUT: LazyLock<Option<Mutex<std::io::BufWriter<std::fs::File>>>> = LazyLock::new(|| {
    let p = std::env::var("QDRANT_M2TRACE").ok().filter(|p| !p.is_empty())?;
    let f = std::fs::OpenOptions::new().create(true).append(true).open(&p).ok()?;
    log::info!("m2 access trace -> {p}");
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
        flush();
    });
    Some(Mutex::new(std::io::BufWriter::with_capacity(4 << 20, f)))
});

#[inline]
pub fn enabled() -> bool {
    OUT.is_some()
}

pub fn write(line: &str) {
    if let Some(m) = OUT.as_ref() {
        let _ = m.lock().unwrap().write_all(line.as_bytes());
    }
}

pub fn flush() {
    if let Some(m) = OUT.as_ref() {
        let _ = m.lock().unwrap().flush();
    }
}
''')
edit("lib/blobstore/src/lib.rs", [("pub mod config;\n", "pub mod config;\npub mod m2trace;\n")])
edit("lib/blobstore/src/blobstore/gridstore/pages.rs", [(
    "pub fn page_path(base_path: &Path, page_id: PageId) -> PathBuf {",
    "impl<S> Pages<S> {\n    pub(crate) fn trace_path(&self) -> &Path {\n        &self.base_path\n    }\n}\n\n"
    "pub fn page_path(base_path: &Path, page_id: PageId) -> PathBuf {")])
TR = ('        if crate::m2trace::enabled() {\n'
      '            crate::m2trace::write(&format!("P {} {} {} {} {}\\n", self.pages.trace_path().display(), point_offset, '
      'pointer.page_id, u64::from(pointer.block_offset) * self.config.block_size_bytes as u64, pointer.length));\n'
      '        }\n')
edit("lib/blobstore/src/blobstore/gridstore/view.rs", [
    ("        let raw = self.read_from_pages::<P>(pointer)?;\n", TR + "        let raw = self.read_from_pages::<P>(pointer)?;\n"),
    ("            pointers.push(((user_data, point_offset), pointer));\n",
     TR.replace("        ", "            ", 1).replace("\n        ", "\n            ") + "            pointers.push(((user_data, point_offset), pointer));\n"),
])

# 2. segment: per-search thread-local buffer, hooks on link reads and vector scoring
M2 = r'''
/// M2 experiment hook: per-thread buffer of the accesses of one HNSW search (QDRANT_M2TRACE).
pub(crate) mod m2 {
    use std::cell::RefCell;
    use std::fmt::Write;
    thread_local! { pub static BUF: RefCell<Option<String>> = const { RefCell::new(None) }; }
    #[inline]
    pub fn push_links(id: u32, level: usize) {
        BUF.with(|b| if let Some(s) = b.borrow_mut().as_mut() { let _ = write!(s, " L{id}.{level}"); });
    }
    #[inline]
    pub fn push_vecs(ids: &[u32]) {
        BUF.with(|b| if let Some(s) = b.borrow_mut().as_mut() { for i in ids { let _ = write!(s, " {i}"); } });
    }
}
'''
G = "lib/segment/src/index/hnsw_index/graph_layers.rs"
edit(G, [
    ("    pub(super) residency: GraphLinksResidency,\n}\n",
     "    pub(super) residency: GraphLinksResidency,\n    pub(super) m2_tag: String,\n}\n" + M2),
    ("            visited_pool: VisitedPool::new(),\n            residency,\n        })",
     "            visited_pool: VisitedPool::new(),\n            residency,\n            m2_tag: dir.to_string_lossy().into_owned(),\n        })"),
    ("            residency: GraphLinksResidency::Pinned,\n        };", "            residency: GraphLinksResidency::Pinned,\n            m2_tag: String::new(),\n        };"),
    ("        self.links.links(point_id, level).for_each(f);", "        m2::push_links(point_id, level);\n        self.links.links(point_id, level).for_each(f);"),
    ("        self.links.links(point_id, level).try_for_each(f)", "        m2::push_links(point_id, level);\n        self.links.links(point_id, level).try_for_each(f)"),
    ("    pub fn search(\n        &self,\n        top: usize,\n        ef: usize,\n        algorithm: SearchAlgorithm,",
     "    pub fn search(\n        &self,\n        top: usize,\n        ef: usize,\n        algorithm: SearchAlgorithm,\n"
     "        points_scorer: &mut FilteredScorer,\n        entry: &SearchEntry,\n        is_stopped: &AtomicBool,\n"
     "    ) -> OperationResult<Vec<ScoredPointOffset>> {\n"
     "        if !blobstore::m2trace::enabled() {\n            return self.search_m2inner(top, ef, algorithm, points_scorer, entry, is_stopped);\n        }\n"
     "        m2::BUF.with(|b| *b.borrow_mut() = Some(String::with_capacity(1 << 14)));\n"
     "        let r = self.search_m2inner(top, ef, algorithm, points_scorer, entry, is_stopped);\n"
     "        let s = m2::BUF.with(|b| b.borrow_mut().take()).unwrap_or_default();\n"
     "        blobstore::m2trace::write(&format!(\"H {}{}\\n\", self.m2_tag, s));\n        r\n    }\n\n"
     "    fn search_m2inner(\n        &self,\n        top: usize,\n        ef: usize,\n        algorithm: SearchAlgorithm,"),
])
edit("lib/segment/src/index/hnsw_index/graph_layers_builder.rs", [
    ("            visited_pool: self.visited_pool,\n            residency,\n        })",
     "            visited_pool: self.visited_pool,\n            residency,\n            m2_tag: String::new(),\n        })"),
    ("            residency: GraphLinksResidency::Pinned,\n        }", "            residency: GraphLinksResidency::Pinned,\n            m2_tag: String::new(),\n        }"),
])
edit("lib/segment/src/index/hnsw_index/point_scorer.rs", [
    ("        point_ids.truncate(n);\n\n        self.score_points_unfiltered(point_ids)",
     "        point_ids.truncate(n);\n        crate::index::hnsw_index::graph_layers::m2::push_vecs(point_ids);\n\n        self.score_points_unfiltered(point_ids)"),
    ("    pub fn score_point(&self, point_id: PointOffsetType) -> OperationResult<ScoreType> {\n        self.raw_scorer",
     "    pub fn score_point(&self, point_id: PointOffsetType) -> OperationResult<ScoreType> {\n"
     "        crate::index::hnsw_index::graph_layers::m2::push_vecs(&[point_id]);\n        self.raw_scorer"),
])
