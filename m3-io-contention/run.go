package main

// run: the RAG context-retrieval workload.
//
//   load generator --(job)--> worker: [queue wait] -> vector search (CPU) -> read k chunks (SSD) -> reply
//
// Load generator:
//   closed: N clients, each sends the next request when the previous one returns (sweep N)
//   open:   requests arrive as a Poisson process at a fixed rate, latency is measured from the
//           scheduled arrival time (no coordinated omission); rates are fractions of a measured capacity
// Knobs that isolate the cause of a slowdown:
//   -workers 8|n        fixed pool (synchronous server) vs one worker per request
//   -io sync|parallel   chunks read one after another vs all k at once
//   -search=false       no CPU search (vector search offloaded, e.g. to a GPU)
//   -hit p              fraction of chunk reads served from an in-memory cache
// Per request we record queue wait / search / I/O; per second and per measurement window we
// record CPU, SSD and stall metrics.

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type runCfg struct {
	tag, family, variant string
	mode, load           string // io mode, closed|open
	n, workers, k        int
	nprobe, nlist        int
	chunk                int64
	warmup, dur          time.Duration
	rep, orderIdx        int
	gmp, cpus            int
	cpuset               string
	searchAlone, qgenUs  float64
	hit                  float64
	lruMB                int64
	zipf                 float64
	search, ts           bool
	perReq               bool
	rate, capacity       float64
	maxInflight          int
}

type job struct {
	q                               []float32
	tEnq                            int64
	m                               bool
	queue, search, io, total, ioDev int64
	done                            chan struct{}
	onDone                          func()
}

type reqRec struct{ queue, search, io, total, ioDev int64 }

// wstate: everything one in-flight request needs; reused, never shared concurrently.
type wstate struct {
	id        int
	bufs      [][]byte
	lat       []int64
	offs      []int64
	keys      []int64
	ids       []int32
	miss      []int
	nio, hits int
	sc        *Scratch
	rng       xrng
	ioLats    []int64
	// async only
	cbs       []iocb
	cbp       []*iocb
	tSub      []int64
	remaining atomic.Int32
	j         *job
	tS, tSe   int64
}

type server struct {
	f         *os.File
	ix        *Index
	slots     int64
	cfg       *runCfg
	seed      int64
	app       *AppStats
	base      time.Time
	cache     []byte
	lru       *lruCache
	aio       *aioCtx
	ioSem     chan struct{}
	ioErr     atomic.Int64
	measuring atomic.Bool
	stopping  atomic.Bool

	mu     sync.RWMutex
	states []*wstate
	free   chan *wstate
}

func newServer(f *os.File, ix *Index, slots int64, cfg *runCfg, seed int64, cache []byte) (*server, error) {
	s := &server{f: f, ix: ix, slots: slots, cfg: cfg, seed: seed, app: &AppStats{}, base: time.Now(), cache: cache,
		free: make(chan *wstate, 1<<16)}
	if cfg.lruMB > 0 {
		s.lru = newLRU(cfg.lruMB<<20, cfg.chunk)
	}
	if cfg.mode == "async" {
		a, err := newAIO(8192)
		if err != nil {
			return nil, fmt.Errorf("native AIO unavailable (%v): -io async cannot run here", err)
		}
		s.aio = a
		s.ioSem = make(chan struct{}, a.size)
	}
	return s, nil
}

func (s *server) close() {
	if s.aio != nil {
		s.aio.destroy()
	}
}

func (s *server) now() int64 { return int64(time.Since(s.base)) }

func (s *server) getState() *wstate {
	select {
	case w := <-s.free:
		return w
	default:
	}
	k := s.cfg.k
	w := &wstate{bufs: make([][]byte, k), lat: make([]int64, k), offs: make([]int64, k), keys: make([]int64, k),
		ids: make([]int32, k), miss: make([]int, 0, k), sc: s.ix.NewScratch(s.cfg.nprobe, k),
		cbs: make([]iocb, k), cbp: make([]*iocb, k), tSub: make([]int64, k)}
	for i := range w.bufs {
		w.bufs[i] = alignedBuf(int(s.cfg.chunk))
	}
	s.mu.Lock()
	w.id = len(s.states)
	w.rng = xrng(mix64(uint64(len(s.states))+uint64(s.seed)<<32+uint64(s.cfg.rep)<<48) | 1)
	s.states = append(s.states, w)
	s.mu.Unlock()
	return w
}

func (s *server) stateByID(id int) *wstate {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.states[id]
}

func (s *server) putState(w *wstate) {
	select {
	case s.free <- w:
	default:
	}
}

func (w *wstate) unif() float64 { return float64(w.rng.next()>>11) / (1 << 53) }

// process runs one request on state w. It returns true when the request is
// complete; in async mode it returns false after submitting the reads and the
// completion thread finishes the request (and calls j.onDone).
func (s *server) process(j *job, w *wstate) bool {
	cfg := s.cfg
	tS := s.now()
	s.app.started.Add(1)
	s.app.busy.Add(1)
	var ids []int32
	if cfg.search {
		ids = s.ix.Search(j.q, cfg.nprobe, cfg.k, w.sc)
	} else { // search offloaded: documents drawn from the (possibly skewed) topic distribution
		for i := range w.ids {
			w.ids[i] = s.ix.RandomDoc(&w.rng)
		}
		ids = w.ids
	}
	tSe := s.now()
	w.miss = w.miss[:0]
	w.nio, w.hits = len(ids), 0
	for i, id := range ids {
		key := int64(mix64(uint64(id)^uint64(s.seed)<<40) % uint64(s.slots))
		w.keys[i], w.offs[i], w.lat[i] = key, key*cfg.chunk, 0
		if s.lru != nil {
			t := time.Now()
			if s.lru.get(key, w.bufs[i]) {
				w.lat[i] = int64(time.Since(t))
				w.hits++
				continue
			}
		} else if cfg.hit > 0 && w.unif() < cfg.hit {
			n := int64(len(s.cache)) / cfg.chunk
			co := int64(mix64(uint64(id))%uint64(n)) * cfg.chunk
			t := time.Now()
			copy(w.bufs[i], s.cache[co:co+cfg.chunk])
			w.lat[i] = int64(time.Since(t))
			w.hits++
			continue
		}
		w.miss = append(w.miss, i)
	}
	if cfg.mode == "async" && len(w.miss) > 0 {
		w.j, w.tS, w.tSe = j, tS, tSe
		w.remaining.Store(int32(len(w.miss)))
		for n, i := range w.miss {
			fillPread(&w.cbs[n], s.f.Fd(), w.bufs[i], w.offs[i], uint64(w.id)<<16|uint64(i))
			w.cbp[n] = &w.cbs[n]
		}
		for range w.miss {
			s.ioSem <- struct{}{} // back-pressure: bounded number of reads in flight
		}
		t0 := s.now()
		for _, i := range w.miss {
			w.tSub[i] = t0
		}
		s.app.inflightIO.Add(int64(len(w.miss)))
		s.app.busy.Add(-1) // the worker is free as soon as the reads are submitted
		if err := s.aio.submit(w.cbp[:len(w.miss)]); err != nil {
			if s.ioErr.Add(1) == 1 {
				fmt.Fprintln(os.Stderr, "io_submit error:", err)
			}
			for range w.miss {
				<-s.ioSem
			}
			s.app.inflightIO.Add(-int64(len(w.miss)))
			s.finish(j, w, tS, tSe, s.now())
			return true
		}
		return false
	}
	read := func(i int) {
		s.app.inflightIO.Add(1)
		t := time.Now()
		if err := preadFull(s.f, w.bufs[i], w.offs[i]); err != nil {
			if s.ioErr.Add(1) == 1 {
				fmt.Fprintln(os.Stderr, "read error:", err)
			}
		}
		w.lat[i] = int64(time.Since(t))
		s.app.inflightIO.Add(-1)
	}
	if cfg.mode == "sync" || len(w.miss) <= 1 {
		for _, i := range w.miss {
			read(i)
		}
	} else {
		var wg sync.WaitGroup
		for _, i := range w.miss {
			wg.Add(1)
			go func(i int) { read(i); wg.Done() }(i)
		}
		wg.Wait()
	}
	if s.lru != nil {
		for _, i := range w.miss {
			s.lru.put(w.keys[i], w.bufs[i])
		}
	}
	s.app.busy.Add(-1)
	s.finish(j, w, tS, tSe, s.now())
	return true
}

// finish fills the job's timings and the counters.
func (s *server) finish(j *job, w *wstate, tS, tSe, tE int64) {
	var mx, sum int64
	for i := 0; i < w.nio; i++ {
		sum += w.lat[i]
		if w.lat[i] > mx {
			mx = w.lat[i]
		}
	}
	if j.m && s.measuring.Load() {
		for _, i := range w.miss {
			w.ioLats = append(w.ioLats, w.lat[i])
		}
	}
	j.queue, j.search, j.io, j.total = tS-j.tEnq, tSe-tS, tE-tSe, tE-j.tEnq
	j.ioDev = sum // sync: back to back
	if s.cfg.mode != "sync" {
		j.ioDev = mx // parallel / async: bounded by the slowest read
	}
	s.app.reqDone.Add(1)
	s.app.ioDone.Add(int64(len(w.miss)))
	s.app.ioBytes.Add(int64(len(w.miss)) * s.cfg.chunk)
	s.app.hits.Add(int64(w.hits))
}

// reaper is the completion thread of the async server: it collects finished
// reads and completes a request when its last read arrives.
func (s *server) reaper(stop *atomic.Bool, done chan struct{}) {
	defer close(done)
	runtime.LockOSThread()
	ev := make([]ioEvent, 256)
	for {
		if stop.Load() && s.app.inflightIO.Load() == 0 {
			return
		}
		n, err := s.aio.getEvents(ev, 1, 20*time.Millisecond)
		if err != nil {
			if s.ioErr.Add(1) == 1 {
				fmt.Fprintln(os.Stderr, "io_getevents error:", err)
			}
			time.Sleep(time.Millisecond)
			continue
		}
		now := s.now()
		for _, e := range ev[:n] {
			w := s.stateByID(int(e.data >> 16))
			i := int(e.data & 0xffff)
			w.lat[i] = now - w.tSub[i]
			if e.res != s.cfg.chunk && s.ioErr.Add(1) == 1 {
				fmt.Fprintln(os.Stderr, "aio read error, res =", e.res)
			}
			<-s.ioSem
			s.app.inflightIO.Add(-1)
			if s.lru != nil {
				s.lru.put(w.keys[i], w.bufs[i])
			}
			if w.remaining.Add(-1) == 0 {
				j := w.j
				w.j = nil
				s.finish(j, w, w.tS, w.tSe, now)
				s.putState(w)
				j.onDone()
			}
		}
	}
}

func (s *server) startReaper() (stop *atomic.Bool, done chan struct{}) {
	stop, done = &atomic.Bool{}, make(chan struct{})
	if s.aio == nil {
		close(done)
		return
	}
	go s.reaper(stop, done)
	return
}

func (s *server) allIoLats() []int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []int64
	for _, w := range s.states {
		out = append(out, w.ioLats...)
	}
	return out
}

// ---------- columns ----------

var summaryCols = []string{"tag", "family", "variant", "rep", "order_idx", "load", "mode", "workers", "n",
	"rate_target", "capacity", "offered_qps", "dropped", "incomplete", "gen_lag_ms",
	"k", "chunk_kb", "nprobe", "nlist", "search", "hit", "cache_mb", "zipf", "hit_obs", "cpus", "cpuset", "gomaxprocs", "ts", "dur_s", "reqs", "qps",
	"lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
	"queue_mean_ms", "queue_p95_ms", "search_mean_ms", "search_p95_ms", "search_alone_ms", "search_inflation", "qgen_us",
	"io_mean_ms", "io_p95_ms", "io_call_mean_ms", "io_gap_mean_ms",
	"ioop_mean_us", "ioop_p50_us", "ioop_p95_us", "ioop_p99_us",
	"our_iops", "our_mbps", "cpu_pct", "client_cpu_pct", "cpu_limit_pct", "cpu_ms_per_req", "cg_cpu_pct", "runq_threads", "sched_mean_us", "sched_p99_us", "threads",
	"avg_inflight_io", "avg_busy_workers", "avg_queued",
	"dev_r_iops", "dev_r_mbps", "dev_r_await_ms", "ioop_over_dev", "dev_aqu", "dev_util_pct", "bg_r_mbps", "dev_w_mbps",
	"host_busy_pct", "host_iowait_pct", "other_cpu_pct",
	"psi_io_some_pct", "psi_io_full_pct", "psi_cpu_some_pct", "psi_cg_io_some_pct", "psi_cg_cpu_some_pct",
	"bg_pre_r_mbps", "bg_pre_w_mbps", "little_n", "io_errors", "warn"}

var tsCols = []string{"run_id", "tag", "rep", "load", "mode", "workers", "n", "rate_target", "t_s", "phase", "qps", "offered_qps", "our_iops", "cpu_pct",
	"runq_threads", "client_cpu_pct", "avg_inflight_io", "avg_busy_workers", "avg_queued", "dev_r_iops", "dev_aqu", "dev_r_await_ms", "bg_r_mbps", "dev_w_mbps",
	"other_cpu_pct", "psi_io_some_pct", "sched_p99_us", "threads"}

func fnan(v float64, f func(float64) string) string {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return ""
	}
	return f(v)
}

func ratio(a, b float64) float64 {
	if b <= 0 {
		return math.NaN()
	}
	return a / b
}

// ---------- command ----------

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	file := fs.String("file", "", "data file (from `m3bench prep`)")
	devF := fs.String("dev", "auto", "block device name in /proc/diskstats")
	tag := fs.String("tag", "run", "label: <variant>_<family>, e.g. A_base")
	mode := fs.String("io", "sync", "chunk reads: sync (one after another) | parallel (k blocking reads at once) | async (native AIO, worker never blocks)")
	workersF := fs.String("workers", "8", "worker pool size: a number, or n (= one worker per request)")
	load := fs.String("load", "closed", "closed (N clients, sweep -n) | open (Poisson arrivals, sweep -rates)")
	nsF := fs.String("n", "1,4,8,16,32,64", "closed loop: concurrent clients to sweep")
	ratesF := fs.String("rates", "0.3,0.5,0.7,0.8,0.9,0.95,1.0", "open loop: arrival rates as fractions of capacity")
	capF := fs.Float64("capacity", 0, "open loop: capacity in req/s (0 = measure it first with a short closed-loop run)")
	probeN := fs.Int("probe-n", 64, "open loop: clients used to measure capacity")
	maxInfl := fs.Int("max-inflight", 4096, "open loop, -workers n: cap on concurrent requests (excess is dropped and counted)")
	order := fs.String("order", "shuffle", "order of points within a repetition: shuffle | asc")
	reps := fs.Int("rep", 1, "repetitions")
	repStart := fs.Int("rep-start", 1, "label of the first repetition (run_all.sh interleaves scenarios per repetition)")
	k := fs.Int("k", 10, "chunks retrieved per request (top-k)")
	chunkF := fs.String("chunk", "4k", "chunk read size (multiple of 4k)")
	nprobe := fs.Int("nprobe", 4, "IVF clusters scanned per query (CPU cost knob)")
	nvec := fs.Int("nvec", 500000, "vectors in the index")
	dim := fs.Int("dim", 384, "vector dimension")
	nlist := fs.Int("nlist", 2048, "IVF clusters")
	search := fs.Bool("search", true, "run the vector search on the CPU (false = search offloaded, random documents)")
	hit := fs.Float64("hit", 0, "fraction of chunk reads served from memory, drawn at random (simple model; see -cache-mb)")
	lruMB := fs.Int64("cache-mb", 0, "LRU chunk cache size in MB (realistic cache; combine with -zipf for popular topics)")
	zipf := fs.Float64("zipf", 0, "query topic skew (0 = uniform)")
	warmup := fs.Duration("warmup", 10*time.Second, "warm-up per point (not measured)")
	dur := fs.Duration("dur", 30*time.Second, "measurement time per point")
	pause := fs.Duration("pause", 2*time.Second, "idle time between points (also samples background load)")
	ts := fs.Bool("ts", true, "write the per-second time series (false = only window snapshots; used to measure sampling overhead)")
	cpus := fs.Int("cpus", 7, "server CPUs the process is pinned to (team limit 800% = 7 server + 1 client)")
	cpuset := fs.String("cpuset", "auto", "auto (least-busy CPUs on the SSD's NUMA node), none, or a list like 0-7")
	clientF := fs.String("client-cpu", "auto", "extra CPU for the load generator and samplers: auto, none, or a CPU number")
	gmp := fs.Int("gomaxprocs", 128, "Go Ps; keep it well above -cpus so blocked reads do not stall runnable work")
	direct := fs.Bool("direct", true, "O_DIRECT reads (bypass the page cache so every miss hits the SSD)")
	outDir := fs.String("out", "results", "output directory (summary.csv, timeseries.csv appended)")
	seed := fs.Int64("seed", 1, "random seed")
	fs.Parse(args)

	if *mode != "sync" && *mode != "parallel" && *mode != "async" {
		return errors.New("-io must be sync, parallel or async")
	}
	if *mode == "async" {
		if err := aioAvailable(); err != nil {
			return fmt.Errorf("-io async needs native AIO, unavailable here: %v", err)
		}
	}
	if *load != "closed" && *load != "open" {
		return errors.New("-load must be closed or open")
	}
	if *hit < 0 || *hit > 1 {
		return errors.New("-hit must be in [0,1]")
	}
	ns, err := parseIntList(*nsF)
	if err != nil {
		return err
	}
	var rates []float64
	for _, p := range strings.Split(*ratesF, ",") {
		if v, err := strconv.ParseFloat(strings.TrimSpace(p), 64); err == nil && v > 0 {
			rates = append(rates, v)
		}
	}
	if *load == "open" && len(rates) == 0 {
		return errors.New("bad -rates")
	}
	chunk, err := parseSize(*chunkF)
	if err != nil || chunk%align != 0 {
		return fmt.Errorf("bad -chunk %q (must be a multiple of 4k)", *chunkF)
	}
	if *k < 1 || *nprobe < 1 || *nprobe > *nlist {
		return errors.New("bad -k / -nprobe")
	}
	if mem := int64(*nvec+*nlist) * int64(*dim) * 4; mem > 8<<30 {
		return fmt.Errorf("index would take %.1f GiB (> 8 GiB); lower -nvec or -dim (team memory limit is 16 GB)", float64(mem)/(1<<30))
	}
	perReq := *workersF == "n"
	fixedW := 0
	if !perReq {
		fixedW, err = strconv.Atoi(*workersF)
		if err != nil || fixedW < 1 {
			return fmt.Errorf("bad -workers %q", *workersF)
		}
	}
	dev := resolveDev(*devF)
	pinned, err := ensurePinnedClient(*cpus, *cpuset, *clientF, dev)
	if err != nil {
		return err
	}
	ncpu := *cpus
	if pinned == "none" {
		*gmp = *cpus
	} else {
		ncpu = effectiveCPUs()
	}
	runtime.GOMAXPROCS(*gmp)
	debug.SetMaxThreads(30000)
	fmt.Printf("[%s] server CPUs: %s (%d), client CPU: %d, GOMAXPROCS=%d, load=%s io=%s workers=%s\n", *tag, pinned, ncpu, clientCPU(), *gmp, *load, *mode, *workersF)

	f, size, err := openData(*file, *direct)
	if err != nil {
		return err
	}
	defer f.Close()
	slots := size / chunk

	t0 := time.Now()
	ix := BuildIndex(*nvec, *dim, *nlist, *zipf, *seed, ncpu)
	fmt.Printf("[%s] index: %d x %d, nlist=%d (%.0f MB) built in %.1fs\n", *tag, *nvec, *dim, *nlist, float64(ix.MemBytes())/1e6, time.Since(t0).Seconds())
	searchAlone, qgenUs := calibrateSearch(ix, *nprobe, *k)
	if *search {
		fmt.Printf("[%s] search alone: %.3f ms/query (nprobe=%d); query generation %.1f us\n", *tag, searchAlone, *nprobe, qgenUs)
	} else {
		searchAlone = math.NaN()
		fmt.Printf("[%s] search disabled (offloaded); documents are drawn at random\n", *tag)
	}
	var cache []byte
	if *hit > 0 {
		cache = make([]byte, 64<<20)
		r := xrng(7)
		for i := 0; i+8 <= len(cache); i += 8 {
			v := r.next()
			for b := 0; b < 8; b++ {
				cache[i+b] = byte(v >> (8 * b))
			}
		}
	}

	if err := os.MkdirAll(*outDir, 0755); err != nil {
		return err
	}
	so, newS, err := openAppend(*outDir + "/summary.csv")
	if err != nil {
		return err
	}
	defer so.Close()
	if newS {
		fmt.Fprintln(so, strings.Join(summaryCols, ","))
	}
	to, newT, err := openAppend(*outDir + "/timeseries.csv")
	if err != nil {
		return err
	}
	defer to.Close()
	tw := bufio.NewWriter(to)
	defer tw.Flush()
	if newT {
		fmt.Fprintln(tw, strings.Join(tsCols, ","))
	}

	variant, family := *tag, ""
	if i := strings.IndexByte(*tag, '_'); i > 0 {
		variant, family = (*tag)[:i], (*tag)[i+1:]
	}
	mk := func(rep int) runCfg {
		return runCfg{tag: *tag, family: family, variant: variant, mode: *mode, load: *load, k: *k, nprobe: *nprobe, nlist: *nlist,
			chunk: chunk, warmup: *warmup, dur: *dur, rep: rep, gmp: *gmp, cpus: ncpu, cpuset: pinned,
			searchAlone: searchAlone, qgenUs: qgenUs, hit: *hit, lruMB: *lruMB, zipf: *zipf, search: *search, ts: *ts, perReq: perReq, maxInflight: *maxInfl}
	}
	workersFor := func(n int) int {
		if perReq {
			return n
		}
		return fixedW
	}

	for rep := *repStart; rep < *repStart+*reps; rep++ {
		rng := rand.New(rand.NewSource(*seed*1000 + int64(rep)*7919 + int64(len(*tag))))
		if *load == "closed" {
			idx := make([]int, len(ns))
			for i := range idx {
				idx[i] = i
			}
			if *order == "shuffle" {
				rng.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
			}
			for oi, i := range idx {
				cfg := mk(rep)
				cfg.n, cfg.workers, cfg.orderIdx = ns[i], workersFor(ns[i]), oi
				bg := background(dev, pause.Seconds())
				row, err := runPoint(f, ix, slots, &cfg, dev, tw, *seed, cache, bg)
				if err != nil {
					return err
				}
				fmt.Fprintln(so, strings.Join(row, ","))
				tw.Flush()
			}
		} else {
			capacity := *capF
			if capacity <= 0 {
				pc := mk(rep)
				pc.load, pc.n, pc.workers, pc.ts = "closed", *probeN, workersFor(*probeN), false
				pc.warmup, pc.dur = 3*time.Second, 8*time.Second
				capacity, err = probeCapacity(f, ix, slots, &pc, dev, *seed, cache)
				if err != nil {
					return err
				}
				fmt.Printf("[%s r%d] capacity (closed loop, n=%d): %.0f req/s\n", *tag, rep, *probeN, capacity)
			}
			idx := make([]int, len(rates))
			for i := range idx {
				idx[i] = i
			}
			if *order == "shuffle" {
				rng.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
			}
			for oi, i := range idx {
				cfg := mk(rep)
				cfg.rate, cfg.capacity, cfg.orderIdx = rates[i]*capacity, capacity, oi
				cfg.n, cfg.workers = 0, fixedW
				bg := background(dev, pause.Seconds())
				row, err := runPoint(f, ix, slots, &cfg, dev, tw, *seed, cache, bg)
				if err != nil {
					return err
				}
				fmt.Fprintln(so, strings.Join(row, ","))
				tw.Flush()
			}
		}
	}
	return nil
}

func calibrateSearch(ix *Index, nprobe, k int) (searchMs, qgenUs float64) {
	r := rand.New(rand.NewSource(99))
	q := make([]float32, ix.dim)
	s := ix.NewScratch(nprobe, k)
	t := time.Now()
	for i := 0; i < 2000; i++ {
		ix.MakeQuery(r, q)
	}
	qgenUs = float64(time.Since(t).Nanoseconds()) / 2000 / 1000
	t = time.Now()
	for i := 0; i < 300; i++ {
		ix.MakeQuery(r, q)
		ix.Search(q, nprobe, k, s)
	}
	searchMs = (float64(time.Since(t).Nanoseconds())/300/1e6 - qgenUs/1000)
	return
}

// ---------- one measurement point ----------

type pointResult struct {
	recs                        []reqRec
	issued, dropped, incomplete int64
	lagMs                       float64
	a, b                        Snapshot
}

func runPoint(f *os.File, ix *Index, slots int64, cfg *runCfg, dev string, tw *bufio.Writer, seed int64, cache []byte, bg Delta) ([]string, error) {
	s, err := newServer(f, ix, slots, cfg, seed, cache)
	if err != nil {
		return nil, err
	}
	defer s.close()
	var pr pointResult
	if cfg.load == "closed" {
		pr = s.driveClosed(dev, tw)
	} else {
		pr = s.driveOpen(dev, tw)
	}
	return s.summarize(pr, bg), nil
}

func probeCapacity(f *os.File, ix *Index, slots int64, cfg *runCfg, dev string, seed int64, cache []byte) (float64, error) {
	s, err := newServer(f, ix, slots, cfg, seed, cache)
	if err != nil {
		return 0, err
	}
	defer s.close()
	pr := s.driveClosed(dev, nil)
	return float64(len(pr.recs)) / cfg.dur.Seconds(), nil
}

// tsLoop writes one time-series row per second until stop is set.
func (s *server) tsLoop(dev string, tw *bufio.Writer, phase *atomic.Int32, stop *atomic.Bool, done chan struct{}) {
	defer close(done)
	pinClient()
	if tw == nil || !s.cfg.ts {
		for !stop.Load() {
			time.Sleep(50 * time.Millisecond)
		}
		return
	}
	cfg := s.cfg
	runID := fmt.Sprintf("%s_n%d_rate%.0f_r%d", cfg.tag, cfg.n, cfg.rate, cfg.rep)
	prev := takeSnapshot(dev, s.app)
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	for !stop.Load() {
		<-tk.C
		cur := takeSnapshot(dev, s.app)
		d := diff(prev, cur)
		prev = cur
		ph := [...]string{"warmup", "measure", "drain"}[phase.Load()]
		row := []string{runID, cfg.tag, fmt.Sprint(cfg.rep), cfg.load, cfg.mode, fmt.Sprint(cfg.workers), fmt.Sprint(cfg.n), f1(cfg.rate),
			f1(time.Since(s.base).Seconds()), ph, f1(d.QPS), f1(d.Offered), f1(d.OurIOPS), f1(d.CPUPct), f2(d.RunqThreads), f1(d.ClientCPUPct),
			f2(d.AvgInflightIO), f2(d.AvgBusy), f2(d.AvgQueued), f1(d.DevRIOPS), f2(d.DevAqu), f3(d.DevRAwaitMs),
			f1(d.BgRMBps), f1(d.DevWMBps), f1(d.OtherCPUPct), fnan(d.PsiHostIOSome, f2), f1(d.SchedP99Us), fmt.Sprint(d.Threads)}
		fmt.Fprintln(tw, strings.Join(row, ","))
	}
}

func (s *server) driveClosed(dev string, tw *bufio.Writer) pointResult {
	cfg := s.cfg
	var stop, gstop atomic.Bool
	jobs := make(chan *job, cfg.n)
	go s.app.gaugeLoop(&gstop)
	rstop, rdone := s.startReaper()
	async := cfg.mode == "async"

	var wwg sync.WaitGroup
	for wi := 0; wi < cfg.workers; wi++ {
		wwg.Add(1)
		go func() {
			defer wwg.Done()
			var w *wstate
			if !async {
				w = s.getState() // a synchronous worker owns its buffers
			}
			for j := range jobs {
				ws := w
				if async {
					ws = s.getState() // requests in flight keep their buffers until the reads land
				}
				if s.process(j, ws) {
					if async {
						s.putState(ws)
					}
					j.onDone()
				}
			}
		}()
	}
	recs := make([][]reqRec, cfg.n)
	var cwg sync.WaitGroup
	for ci := 0; ci < cfg.n; ci++ {
		cwg.Add(1)
		go func(ci int) {
			defer cwg.Done()
			r := rand.New(rand.NewSource(s.seed*1000003 + int64(cfg.rep)*7919 + int64(ci)))
			j := &job{q: make([]float32, s.ix.dim), done: make(chan struct{}, 1)}
			j.onDone = func() { j.done <- struct{}{} }
			for !stop.Load() {
				if cfg.search {
					s.ix.MakeQuery(r, j.q)
				}
				j.m = s.measuring.Load()
				j.tEnq = s.now()
				s.app.enq.Add(1)
				jobs <- j
				<-j.done
				if j.m && s.measuring.Load() {
					recs[ci] = append(recs[ci], reqRec{j.queue, j.search, j.io, j.total, j.ioDev})
				}
			}
		}(ci)
	}
	var phase atomic.Int32
	tsDone := make(chan struct{})
	go s.tsLoop(dev, tw, &phase, &gstop, tsDone)

	time.Sleep(cfg.warmup)
	a := takeSnapshot(dev, s.app)
	s.measuring.Store(true)
	phase.Store(1)
	time.Sleep(cfg.dur)
	s.measuring.Store(false)
	b := takeSnapshot(dev, s.app)
	phase.Store(2)
	stop.Store(true)
	cwg.Wait()
	close(jobs)
	wwg.Wait()
	rstop.Store(true)
	<-rdone
	gstop.Store(true)
	<-tsDone
	var all []reqRec
	for _, r := range recs {
		all = append(all, r...)
	}
	return pointResult{recs: all, a: a, b: b}
}

func (s *server) driveOpen(dev string, tw *bufio.Writer) pointResult {
	cfg := s.cfg
	var gstop, genDone atomic.Bool
	go s.app.gaugeLoop(&gstop)
	rstop, rdone := s.startReaper()
	async := cfg.mode == "async"
	mStart := int64(cfg.warmup)
	mEnd := mStart + int64(cfg.dur)

	var shardMu [64]sync.Mutex
	var shards [64][]reqRec
	var measuredDone, issued, dropped atomic.Int64
	var lagSum atomic.Int64
	var seq atomic.Int64
	record := func(j *job) {
		if !j.m {
			return
		}
		i := seq.Add(1) & 63
		shardMu[i].Lock()
		shards[i] = append(shards[i], reqRec{j.queue, j.search, j.io, j.total, j.ioDev})
		shardMu[i].Unlock()
		measuredDone.Add(1)
	}

	var jobs chan *job
	var wwg sync.WaitGroup
	var inflight atomic.Int64
	var rwg sync.WaitGroup
	if !cfg.perReq {
		jobs = make(chan *job, 1<<20)
		for wi := 0; wi < cfg.workers; wi++ {
			wwg.Add(1)
			go func() {
				defer wwg.Done()
				var w *wstate
				if !async {
					w = s.getState()
				}
				for j := range jobs {
					if s.stopping.Load() {
						continue
					}
					ws := w
					if async {
						ws = s.getState()
					}
					if s.process(j, ws) {
						if async {
							s.putState(ws)
						}
						j.onDone()
					}
				}
			}()
		}
	}
	dispatch := func(j *job) bool {
		if !cfg.perReq {
			select {
			case jobs <- j:
				return true
			default:
				return false
			}
		}
		if inflight.Add(1) > int64(cfg.maxInflight) {
			inflight.Add(-1)
			return false
		}
		rwg.Add(1)
		go func() {
			defer rwg.Done()
			w := s.getState()
			if s.process(j, w) {
				s.putState(w)
				j.onDone()
			}
		}()
		return true
	}
	// Poisson arrival generator; tEnq is the *scheduled* arrival time.
	go func() {
		defer genDone.Store(true)
		pinClient()
		syscall.RawSyscall(syscall.SYS_PRCTL, 29 /*PR_SET_TIMERSLACK*/, 1000, 0) // 1us; only affects this thread
		r := rand.New(rand.NewSource(s.seed*31 + int64(cfg.rep)*7919 + int64(cfg.rate)))
		next := s.now()
		for next < mEnd {
			now := s.now()
			if next > now {
				// The generator thread sleeps in the kernel (nanosleep with 1us timer slack), not on a Go
				// timer: Go timers are fired by scheduler threads that run on the (possibly saturated)
				// server CPUs. Very short gaps are spun on the dedicated client CPU.
				if gap := next - now; gap > 60_000 {
					ts := syscall.NsecToTimespec(gap - 40_000)
					syscall.Nanosleep(&ts, nil)
				}
				continue
			}
			for next <= now && next < mEnd {
				j := &job{tEnq: next, m: next >= mStart}
				if cfg.perReq {
					j.onDone = func() { record(j); inflight.Add(-1) }
				} else {
					j.onDone = func() { record(j) }
				}
				if cfg.search {
					j.q = make([]float32, s.ix.dim)
					s.ix.MakeQuery(r, j.q)
				}
				if j.m {
					issued.Add(1)
					lagSum.Add(s.now() - next)
				}
				s.app.enq.Add(1)
				if !dispatch(j) && j.m {
					dropped.Add(1)
				}
				next += int64(r.ExpFloat64() / cfg.rate * 1e9)
			}
		}
	}()
	var phase atomic.Int32
	tsDone := make(chan struct{})
	go s.tsLoop(dev, tw, &phase, &gstop, tsDone)

	time.Sleep(cfg.warmup)
	s.measuring.Store(true)
	a := takeSnapshot(dev, s.app)
	phase.Store(1)
	time.Sleep(cfg.dur)
	b := takeSnapshot(dev, s.app)
	phase.Store(2)
	for !genDone.Load() {
		time.Sleep(time.Millisecond)
	}
	// let measured requests finish (bounded), then stop
	deadline := time.Now().Add(10 * time.Second)
	for measuredDone.Load() < issued.Load()-dropped.Load() && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	s.measuring.Store(false)
	s.stopping.Store(true)
	if jobs != nil {
		close(jobs)
		wwg.Wait()
	}
	rwg.Wait()
	for t := time.Now(); s.app.inflightIO.Load() > 0 && time.Since(t) < 10*time.Second; {
		time.Sleep(5 * time.Millisecond) // async reads still in flight
	}
	rstop.Store(true)
	<-rdone
	gstop.Store(true)
	<-tsDone
	var all []reqRec
	for i := range shards {
		shardMu[i].Lock()
		all = append(all, shards[i]...)
		shardMu[i].Unlock()
	}
	inc := issued.Load() - dropped.Load() - int64(len(all))
	if inc < 0 {
		inc = 0
	}
	lag := 0.0
	if issued.Load() > 0 {
		lag = float64(lagSum.Load()) / float64(issued.Load()) / 1e6
	}
	return pointResult{recs: all, issued: issued.Load(), dropped: dropped.Load(), incomplete: inc, lagMs: lag, a: a, b: b}
}

// ---------- summary row ----------

func (s *server) summarize(pr pointResult, bg Delta) []string {
	cfg := s.cfg
	d := diff(pr.a, pr.b)
	var tot, que, sea, io, ioop, iodev, iogap []int64
	for _, r := range pr.recs {
		tot = append(tot, r.total)
		que = append(que, r.queue)
		sea = append(sea, r.search)
		io = append(io, r.io)
		iodev = append(iodev, r.ioDev)
		iogap = append(iogap, r.io-r.ioDev)
	}
	ioop = s.allIoLats()
	for _, x := range [][]int64{tot, que, sea, io, ioop, iodev, iogap} {
		sortI64(x)
	}
	reqs := len(tot)
	qps := float64(reqs) / cfg.dur.Seconds()
	offered := math.NaN()
	if cfg.load == "open" {
		offered = float64(pr.issued) / cfg.dur.Seconds()
		qps = d.QPS // completions inside the window (late completions during drain do not count as throughput)
	}
	ms := func(v float64) string { return f3(v / 1e6) }
	us := func(v float64) string { return f1(v / 1e3) }
	cpuPerReq := 0.0
	if qps > 0 {
		cpuPerReq = d.CPUPct / 100 / qps * 1000
	}
	var warn []string
	if d.DevOK && d.OurMBps > 1 && d.DevRMBps < 0.8*d.OurMBps {
		warn = append(warn, "dev_reads_low(page_cache?)")
	}
	if d.BgRMBps > 50 || d.DevWMBps > 50 || bg.DevRMBps > 50 || bg.DevWMBps > 50 {
		warn = append(warn, "background_io")
	}
	if d.OtherCPUPct > 100 || bg.OtherCPUPct > 100 {
		warn = append(warn, "other_cpu_in_container")
	}
	if reqs < 200 {
		warn = append(warn, "few_requests")
	}
	if s.ioErr.Load() > 0 {
		warn = append(warn, "io_errors")
	}
	if pr.dropped > 0 || pr.incomplete > 0 {
		warn = append(warn, "overload(dropped/incomplete)")
	}
	if cfg.load == "open" && pr.lagMs > 0.5 {
		warn = append(warn, "generator_lag")
	}
	b2s := func(b bool) string {
		if b {
			return "1"
		}
		return "0"
	}
	inflation := math.NaN()
	if cfg.search && cfg.searchAlone > 0 {
		inflation = mean(sea) / 1e6 / cfg.searchAlone
	}
	row := []string{cfg.tag, cfg.family, cfg.variant, fmt.Sprint(cfg.rep), fmt.Sprint(cfg.orderIdx), cfg.load, cfg.mode, fmt.Sprint(cfg.workers), fmt.Sprint(cfg.n),
		fnan(cfg.rate, f1), fnan(cfg.capacity, f1), fnan(offered, f1), fmt.Sprint(pr.dropped), fmt.Sprint(pr.incomplete), f3(pr.lagMs),
		fmt.Sprint(cfg.k), fmt.Sprint(cfg.chunk / 1024), fmt.Sprint(cfg.nprobe), fmt.Sprint(cfg.nlist), b2s(cfg.search), f2(cfg.hit), fmt.Sprint(cfg.lruMB), f2(cfg.zipf), f3(d.HitRatio),
		fmt.Sprint(cfg.cpus), strings.ReplaceAll(cfg.cpuset, ",", ";"), fmt.Sprint(cfg.gmp), b2s(cfg.ts), f1(cfg.dur.Seconds()), fmt.Sprint(reqs), f1(qps),
		ms(mean(tot)), ms(pct(tot, .5)), ms(pct(tot, .95)), ms(pct(tot, .99)),
		ms(mean(que)), ms(pct(que, .95)), ms(mean(sea)), ms(pct(sea, .95)), fnan(cfg.searchAlone, f3), fnan(inflation, f2), f2(cfg.qgenUs),
		ms(mean(io)), ms(pct(io, .95)), ms(mean(iodev)), ms(mean(iogap)),
		us(mean(ioop)), us(pct(ioop, .5)), us(pct(ioop, .95)), us(pct(ioop, .99)),
		f1(d.OurIOPS), f1(d.OurMBps), f1(d.CPUPct), f1(d.ClientCPUPct), fmt.Sprint(cfg.cpus * 100), f3(cpuPerReq), fnan(d.CgCPUPct, f1), f2(d.RunqThreads),
		f1(d.SchedMeanUs), f1(d.SchedP99Us), fmt.Sprint(d.Threads),
		f2(d.AvgInflightIO), f2(d.AvgBusy), f2(d.AvgQueued),
		f1(d.DevRIOPS), f1(d.DevRMBps), f3(d.DevRAwaitMs), fnan(ratio(mean(ioop)/1e6, d.DevRAwaitMs), f2), f2(d.DevAqu), f1(d.DevUtilPct), f1(d.BgRMBps), f1(d.DevWMBps),
		f1(d.HostBusyPct), f2(d.HostIowaitPct), f1(d.OtherCPUPct),
		fnan(d.PsiHostIOSome, f2), fnan(d.PsiHostIOFull, f2), fnan(d.PsiHostCPUSome, f2), fnan(d.PsiCgIOSome, f2), fnan(d.PsiCgCPUSome, f2),
		f1(bg.DevRMBps), f1(bg.DevWMBps), f2(qps * mean(tot) / 1e9), fmt.Sprint(s.ioErr.Load()), strings.Join(warn, "|")}
	if len(row) != len(summaryCols) {
		panic(fmt.Sprintf("summary row has %d fields, header %d", len(row), len(summaryCols)))
	}
	label := fmt.Sprintf("n=%-3d w=%-3d", cfg.n, cfg.workers)
	if cfg.load == "open" {
		label = fmt.Sprintf("rate=%6.0f (%.2fx cap)", cfg.rate, cfg.rate/cfg.capacity)
	}
	fmt.Printf("[%s r%d] %s qps=%8.1f p95=%7.2fms | queue %6.2f search %6.2f io %6.2f (calls %5.2f) ms | cpu %5.0f%% (+client %.0f%%) runq %5.1f | ioop p95 %6.0fus dev aqu %5.1f await %.3fms | bg %.0fMB/s %s\n",
		cfg.tag, cfg.rep, label, qps, pct(tot, .95)/1e6, mean(que)/1e6, mean(sea)/1e6, mean(io)/1e6, mean(iodev)/1e6,
		d.CPUPct, d.ClientCPUPct, d.RunqThreads, pct(ioop, .95)/1e3, d.DevAqu, d.DevRAwaitMs, d.BgRMBps, strings.Join(warn, "|"))
	return row
}
