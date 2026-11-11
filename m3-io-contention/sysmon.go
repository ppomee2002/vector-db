package main

// sysmon: everything iostat / mpstat / pidstat would give us, read straight
// from /proc and /sys so nothing has to be installed and no root is needed.

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/metrics"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

// ---------- block device (/proc/diskstats) ----------

type DiskStat struct {
	reads, rsect, rticks, writes, wsect, wticks, inflight, ioticks, tiq uint64
}

func readDisk(dev string) (DiskStat, bool) {
	f, err := os.Open("/proc/diskstats")
	if err != nil {
		return DiskStat{}, false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fs := strings.Fields(sc.Text())
		if len(fs) < 14 || fs[2] != dev {
			continue
		}
		v := make([]uint64, len(fs))
		for i := 3; i < len(fs); i++ {
			v[i], _ = strconv.ParseUint(fs[i], 10, 64)
		}
		return DiskStat{reads: v[3], rsect: v[5], rticks: v[6], writes: v[7], wsect: v[9], wticks: v[10],
			inflight: v[11], ioticks: v[12], tiq: v[13]}, true
	}
	return DiskStat{}, false
}

// detectDev picks the whole-disk device (not a partition) with the most
// cumulative read traffic: on a single-SSD server that is the SSD.
func detectDev() string {
	ents, _ := os.ReadDir("/sys/block")
	best, bestSect := "", uint64(0)
	for _, e := range ents {
		n := e.Name()
		if strings.HasPrefix(n, "loop") || strings.HasPrefix(n, "ram") || strings.HasPrefix(n, "zram") {
			continue
		}
		if d, ok := readDisk(n); ok && (best == "" || d.rsect > bestSect) {
			best, bestSect = n, d.rsect
		}
	}
	return best
}

func resolveDev(s string) string {
	if s == "" || s == "auto" {
		d := detectDev()
		if d == "" {
			fmt.Fprintln(os.Stderr, "warning: could not detect block device; device metrics disabled")
		}
		return d
	}
	return s
}

func sysStr(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return "?"
	}
	return strings.TrimSpace(string(b))
}

// ---------- CPU ----------

type HostCPU struct{ busy, iowait, total uint64 }

func readHostCPU() HostCPU {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return HostCPU{}
	}
	line := strings.SplitN(string(b), "\n", 2)[0]
	fs := strings.Fields(line)
	var v [10]uint64
	for i := 1; i < len(fs) && i <= 10; i++ {
		v[i-1], _ = strconv.ParseUint(fs[i], 10, 64)
	}
	// user nice system idle iowait irq softirq steal guest guest_nice
	total := v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]
	return HostCPU{busy: total - v[3] - v[4], iowait: v[4], total: total}
}

func selfCPUns() int64 {
	var ru syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &ru) != nil {
		return 0
	}
	return ru.Utime.Nano() + ru.Stime.Nano()
}

// cgroup v2 of this container (includes other users of the same container).
func readCgUsage() (uint64, bool) {
	b, err := os.ReadFile("/sys/fs/cgroup/cpu.stat")
	if err != nil {
		return 0, false
	}
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "usage_usec ") {
			v, _ := strconv.ParseUint(strings.TrimPrefix(l, "usage_usec "), 10, 64)
			return v, true
		}
	}
	return 0, false
}

// readRunqWait: per thread of this process, the time spent runnable but
// waiting for a CPU (/proc/self/task/*/schedstat, 2nd field).
func readRunqWait() map[int]uint64 {
	out := map[int]uint64{}
	ents, err := os.ReadDir("/proc/self/task")
	if err != nil {
		return out
	}
	for _, e := range ents {
		tid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile("/proc/self/task/" + e.Name() + "/schedstat")
		if err != nil {
			continue
		}
		fs := strings.Fields(string(b))
		if len(fs) >= 2 {
			v, _ := strconv.ParseUint(fs[1], 10, 64)
			out[tid] = v
		}
	}
	return out
}

func readThreads() int {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "Threads:") {
			v, _ := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(l, "Threads:")))
			return v
		}
	}
	return 0
}

// otherProcTicks: utime+stime (clock ticks) of every other visible process.
func otherProcTicks() map[int]uint64 {
	self := os.Getpid()
	m := map[int]uint64{}
	ents, _ := os.ReadDir("/proc")
	for _, e := range ents {
		pid, err := strconv.Atoi(e.Name())
		if err != nil || pid == self {
			continue
		}
		b, err := os.ReadFile("/proc/" + e.Name() + "/stat")
		if err != nil {
			continue
		}
		s := string(b)
		i := strings.LastIndexByte(s, ')')
		if i < 0 {
			continue
		}
		fs := strings.Fields(s[i+1:])
		if len(fs) < 13 {
			continue
		}
		u, _ := strconv.ParseUint(fs[11], 10, 64)
		st, _ := strconv.ParseUint(fs[12], 10, 64)
		m[pid] = u + st
	}
	return m
}

type ProcInfo struct {
	pid  int
	user string
	comm string
}

func uidNames() map[string]string {
	m := map[string]string{}
	b, err := os.ReadFile("/etc/passwd")
	if err != nil {
		return m
	}
	for _, l := range strings.Split(string(b), "\n") {
		fs := strings.Split(l, ":")
		if len(fs) > 2 {
			m[fs[2]] = fs[0]
		}
	}
	return m
}

func listProcs() []ProcInfo {
	names := uidNames()
	var out []ProcInfo
	ents, _ := os.ReadDir("/proc")
	for _, e := range ents {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(filepath.Join("/proc", e.Name(), "status"))
		if err != nil {
			continue
		}
		var uid, comm string
		for _, l := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(l, "Uid:") {
				if fs := strings.Fields(l); len(fs) > 1 {
					uid = fs[1]
				}
			} else if strings.HasPrefix(l, "Name:") {
				comm = strings.TrimSpace(strings.TrimPrefix(l, "Name:"))
			}
		}
		u := names[uid]
		if u == "" {
			u = uid
		}
		out = append(out, ProcInfo{pid, u, comm})
	}
	return out
}

// ---------- PSI ----------

type PSI struct {
	some, full uint64
	ok         bool
}

func readPSI(path string) PSI {
	b, err := os.ReadFile(path)
	if err != nil {
		return PSI{}
	}
	p := PSI{ok: true}
	for _, l := range strings.Split(string(b), "\n") {
		i := strings.Index(l, "total=")
		if i < 0 {
			continue
		}
		v, _ := strconv.ParseUint(strings.TrimSpace(l[i+6:]), 10, 64)
		if strings.HasPrefix(l, "some") {
			p.some = v
		} else if strings.HasPrefix(l, "full") {
			p.full = v
		}
	}
	return p
}

// ---------- Go scheduler latency (goroutine runnable -> running) ----------

const schedMetric = "/sched/latencies:seconds"

func readSched() ([]uint64, []float64) {
	s := []metrics.Sample{{Name: schedMetric}}
	metrics.Read(s)
	if s[0].Value.Kind() != metrics.KindFloat64Histogram {
		return nil, nil
	}
	h := s[0].Value.Float64Histogram()
	c := make([]uint64, len(h.Counts))
	copy(c, h.Counts)
	return c, h.Buckets
}

func histStats(a, b []uint64, buckets []float64) (meanUs, p99Us float64) {
	if len(a) != len(b) || len(b) == 0 {
		return 0, 0
	}
	d := make([]uint64, len(b))
	var n uint64
	var sum float64
	for i := range b {
		d[i] = b[i] - a[i]
		n += d[i]
		lo, hi := buckets[i], buckets[i+1]
		if math.IsInf(lo, -1) {
			lo = 0
		}
		if math.IsInf(hi, 1) {
			hi = lo
		}
		sum += float64(d[i]) * (lo + hi) / 2
	}
	if n == 0 {
		return 0, 0
	}
	target := uint64(math.Ceil(0.99 * float64(n)))
	var c uint64
	for i := range d {
		c += d[i]
		if c >= target {
			hi := buckets[i+1]
			if math.IsInf(hi, 1) {
				hi = buckets[i]
			}
			p99Us = hi * 1e6
			break
		}
	}
	return sum / float64(n) * 1e6, p99Us
}

// ---------- app-level counters (filled by the workload) ----------

type AppStats struct {
	enq, started, reqDone, ioDone, ioBytes, hits atomic.Int64
	inflightIO, busy                             atomic.Int64
	// gauge sampler (100 Hz)
	gN, gInflight, gBusy, gQueued atomic.Int64
}

func (a *AppStats) gaugeLoop(stop *atomic.Bool) {
	pinClient()
	t := time.NewTicker(10 * time.Millisecond)
	defer t.Stop()
	for !stop.Load() {
		<-t.C
		a.gN.Add(1)
		a.gInflight.Add(a.inflightIO.Load())
		a.gBusy.Add(a.busy.Load())
		a.gQueued.Add(a.enq.Load() - a.started.Load())
	}
}

// ---------- snapshots and deltas ----------

type Snapshot struct {
	t                          time.Time
	disk                       DiskStat
	diskOK                     bool
	host                       HostCPU
	selfNs                     int64
	cg                         uint64
	cgOK                       bool
	runq                       map[int]uint64
	threads                    int
	hIO, hCPU, cIO, cCPU       PSI
	other, client              map[int]uint64
	schedC                     []uint64
	schedB                     []float64
	reqs, ios, iob, enq, hits  int64
	gN, gInflight, gBusy, gQue int64
}

func takeSnapshot(dev string, app *AppStats) Snapshot {
	s := Snapshot{t: time.Now()}
	if dev != "" {
		s.disk, s.diskOK = readDisk(dev)
	}
	s.host = readHostCPU()
	s.selfNs = selfCPUns()
	s.cg, s.cgOK = readCgUsage()
	s.runq = readRunqWait()
	s.threads = readThreads()
	s.hIO = readPSI("/proc/pressure/io")
	s.hCPU = readPSI("/proc/pressure/cpu")
	s.cIO = readPSI("/sys/fs/cgroup/io.pressure")
	s.cCPU = readPSI("/sys/fs/cgroup/cpu.pressure")
	s.other = otherProcTicks()
	s.client = clientTicks()
	s.schedC, s.schedB = readSched()
	if app != nil {
		s.reqs, s.ios, s.iob = app.reqDone.Load(), app.ioDone.Load(), app.ioBytes.Load()
		s.enq, s.hits = app.enq.Load(), app.hits.Load()
		s.gN, s.gInflight, s.gBusy, s.gQue = app.gN.Load(), app.gInflight.Load(), app.gBusy.Load(), app.gQueued.Load()
	}
	return s
}

type Delta struct {
	Dt                                           float64
	QPS, OurIOPS, OurMBps, Offered, HitRatio     float64
	CPUPct, CgCPUPct, HostBusyPct, HostIowaitPct float64
	OtherCPUPct, RunqThreads, ClientCPUPct       float64
	DevRIOPS, DevRMBps, DevWMBps, DevRAwaitMs    float64
	DevAqu, DevUtilPct, BgRMBps                  float64
	PsiHostIOSome, PsiHostIOFull, PsiHostCPUSome float64
	PsiCgIOSome, PsiCgCPUSome                    float64
	SchedMeanUs, SchedP99Us                      float64
	AvgInflightIO, AvgBusy, AvgQueued            float64
	Threads                                      int
	DevOK                                        bool
}

func psiPct(a, b PSI, dt float64, full bool) float64 {
	if !a.ok || !b.ok {
		return math.NaN()
	}
	if full {
		return float64(b.full-a.full) / (dt * 1e6) * 100
	}
	return float64(b.some-a.some) / (dt * 1e6) * 100
}

func diff(a, b Snapshot) Delta {
	dt := b.t.Sub(a.t).Seconds()
	d := Delta{Dt: dt, Threads: b.threads}
	if dt <= 0 {
		return d
	}
	d.QPS = float64(b.reqs-a.reqs) / dt
	d.Offered = float64(b.enq-a.enq) / dt
	if tot := float64(b.ios - a.ios + b.hits - a.hits); tot > 0 {
		d.HitRatio = float64(b.hits-a.hits) / tot
	}
	d.OurIOPS = float64(b.ios-a.ios) / dt
	d.OurMBps = float64(b.iob-a.iob) / dt / 1e6
	d.CPUPct = float64(b.selfNs-a.selfNs) / (dt * 1e9) * 100
	if a.cgOK && b.cgOK {
		d.CgCPUPct = float64(b.cg-a.cg) / (dt * 1e6) * 100
	} else {
		d.CgCPUPct = math.NaN()
	}
	if t := float64(b.host.total - a.host.total); t > 0 {
		d.HostBusyPct = float64(b.host.busy-a.host.busy) / t * 100
		d.HostIowaitPct = float64(b.host.iowait-a.host.iowait) / t * 100
	}
	var ot uint64
	for pid, v := range b.other {
		if pv, ok := a.other[pid]; ok {
			if v >= pv {
				ot += v - pv
			}
		} else {
			ot += v
		}
	}
	d.OtherCPUPct = float64(ot) / dt // ticks are 1/100 s -> percent of one core
	var ct uint64
	for tid, v := range b.client {
		if pv, ok := a.client[tid]; ok && v >= pv {
			ct += v - pv
		}
	}
	d.ClientCPUPct = float64(ct) / dt
	d.CPUPct = math.Max(0, d.CPUPct-d.ClientCPUPct) // server CPU only
	var rq uint64
	for tid, v := range b.runq { // only threads alive at both ends (threads come and go)
		if pv, ok := a.runq[tid]; ok && v >= pv {
			rq += v - pv
		}
	}
	d.RunqThreads = float64(rq) / (dt * 1e9)
	if a.diskOK && b.diskOK {
		d.DevOK = true
		dr := float64(b.disk.reads - a.disk.reads)
		d.DevRIOPS = dr / dt
		d.DevRMBps = float64(b.disk.rsect-a.disk.rsect) * 512 / dt / 1e6
		d.DevWMBps = float64(b.disk.wsect-a.disk.wsect) * 512 / dt / 1e6
		if dr > 0 {
			d.DevRAwaitMs = float64(b.disk.rticks-a.disk.rticks) / dr
		}
		d.DevAqu = float64(b.disk.tiq-a.disk.tiq) / (dt * 1000)
		d.DevUtilPct = float64(b.disk.ioticks-a.disk.ioticks) / (dt * 1000) * 100
		d.BgRMBps = math.Max(0, d.DevRMBps-d.OurMBps)
	}
	d.PsiHostIOSome = psiPct(a.hIO, b.hIO, dt, false)
	d.PsiHostIOFull = psiPct(a.hIO, b.hIO, dt, true)
	d.PsiHostCPUSome = psiPct(a.hCPU, b.hCPU, dt, false)
	d.PsiCgIOSome = psiPct(a.cIO, b.cIO, dt, false)
	d.PsiCgCPUSome = psiPct(a.cCPU, b.cCPU, dt, false)
	d.SchedMeanUs, d.SchedP99Us = histStats(a.schedC, b.schedC, b.schedB)
	if n := float64(b.gN - a.gN); n > 0 {
		d.AvgInflightIO = float64(b.gInflight-a.gInflight) / n
		d.AvgBusy = float64(b.gBusy-a.gBusy) / n
		d.AvgQueued = float64(b.gQue-a.gQue) / n
	}
	return d
}

// background measures for `sec` seconds what everybody else is doing.
func background(dev string, sec float64) Delta {
	a := takeSnapshot(dev, nil)
	time.Sleep(time.Duration(sec * float64(time.Second)))
	b := takeSnapshot(dev, nil)
	return diff(a, b)
}
