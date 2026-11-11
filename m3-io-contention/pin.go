package main

// CPU pinning. The team may use 8 cores. Limiting Go with GOMAXPROCS=8 is
// wrong for this experiment: a goroutine blocked in pread() keeps its P, so
// with 8 Ps the reads themselves get serialised by the Go runtime and the
// app can never put more than ~8 reads on the SSD. Instead we restrict the
// whole process to 8 CPUs with sched_setaffinity (no root needed) and give
// Go many Ps, so the program behaves like an ordinary multi-threaded server
// on an 8-core machine: CPU contention shows up as OS run-queue wait.

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const maskWords = 16 // 1024 CPUs

type cpuMask [maskWords]uint64

func getAffinity() (cpuMask, error) {
	var m cpuMask
	_, _, e := syscall.RawSyscall(syscall.SYS_SCHED_GETAFFINITY, 0, uintptr(len(m)*8), uintptr(unsafe.Pointer(&m[0])))
	if e != 0 {
		return m, e
	}
	return m, nil
}

func setAffinity(m cpuMask) error {
	_, _, e := syscall.RawSyscall(syscall.SYS_SCHED_SETAFFINITY, 0, uintptr(len(m)*8), uintptr(unsafe.Pointer(&m[0])))
	if e != 0 {
		return e
	}
	return nil
}

func (m cpuMask) has(c int) bool { return c >= 0 && c < maskWords*64 && m[c/64]&(1<<(c%64)) != 0 }
func (m *cpuMask) set(c int)     { m[c/64] |= 1 << (c % 64) }
func (m cpuMask) list() []int {
	var out []int
	for c := 0; c < maskWords*64; c++ {
		if m.has(c) {
			out = append(out, c)
		}
	}
	return out
}

func parseCPUList(s string) []int {
	var out []int
	for _, p := range strings.Split(strings.TrimSpace(s), ",") {
		if p == "" {
			continue
		}
		if i := strings.IndexByte(p, '-'); i > 0 {
			a, e1 := strconv.Atoi(p[:i])
			b, e2 := strconv.Atoi(p[i+1:])
			if e1 == nil && e2 == nil {
				for c := a; c <= b; c++ {
					out = append(out, c)
				}
			}
		} else if c, err := strconv.Atoi(p); err == nil {
			out = append(out, c)
		}
	}
	return out
}

func fmtCPUList(cs []int) string {
	var parts []string
	for i := 0; i < len(cs); {
		j := i
		for j+1 < len(cs) && cs[j+1] == cs[j]+1 {
			j++
		}
		if j > i {
			parts = append(parts, fmt.Sprintf("%d-%d", cs[i], cs[j]))
		} else {
			parts = append(parts, fmt.Sprint(cs[i]))
		}
		i = j + 1
	}
	return strings.Join(parts, ",")
}

// devNumaNode walks up from the block device to the PCI device that owns it.
func devNumaNode(dev string) int {
	p, err := filepath.EvalSymlinks("/sys/block/" + dev + "/device")
	if err != nil {
		return -1
	}
	for i := 0; i < 12 && p != "/" && p != "."; i++ {
		if b, err := os.ReadFile(p + "/numa_node"); err == nil {
			if n, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil && n >= 0 {
				return n
			}
		}
		p = filepath.Dir(p)
	}
	return -1
}

func perCPUBusy(sec float64) map[int]float64 {
	read := func() map[int][2]uint64 {
		m := map[int][2]uint64{}
		b, _ := os.ReadFile("/proc/stat")
		for _, l := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(l, "cpu") || strings.HasPrefix(l, "cpu ") {
				continue
			}
			fs := strings.Fields(l)
			c, err := strconv.Atoi(strings.TrimPrefix(fs[0], "cpu"))
			if err != nil || len(fs) < 9 {
				continue
			}
			var tot, idle uint64
			for i := 1; i <= 8; i++ {
				v, _ := strconv.ParseUint(fs[i], 10, 64)
				tot += v
				if i == 4 || i == 5 {
					idle += v
				}
			}
			m[c] = [2]uint64{tot - idle, tot}
		}
		return m
	}
	a := read()
	time.Sleep(time.Duration(sec * float64(time.Second)))
	b := read()
	out := map[int]float64{}
	for c, v := range b {
		if u, ok := a[c]; ok && v[1] > u[1] {
			out[c] = float64(v[0]-u[0]) / float64(v[1]-u[1])
		}
	}
	return out
}

// chooseCPUs picks n allowed CPUs: on the SSD's NUMA node if known, the least
// busy ones first, at most one hyper-thread per physical core when possible.
func chooseCPUs(n int, allowed []int, dev string) []int {
	cand := allowed
	if node := devNumaNode(dev); node >= 0 {
		local := map[int]bool{}
		for _, c := range parseCPUList(sysStr(fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", node))) {
			local[c] = true
		}
		var lc []int
		for _, c := range allowed {
			if local[c] {
				lc = append(lc, c)
			}
		}
		if len(lc) >= n {
			cand = lc
		}
	}
	busy := perCPUBusy(1)
	sort.SliceStable(cand, func(i, j int) bool { return busy[cand[i]] < busy[cand[j]] })
	used := map[int]bool{}
	var out []int
	for pass := 0; pass < 2 && len(out) < n; pass++ {
		for _, c := range cand {
			if len(out) == n {
				break
			}
			if used[c] {
				continue
			}
			sib := parseCPUList(sysStr(fmt.Sprintf("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c)))
			clash := false
			for _, s := range sib {
				if s != c && used[s] {
					clash = true
				}
			}
			if pass == 0 && clash {
				continue
			}
			used[c] = true
			out = append(out, c)
		}
	}
	sort.Ints(out)
	return out
}

// ensurePinned restricts the process to `cpus` CPUs and re-executes itself so
// every thread inherits the mask. spec: "auto", "none" or an explicit list.
// Returns the CPU list in effect.
func ensurePinned(cpus int, spec, dev string) (string, error) {
	return ensurePinnedClient(cpus, spec, "none", dev)
}

// ensurePinnedClient also reserves one extra CPU for the load generator and
// samplers. clientSpec: "auto" (pick one more CPU), "none", or a CPU number.
func ensurePinnedClient(cpus int, spec, clientSpec, dev string) (string, error) {
	if v := os.Getenv("M3_PINNED"); v != "" {
		return v, nil
	}
	if spec == "none" {
		return "none", nil
	}
	allowed, err := getAffinity()
	if err != nil {
		return "", fmt.Errorf("sched_getaffinity: %v", err)
	}
	al := allowed.list()
	var chosen []int
	client := -1
	if spec == "auto" || spec == "" {
		want := cpus
		if clientSpec == "auto" {
			want++
		}
		if len(al) <= want {
			chosen = append([]int(nil), al...)
		} else {
			chosen = chooseCPUs(want, al, dev)
		}
		if clientSpec == "auto" && len(chosen) > 1 {
			client = chosen[len(chosen)-1]
			chosen = chosen[:len(chosen)-1]
		}
	} else {
		for _, c := range parseCPUList(spec) {
			if allowed.has(c) {
				chosen = append(chosen, c)
			}
		}
		if len(chosen) == 0 {
			return "", fmt.Errorf("none of the CPUs in -cpuset %q are allowed (allowed: %s)", spec, fmtCPUList(al))
		}
	}
	if clientSpec != "auto" && clientSpec != "none" && clientSpec != "" {
		if c, err := strconv.Atoi(clientSpec); err == nil && allowed.has(c) {
			client = c
		}
	}
	var m cpuMask
	for _, c := range chosen {
		m.set(c)
	}
	list := fmtCPUList(chosen)
	if client >= 0 {
		for _, c := range chosen {
			if c == client {
				return "", fmt.Errorf("client CPU %d is also a server CPU", client)
			}
		}
		os.Setenv("M3_CLIENT_CPU", strconv.Itoa(client))
	}
	runtime.LockOSThread()
	if err := setAffinity(m); err != nil {
		return "", fmt.Errorf("sched_setaffinity(%s): %v", list, err)
	}
	os.Setenv("M3_PINNED", list)
	exe, err := os.Executable()
	if err != nil {
		exe = "/proc/self/exe"
	}
	fmt.Fprintf(os.Stderr, "pinned to CPUs %s (%d of %d allowed), client CPU %d; re-executing\n", list, len(chosen), len(al), client)
	return "", syscall.Exec(exe, os.Args, os.Environ())
}

// effectiveCPUs counts the CPUs of the current affinity mask.
func effectiveCPUs() int {
	m, err := getAffinity()
	if err != nil {
		return runtime.NumCPU()
	}
	return len(m.list())
}

// cmdCPUs prints the CPU list `auto` would choose, so a whole experiment can
// use one fixed set (run_all.sh calls it once).
func cmdCPUs(args []string) error {
	fs := flag.NewFlagSet("cpus", flag.ExitOnError)
	n := fs.Int("cpus", 8, "how many CPUs")
	devF := fs.String("dev", "auto", "block device (its NUMA node is preferred)")
	fs.Parse(args)
	allowed, err := getAffinity()
	if err != nil {
		return err
	}
	al := allowed.list()
	if len(al) <= *n {
		fmt.Println(fmtCPUList(al))
		return nil
	}
	fmt.Println(fmtCPUList(chooseCPUs(*n, al, resolveDev(*devF))))
	return nil
}

// ---------- client CPU: load generator + samplers run on their own CPU ----------
//
// The server threads are pinned to the experiment's CPUs. The open-loop
// arrival generator and the samplers are pinned to one extra CPU, so (a) a
// saturated server cannot delay arrivals and (b) measuring costs no server CPU.
// Their CPU time is tracked per thread and subtracted from the server's.

var clientTids struct {
	sync.Mutex
	m map[int]bool
}

func clientCPU() int {
	v := os.Getenv("M3_CLIENT_CPU")
	if v == "" {
		return -1
	}
	c, err := strconv.Atoi(v)
	if err != nil {
		return -1
	}
	return c
}

// pinClient locks the calling goroutine to its own OS thread on the client
// CPU. The goroutine must not return to normal scheduling afterwards (the
// thread is discarded when the goroutine exits).
func pinClient() {
	c := clientCPU()
	if c < 0 {
		return
	}
	runtime.LockOSThread()
	var m cpuMask
	m.set(c)
	if err := setAffinity(m); err != nil {
		return
	}
	tid := syscall.Gettid()
	clientTids.Lock()
	if clientTids.m == nil {
		clientTids.m = map[int]bool{}
	}
	clientTids.m[tid] = true
	clientTids.Unlock()
}

// clientTicks: utime+stime (1/100 s) of live client threads.
func clientTicks() map[int]uint64 {
	clientTids.Lock()
	defer clientTids.Unlock()
	out := map[int]uint64{}
	for tid := range clientTids.m {
		b, err := os.ReadFile(fmt.Sprintf("/proc/self/task/%d/stat", tid))
		if err != nil {
			delete(clientTids.m, tid)
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
		out[tid] = u + st
	}
	return out
}
