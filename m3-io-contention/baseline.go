package main

// baseline: fio-like random reads (psync, O_DIRECT) at increasing queue
// depth. Gives the SSD's own latency/IOPS curve, the yardstick for
// "is the SSD saturated?" in the RAG runs.

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func cmdBaseline(args []string) error {
	fs := flag.NewFlagSet("baseline", flag.ExitOnError)
	file := fs.String("file", "", "data file")
	devF := fs.String("dev", "auto", "block device name in /proc/diskstats (auto = detect)")
	bsF := fs.String("bs", "4k,64k", "read sizes")
	qdF := fs.String("qd", "1,2,4,8,16,32,64,128,256", "queue depths (concurrent synchronous readers)")
	dur := fs.Duration("dur", 10*time.Second, "measurement time per point")
	cpus := fs.Int("cpus", 7, "server CPUs the process is pinned to (team limit 800% = 7 server + 1 client)")
	cpuset := fs.String("cpuset", "auto", "auto (least-busy CPUs on the SSD's NUMA node), none, or a list like 0-7")
	gmp := fs.Int("gomaxprocs", 128, "Go Ps; keep it well above -cpus so blocked reads do not stall runnable work")
	out := fs.String("out", "results/baseline.csv", "output CSV (appended)")
	direct := fs.Bool("direct", true, "use O_DIRECT (bypass page cache)")
	engine := fs.String("engine", "psync", "psync (qd threads, one blocking read each) | aio (one thread keeps qd native-AIO reads in flight)")
	fs.Parse(args)
	dev := resolveDev(*devF)
	pinned, err := ensurePinned(*cpus, *cpuset, dev)
	if err != nil {
		return err
	}
	if pinned == "none" {
		*gmp = *cpus
	}
	runtime.GOMAXPROCS(*gmp)
	debug.SetMaxThreads(30000)
	sizes, err := parseSizeList(*bsF)
	if err != nil {
		return err
	}
	qds, err := parseIntList(*qdF)
	if err != nil {
		return err
	}
	f, size, err := openData(*file, *direct)
	if err != nil {
		return err
	}
	defer f.Close()
	o, newFile, err := openAppend(*out)
	if err != nil {
		return err
	}
	defer o.Close()
	if newFile {
		fmt.Fprintln(o, "engine,bs_kb,qd,ios,iops,mbps,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,cpu_pct,dev_r_iops,dev_r_mbps,dev_aqu,dev_r_await_ms,bg_r_mbps,dev_w_mbps,other_cpu_pct,psi_io_some_pct,warn")
	}
	if *engine != "psync" && *engine != "aio" {
		return fmt.Errorf("bad -engine %q", *engine)
	}
	if *engine == "aio" {
		if err := aioAvailable(); err != nil {
			return fmt.Errorf("native AIO unavailable: %v", err)
		}
	}
	fmt.Printf("baseline on %s (engine=%s, dev=%s, O_DIRECT=%v, CPUs=%s, GOMAXPROCS=%d)\n", *file, *engine, dev, *direct, pinned, *gmp)
	fmt.Println("bs_kb\tqd\tiops\tMB/s\tmean_us\tp99_us\tcpu%\tdev_aqu\tbg_MB/s")
	for _, bs := range sizes {
		slots := size / bs
		for _, qd := range qds {
			var stop, meas atomic.Bool
			app := &AppStats{}
			lats := make([][]int64, qd)
			var wg sync.WaitGroup
			if *engine == "aio" {
				wg.Add(1)
				go func() {
					defer wg.Done()
					lats[0] = aioLoop(f, bs, slots, qd, &stop, &meas, app)
				}()
			}
			for i := 0; i < qd && *engine == "psync"; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					rng := rand.New(rand.NewSource(int64(i)*7919 + bs))
					b := alignedBuf(int(bs))
					for !stop.Load() {
						off := rng.Int63n(slots) * bs
						t0 := time.Now()
						if err := preadFull(f, b, off); err != nil {
							fmt.Fprintln(os.Stderr, "read error:", err)
							return
						}
						if meas.Load() {
							lats[i] = append(lats[i], int64(time.Since(t0)))
						}
						app.ioDone.Add(1)
						app.ioBytes.Add(bs)
					}
				}(i)
			}
			time.Sleep(500 * time.Millisecond) // ramp
			a := takeSnapshot(dev, app)
			meas.Store(true)
			time.Sleep(*dur)
			meas.Store(false)
			b := takeSnapshot(dev, app)
			stop.Store(true)
			wg.Wait()
			d := diff(a, b)
			var all []int64
			for _, l := range lats {
				all = append(all, l...)
			}
			sortI64(all)
			var warn []string
			if d.DevOK && d.OurMBps > 1 && d.DevRMBps < 0.8*d.OurMBps {
				warn = append(warn, "dev_reads_low(page_cache?)")
			}
			if d.BgRMBps > 50 || d.DevWMBps > 50 {
				warn = append(warn, "background_io")
			}
			line := []string{*engine, fmt.Sprint(bs / 1024), fmt.Sprint(qd), fmt.Sprint(len(all)), f1(d.OurIOPS), f1(d.OurMBps),
				f1(mean(all) / 1e3), f1(pct(all, .5) / 1e3), f1(pct(all, .95) / 1e3), f1(pct(all, .99) / 1e3),
				f1(d.CPUPct), f1(d.DevRIOPS), f1(d.DevRMBps), f2(d.DevAqu), f3(d.DevRAwaitMs), f1(d.BgRMBps), f1(d.DevWMBps),
				f1(d.OtherCPUPct), f2(d.PsiHostIOSome), strings.Join(warn, "|")}
			fmt.Fprintln(o, strings.Join(line, ","))
			fmt.Printf("%d\t%d\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.1f\t%.0f %s\n", bs/1024, qd, d.OurIOPS, d.OurMBps,
				mean(all)/1e3, pct(all, .99)/1e3, d.CPUPct, d.DevAqu, d.BgRMBps, strings.Join(warn, "|"))
			time.Sleep(time.Second)
		}
	}
	return nil
}

func openAppend(path string) (*os.File, bool, error) {
	if i := strings.LastIndexByte(path, '/'); i > 0 {
		if err := os.MkdirAll(path[:i], 0755); err != nil {
			return nil, false, err
		}
	}
	_, err := os.Stat(path)
	newFile := os.IsNotExist(err)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	return f, newFile, err
}

// aioLoop keeps qd native-AIO reads in flight from a single thread and returns
// the latencies observed while meas was set.
func aioLoop(f *os.File, bs, slots int64, qd int, stop, meas *atomic.Bool, app *AppStats) []int64 {
	runtime.LockOSThread()
	a, err := newAIO(qd * 2)
	if err != nil {
		fmt.Fprintln(os.Stderr, "io_setup:", err)
		return nil
	}
	defer a.destroy()
	rng := rand.New(rand.NewSource(bs*31 + int64(qd)))
	bufs := make([][]byte, qd)
	cbs := make([]iocb, qd)
	ptr := make([]*iocb, 1)
	tsub := make([]time.Time, qd)
	var lats []int64
	sub := func(i int) bool {
		fillPread(&cbs[i], f.Fd(), bufs[i], rng.Int63n(slots)*bs, uint64(i))
		ptr[0] = &cbs[i]
		tsub[i] = time.Now()
		if err := a.submit(ptr); err != nil {
			fmt.Fprintln(os.Stderr, "io_submit:", err)
			return false
		}
		return true
	}
	for i := 0; i < qd; i++ {
		bufs[i] = alignedBuf(int(bs))
		if !sub(i) {
			return lats
		}
	}
	ev := make([]ioEvent, qd)
	inflight := qd
	for inflight > 0 {
		n, err := a.getEvents(ev, 1, 50*time.Millisecond)
		if err != nil {
			fmt.Fprintln(os.Stderr, "io_getevents:", err)
			return lats
		}
		now := time.Now()
		for _, e := range ev[:n] {
			i := int(e.data)
			inflight--
			if meas.Load() {
				lats = append(lats, int64(now.Sub(tsub[i])))
			}
			app.ioDone.Add(1)
			app.ioBytes.Add(bs)
			if !stop.Load() && sub(i) {
				inflight++
			}
		}
	}
	return lats
}
