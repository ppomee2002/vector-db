package main

// selftest: checks on THIS server every assumption the experiment relies on,
// so a problem shows up before a 2-hour run instead of after it.

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"time"
)

type check struct {
	name, status, detail string
	critical             bool
}

func cmdSelftest(args []string) error {
	fs := flag.NewFlagSet("selftest", flag.ExitOnError)
	file := fs.String("file", "", "data file (from prep)")
	devF := fs.String("dev", "auto", "block device")
	cpus := fs.Int("cpus", 7, "server CPUs the experiment will use")
	direct := fs.Bool("direct", true, "O_DIRECT (false only for testing on tmpfs)")
	fs.Parse(args)
	var cs []check
	add := func(name, status, detail string, critical bool) {
		cs = append(cs, check{name, status, detail, critical})
	}
	dev := resolveDev(*devF)

	// 1. data file + O_DIRECT
	f, size, err := openData(*file, *direct)
	if err != nil {
		add("data file / O_DIRECT", "FAIL", err.Error(), true)
	} else {
		add("data file / O_DIRECT", "PASS", fmt.Sprintf("%s, %.1f GiB, O_DIRECT=%v", *file, float64(size)/(1<<30), *direct), true)
	}

	// 2. our reads reach the block device we measure
	if f != nil && dev != "" {
		app := &AppStats{}
		a := takeSnapshot(dev, app)
		end := time.Now().Add(2 * time.Second)
		done := make(chan struct{})
		for g := 0; g < 8; g++ {
			go func(g int) {
				r := rand.New(rand.NewSource(int64(g)))
				b := alignedBuf(4096)
				for time.Now().Before(end) {
					if preadFull(f, b, r.Int63n(size/4096)*4096) == nil {
						app.ioDone.Add(1)
						app.ioBytes.Add(4096)
					}
				}
				done <- struct{}{}
			}(g)
		}
		for g := 0; g < 8; g++ {
			<-done
		}
		d := diff(a, takeSnapshot(dev, app))
		ratio := 0.0
		if d.OurMBps > 0 {
			ratio = d.DevRMBps / d.OurMBps
		}
		st := "PASS"
		if ratio < 0.8 {
			st = "FAIL"
		}
		add("reads reach device "+dev, st, fmt.Sprintf("our %.1f MB/s, device %.1f MB/s (x%.2f), read await %.3f ms, our IOPS %.0f",
			d.OurMBps, d.DevRMBps, ratio, d.DevRAwaitMs, d.OurIOPS), true)
	} else if dev == "" {
		add("block device", "FAIL", "no device found in /proc/diskstats; pass -dev", true)
	}

	// 3. device facts
	if dev != "" {
		b := "/sys/block/" + dev
		node := devNumaNode(dev)
		ns := "unknown"
		if node >= 0 {
			ns = fmt.Sprintf("node %d (cpus %s)", node, sysStr(fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", node)))
		}
		st := "PASS"
		if node < 0 {
			st = "WARN"
		}
		add("device / NUMA", st, fmt.Sprintf("%s %s rotational=%s sched=%s; SSD NUMA: %s", sysStr(b+"/device/vendor"),
			sysStr(b+"/device/model"), sysStr(b+"/queue/rotational"), sysStr(b+"/queue/scheduler"), ns), false)
	}

	// 4. CPU pinning (server set + client CPU), tested on a throw-away thread
	allowed, err := getAffinity()
	if err != nil {
		add("CPU affinity", "FAIL", err.Error(), true)
	} else {
		al := allowed.list()
		chosen := al
		if len(al) > *cpus+1 {
			chosen = chooseCPUs(*cpus+1, al, dev)
		}
		res := make(chan error)
		go func() {
			runtime.LockOSThread() // thread exits with the goroutine
			var m cpuMask
			for _, c := range chosen[:len(chosen)-1] {
				m.set(c)
			}
			if err := setAffinity(m); err != nil {
				res <- err
				return
			}
			var one cpuMask
			one.set(chosen[len(chosen)-1])
			res <- setAffinity(one)
		}()
		if err := <-res; err != nil {
			add("CPU pinning", "FAIL", err.Error(), true)
		} else {
			st := "PASS"
			if len(al) < *cpus+1 {
				st = "WARN"
			}
			add("CPU pinning", st, fmt.Sprintf("allowed %d CPUs; would use server %s + client %d", len(al),
				fmtCPUList(chosen[:len(chosen)-1]), chosen[len(chosen)-1]), true)
		}
	}

	// 5. native AIO (needed by -io async and baseline -engine aio)
	if err := aioAvailable(); err != nil {
		add("native AIO", "WARN", fmt.Sprintf("%v -> async scenarios (S_*) and the aio baseline will be skipped", err), false)
	} else if f != nil {
		a, _ := newAIO(16)
		b := alignedBuf(4096)
		var cb iocb
		fillPread(&cb, f.Fd(), b, 0, 7)
		err := a.submit([]*iocb{&cb})
		ev := make([]ioEvent, 1)
		n := 0
		if err == nil {
			n, err = a.getEvents(ev, 1, 2*time.Second)
		}
		a.destroy()
		if err != nil || n != 1 || ev[0].res != 4096 {
			add("native AIO", "WARN", fmt.Sprintf("read failed (err=%v n=%d res=%d) -> async scenarios unreliable", err, n, ev[0].res), false)
		} else {
			add("native AIO", "PASS", "io_submit/io_getevents read OK; aio-nr/max "+sysStr("/proc/sys/fs/aio-nr")+"/"+sysStr("/proc/sys/fs/aio-max-nr"), false)
		}
	}

	// 6. metrics sources
	var miss []string
	for _, p := range []string{"/proc/pressure/io", "/proc/pressure/cpu", "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/io.pressure", "/proc/self/task/" + fmt.Sprint(syscall.Gettid()) + "/schedstat"} {
		if _, err := os.ReadFile(p); err != nil {
			miss = append(miss, p)
		}
	}
	if len(miss) == 0 {
		add("metric sources", "PASS", "PSI (host+cgroup), cgroup cpu.stat, per-thread schedstat", false)
	} else {
		add("metric sources", "WARN", "unreadable: "+strings.Join(miss, ", ")+" (those columns stay empty)", false)
	}

	// 7. resources and neighbours
	var sfs syscall.Statfs_t
	home, _ := os.UserHomeDir()
	if syscall.Statfs(home, &sfs) == nil {
		free := float64(sfs.Bavail) * float64(sfs.Bsize) / (1 << 30)
		st := "PASS"
		if free < 10 {
			st = "WARN"
		}
		add("free disk (home)", st, fmt.Sprintf("%.0f GiB", free), false)
	}
	add("memory", "PASS", memInfo(), false)
	add("GPU", "INFO", gpuBusy(), false)
	bg := background(dev, 2)
	st := "PASS"
	if bg.DevRMBps > 30 || bg.DevWMBps > 20 || bg.OtherCPUPct > 100 {
		st = "WARN"
	}
	add("background now", st, fmt.Sprintf("device read %.1f MB/s write %.1f MB/s, other procs in container %.0f%% CPU, host CPU %.1f%%",
		bg.DevRMBps, bg.DevWMBps, bg.OtherCPUPct, bg.HostBusyPct), false)
	if p, err := exec.LookPath("python3"); err == nil {
		out, _ := exec.Command(p, "-c", "import sys;print(sys.version.split()[0])").Output()
		add("python3", "PASS", strings.TrimSpace(string(out))+" (run_all.sh needs it; analyze.py also needs pandas+matplotlib, can run on your PC)", false)
	} else {
		add("python3", "FAIL", "run_all.sh needs python3", true)
	}

	fail := 0
	fmt.Println("== m3bench selftest ==")
	for _, c := range cs {
		fmt.Printf("[%-4s] %-26s %s\n", c.status, c.name, c.detail)
		if c.status == "FAIL" && c.critical {
			fail++
		}
	}
	if fail > 0 {
		return fmt.Errorf("%d critical check(s) failed", fail)
	}
	fmt.Println("selftest OK")
	return nil
}
