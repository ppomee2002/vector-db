package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// cmdSearchbench: single-thread cost of the Go IVF search over a grid of
// (nlist, nprobe), used to match the CPU cost of a real vector library.
func cmdSearchbench(args []string) error {
	fs := flag.NewFlagSet("searchbench", flag.ExitOnError)
	nvec := fs.Int("nvec", 500000, "vectors")
	dim := fs.Int("dim", 384, "dimension")
	k := fs.Int("k", 10, "top-k")
	nlistsF := fs.String("nlists", "256,512,1024,2048,4096", "nlist values")
	nprobesF := fs.String("nprobes", "1,2,4,8,16,32,64", "nprobe values")
	q := fs.Int("q", 300, "queries per point")
	cpuset := fs.String("cpuset", "auto", "CPU to run on (one CPU is used)")
	out := fs.String("out", "", "CSV output (optional)")
	fs.Parse(args)
	pinned, err := ensurePinned(1, *cpuset, resolveDev("auto"))
	if err != nil {
		return err
	}
	runtime.GOMAXPROCS(1)
	nlists, err := parseIntList(*nlistsF)
	if err != nil {
		return err
	}
	nprobes, err := parseIntList(*nprobesF)
	if err != nil {
		return err
	}
	var w *os.File
	if *out != "" {
		var nf bool
		w, nf, err = openAppend(*out)
		if err != nil {
			return err
		}
		defer w.Close()
		if nf {
			fmt.Fprintln(w, "nlist,nprobe,ms_per_query,vectors_scanned")
		}
	}
	fmt.Printf("searchbench on CPU %s (single thread), nvec=%d dim=%d\n", pinned, *nvec, *dim)
	fmt.Println("nlist\tnprobe\tms/query")
	for _, nl := range nlists {
		ix := BuildIndex(*nvec, *dim, nl, 0, 1, 1)
		r := rand.New(rand.NewSource(5))
		qv := make([]float32, *dim)
		for _, np := range nprobes {
			if np > nl {
				continue
			}
			s := ix.NewScratch(np, *k)
			for i := 0; i < 20; i++ { // warm
				ix.MakeQuery(r, qv)
				ix.Search(qv, np, *k, s)
			}
			var tot time.Duration
			for i := 0; i < *q; i++ {
				ix.MakeQuery(r, qv)
				t := time.Now()
				ix.Search(qv, np, *k, s)
				tot += time.Since(t)
			}
			msq := tot.Seconds() * 1000 / float64(*q)
			fmt.Printf("%d\t%d\t%.4f\n", nl, np, msq)
			if w != nil {
				fmt.Fprintf(w, "%d,%d,%.5f,%d\n", nl, np, msq, nl+np*(*nvec/nl))
			}
		}
	}
	return nil
}

// cmdQuiet waits until nobody else is loading the SSD / CPUs (or the
// time limit passes) so an experiment does not start on top of someone else's job.
func cmdQuiet(args []string) error {
	fs := flag.NewFlagSet("quiet", flag.ExitOnError)
	devF := fs.String("dev", "auto", "block device")
	maxW := fs.Duration("max", 10*time.Minute, "give up waiting after this long (then continue with a warning)")
	need := fs.Int("need", 3, "consecutive quiet samples required")
	every := fs.Duration("every", 5*time.Second, "sample length")
	rMax := fs.Float64("read-mbps", 30, "max background read MB/s")
	wMax := fs.Float64("write-mbps", 20, "max background write MB/s (includes SSD settling after prep)")
	oMax := fs.Float64("other-cpu", 100, "max CPU%% of other processes in this container")
	fs.Parse(args)
	dev := resolveDev(*devF)
	deadline := time.Now().Add(*maxW)
	ok := 0
	for {
		d := background(dev, every.Seconds())
		gpu := gpuBusy()
		quiet := d.DevRMBps <= *rMax && d.DevWMBps <= *wMax && d.OtherCPUPct <= *oMax
		fmt.Printf("[quiet] read %.1f MB/s write %.1f MB/s aqu %.2f | other CPU %.0f%% | host CPU %.1f%% | GPU %s -> %v\n",
			d.DevRMBps, d.DevWMBps, d.DevAqu, d.OtherCPUPct, d.HostBusyPct, gpu, quiet)
		if quiet {
			ok++
			if ok >= *need {
				fmt.Println("[quiet] server is quiet, starting")
				return nil
			}
		} else {
			ok = 0
		}
		if time.Now().After(deadline) {
			fmt.Println("[quiet] WARNING: still busy after", *maxW, "- continuing anyway (results flagged by background_io)")
			return nil
		}
	}
}

// gpuBusy summarises GPU use by anyone (nvidia-smi), or "n/a".
func gpuBusy() string {
	p, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return "n/a"
	}
	out, err := exec.Command(p, "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return "n/a"
	}
	var parts []string
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		fs := strings.Split(l, ",")
		if len(fs) == 3 {
			parts = append(parts, fmt.Sprintf("gpu%s %s%% %sMiB", strings.TrimSpace(fs[0]), strings.TrimSpace(fs[1]), strings.TrimSpace(fs[2])))
		}
	}
	return strings.Join(parts, ", ")
}
