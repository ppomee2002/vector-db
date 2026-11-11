package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"
)

// cmdEnv prints the facts the report needs about the machine, plus who else
// is using CPU / disk right now. Read-only.
func cmdEnv(args []string) error {
	fs := flag.NewFlagSet("env", flag.ExitOnError)
	devF := fs.String("dev", "auto", "block device name")
	file := fs.String("file", "", "data file (optional)")
	sec := fs.Float64("sample", 3, "seconds to sample background load")
	fs.Parse(args)
	dev := resolveDev(*devF)
	host, _ := os.Hostname()
	fmt.Println("== m3bench env ==", time.Now().Format(time.RFC3339))
	fmt.Println("version     :", version)
	fmt.Println("hostname    :", host)
	fmt.Println("kernel      :", sysStr("/proc/sys/kernel/osrelease"))
	fmt.Println("server      :", sysStr("/sys/class/dmi/id/product_name"))
	fmt.Println("cpu model   :", cpuModel())
	fmt.Println("visible cpus:", runtime.NumCPU())
	fmt.Println("cgroup cpu.max   :", sysStr("/sys/fs/cgroup/cpu.max"))
	fmt.Println("cgroup memory.max:", sysStr("/sys/fs/cgroup/memory.max"))
	fmt.Println("mem         :", memInfo())
	if dev != "" {
		b := "/sys/block/" + dev
		fmt.Printf("device      : %s  vendor=%s model=%s rotational=%s scheduler=%s nr_requests=%s max_sectors_kb=%s\n",
			dev, sysStr(b+"/device/vendor"), sysStr(b+"/device/model"), sysStr(b+"/queue/rotational"),
			sysStr(b+"/queue/scheduler"), sysStr(b+"/queue/nr_requests"), sysStr(b+"/queue/max_sectors_kb"))
	}
	if *file != "" {
		if st, err := os.Stat(*file); err == nil {
			fmt.Printf("data file   : %s (%.1f GiB)\n", *file, float64(st.Size())/(1<<30))
		} else {
			fmt.Printf("data file   : %s (missing)\n", *file)
		}
	}
	fmt.Println("gpu         :", gpuBusy())
	if node := devNumaNode(dev); node >= 0 {
		fmt.Printf("ssd numa    : node %d (cpus %s)\n", node, sysStr(fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", node)))
	}
	if m, err := getAffinity(); err == nil {
		fmt.Println("allowed cpus:", fmtCPUList(m.list()))
	}
	fmt.Println("psi io      :", strings.ReplaceAll(sysStr("/proc/pressure/io"), "\n", " | "))
	// processes by user
	me := uidNames()[fmt.Sprint(os.Getuid())]
	cnt := map[string]int{}
	var others []string
	for _, p := range listProcs() {
		cnt[p.user]++
		if p.user != me && p.user != "root" {
			others = append(others, fmt.Sprintf("%s:%s(%d)", p.user, p.comm, p.pid))
		}
	}
	var us []string
	for u, c := range cnt {
		us = append(us, fmt.Sprintf("%s=%d", u, c))
	}
	sort.Strings(us)
	fmt.Println("processes   :", strings.Join(us, " "))
	if len(others) > 0 {
		fmt.Println("OTHER USERS ACTIVE:", strings.Join(others, " "))
	}
	d := background(dev, *sec)
	fmt.Printf("background (%.0fs): device read %.1f MB/s (%.0f IOPS), write %.1f MB/s, aqu-sz %.2f | host CPU busy %.1f%% | other procs in this container %.0f%% of a core | psi io some %.2f%%\n",
		*sec, d.DevRMBps, d.DevRIOPS, d.DevWMBps, d.DevAqu, d.HostBusyPct, d.OtherCPUPct, d.PsiHostIOSome)
	if d.DevRMBps > 50 || d.DevWMBps > 50 || d.PsiHostIOSome > 5 {
		fmt.Println("WARNING: the SSD is busy with someone else's work right now; results will be noisy.")
	}
	return nil
}

func cpuModel() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "?"
	}
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "model name") {
			if i := strings.IndexByte(l, ':'); i > 0 {
				return strings.TrimSpace(l[i+1:])
			}
		}
	}
	return "?"
}

func memInfo() string {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return "?"
	}
	var parts []string
	for _, l := range strings.Split(string(b), "\n") {
		for _, k := range []string{"MemTotal:", "MemAvailable:", "Cached:"} {
			if strings.HasPrefix(l, k) {
				parts = append(parts, strings.Join(strings.Fields(l), " "))
			}
		}
	}
	return strings.Join(parts, ", ")
}
