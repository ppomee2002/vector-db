package main

import (
	"fmt"
	"os"
)

const version = "m3bench 0.3 (M3: I/O waiting & contention in RAG context retrieval)"

func usage() {
	fmt.Fprint(os.Stderr, `usage: m3bench <command> [flags]

commands:
  selftest  check every assumption of the experiment on this server (run this first)
  env       print server/environment info and current background load (read-only)
  cpus      print the CPU set the experiment will be pinned to (least-busy CPUs near the SSD)
  prep      create the data file (random bytes, written with O_DIRECT)
  quiet     wait until the SSD/CPUs are not busy with someone else's work
  searchbench  single-thread cost of the Go vector search (for calibration against FAISS)
  baseline  raw random-read sweep over queue depths (fio-like, psync) -> SSD reference curve
  run       RAG retrieval workload: vector search (CPU) + chunk reads (SSD), closed-loop clients
  version

run "m3bench <command> -h" for flags.
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "env":
		err = cmdEnv(os.Args[2:])
	case "cpus":
		err = cmdCPUs(os.Args[2:])
	case "selftest":
		err = cmdSelftest(os.Args[2:])
	case "searchbench":
		err = cmdSearchbench(os.Args[2:])
	case "quiet":
		err = cmdQuiet(os.Args[2:])
	case "prep":
		err = cmdPrep(os.Args[2:])
	case "baseline":
		err = cmdBaseline(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println(version)
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
