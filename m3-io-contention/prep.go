package main

import (
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

func cmdPrep(args []string) error {
	fs := flag.NewFlagSet("prep", flag.ExitOnError)
	file := fs.String("file", "", "data file path (e.g. ~/m3data/data.bin)")
	sizeGB := fs.Float64("size", 16, "file size in GiB")
	force := fs.Bool("force", false, "overwrite an existing file")
	marginGB := fs.Float64("margin", 10, "free space (GiB) that must remain after writing")
	fs.Parse(args)
	if *file == "" {
		return errors.New("-file is required")
	}
	const blk = 4 << 20
	size := int64(*sizeGB*float64(1<<30)) / blk * blk
	if size < 1<<28 {
		return errors.New("size too small (min 0.25 GiB)")
	}
	if st, err := os.Stat(*file); err == nil && st.Size() == size && !*force {
		fmt.Printf("%s already exists with %.1f GiB, skipping (use -force to rewrite)\n", *file, float64(size)/(1<<30))
		return nil
	}
	dir := filepath.Dir(*file)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	var sfs syscall.Statfs_t
	if err := syscall.Statfs(dir, &sfs); err == nil {
		avail := int64(sfs.Bavail) * int64(sfs.Bsize)
		if size > avail-int64(*marginGB*float64(1<<30)) {
			return fmt.Errorf("not enough free space: need %.1f GiB + %.0f GiB margin, have %.1f GiB", float64(size)/(1<<30), *marginGB, float64(avail)/(1<<30))
		}
	}
	direct := true
	f, err := os.OpenFile(*file, os.O_CREATE|os.O_WRONLY|os.O_TRUNC|syscall.O_DIRECT, 0644)
	if err != nil {
		direct = false
		f, err = os.OpenFile(*file, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
		if err != nil {
			return err
		}
	}
	fmt.Printf("writing %.1f GiB random data to %s (O_DIRECT=%v)\n", float64(size)/(1<<30), *file, direct)
	buf := alignedBuf(blk)
	r := xrng(0x9E3779B97F4A7C15)
	t0 := time.Now()
	for off := int64(0); off < size; off += blk {
		for i := 0; i < blk; i += 8 {
			binary.LittleEndian.PutUint64(buf[i:], r.next())
		}
		if _, err := f.WriteAt(buf, off); err != nil {
			f.Close()
			return fmt.Errorf("write at %d: %v", off, err)
		}
		if (off+blk)%(1<<30) == 0 {
			el := time.Since(t0).Seconds()
			fmt.Printf("  %3.0f GiB  (%.0f MB/s)\n", float64(off+blk)/(1<<30), float64(off+blk)/el/1e6)
		}
	}
	if err := f.Sync(); err != nil {
		return err
	}
	_ = fadviseDontneed(f)
	f.Close()
	fmt.Printf("done in %.1fs\n", time.Since(t0).Seconds())
	return nil
}
