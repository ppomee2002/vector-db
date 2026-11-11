package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

const align = 4096

// alignedBuf returns a 4KiB-aligned slice (required by O_DIRECT).
func alignedBuf(n int) []byte {
	b := make([]byte, n+align)
	p := uintptr(unsafe.Pointer(&b[0]))
	o := int((align - p%align) % align)
	return b[o : o+n : o+n]
}

// fadviseDontneed drops the file's pages from the page cache (no root needed).
func fadviseDontneed(f *os.File) error {
	_, _, e := syscall.Syscall6(syscall.SYS_FADVISE64, f.Fd(), 0, 0, 4 /*POSIX_FADV_DONTNEED*/, 0, 0)
	if e != 0 {
		return e
	}
	return nil
}

// openData opens the data file for reading (optionally O_DIRECT) and checks
// that one aligned read works, so a filesystem without O_DIRECT fails early.
func openData(path string, direct bool) (*os.File, int64, error) {
	flag := os.O_RDONLY
	if direct {
		flag |= syscall.O_DIRECT
	}
	f, err := os.OpenFile(path, flag, 0)
	if err != nil {
		if direct && errors.Is(err, syscall.EINVAL) {
			return nil, 0, fmt.Errorf("O_DIRECT not supported on this filesystem (%v); rerun with -direct=false (results will then include page-cache hits)", err)
		}
		return nil, 0, err
	}
	st, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	if st.Size() < 1<<26 {
		return nil, 0, fmt.Errorf("data file too small (%d bytes); run `m3bench prep` first", st.Size())
	}
	b := alignedBuf(align)
	if _, err := f.ReadAt(b, 0); err != nil && err != io.EOF {
		return nil, 0, fmt.Errorf("test read failed (O_DIRECT=%v): %v", direct, err)
	}
	if !direct {
		_ = fadviseDontneed(f)
	}
	return f, st.Size(), nil
}

func preadFull(f *os.File, b []byte, off int64) error {
	got := 0
	for got < len(b) {
		k, err := f.ReadAt(b[got:], off+int64(got))
		got += k
		if err != nil {
			if err == io.EOF && got > 0 {
				return nil
			}
			return err
		}
	}
	return nil
}

func sortI64(a []int64) { sort.Slice(a, func(i, j int) bool { return a[i] < a[j] }) }

// pct returns the q-quantile of a sorted slice (nearest rank).
func pct(sorted []int64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q*float64(len(sorted))+0.5) - 1
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return float64(sorted[i])
}

func mean(a []int64) float64 {
	if len(a) == 0 {
		return 0
	}
	var s float64
	for _, x := range a {
		s += float64(x)
	}
	return s / float64(len(a))
}

func parseIntList(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("bad number %q", p)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, errors.New("empty list")
	}
	return out, nil
}

// parseSize parses "4k", "64K", "1m", "4096".
func parseSize(s string) (int64, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	mul := int64(1)
	switch {
	case strings.HasSuffix(s, "k"):
		mul, s = 1<<10, s[:len(s)-1]
	case strings.HasSuffix(s, "m"):
		mul, s = 1<<20, s[:len(s)-1]
	case strings.HasSuffix(s, "g"):
		mul, s = 1<<30, s[:len(s)-1]
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("bad size %q", s)
	}
	return v * mul, nil
}

func parseSizeList(s string) ([]int64, error) {
	var out []int64
	for _, p := range strings.Split(s, ",") {
		if strings.TrimSpace(p) == "" {
			continue
		}
		v, err := parseSize(p)
		if err != nil {
			return nil, err
		}
		if v%align != 0 {
			return nil, fmt.Errorf("size %d is not a multiple of 4096", v)
		}
		out = append(out, v)
	}
	return out, nil
}

func mix64(x uint64) uint64 {
	x ^= x >> 33
	x *= 0xff51afd7ed558ccd
	x ^= x >> 33
	x *= 0xc4ceb9fe1a85ec53
	x ^= x >> 33
	return x
}

func f2(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) }
func f3(v float64) string { return strconv.FormatFloat(v, 'f', 3, 64) }
func f1(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) }
