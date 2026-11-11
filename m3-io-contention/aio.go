package main

// Linux native AIO (io_setup / io_submit / io_getevents) with O_DIRECT:
// real asynchronous reads, no thread blocks per read. Used by
//   - run -io async : workers do the CPU search, submit the k reads and move on
//                     to the next request; a completion thread (reaper) finishes
//                     the request when its last read lands (event-driven server)
//   - baseline -engine aio : one thread keeps QD reads in flight, so the SSD's
//                     own limit can be reached without spending many CPUs

import (
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

type iocb struct { // struct iocb, 64 bytes, x86-64 little endian
	data      uint64
	key       uint32
	rwFlags   int32
	opcode    uint16
	reqprio   int16
	fd        uint32
	buf       uint64
	nbytes    uint64
	offset    int64
	reserved2 uint64
	flags     uint32
	resfd     uint32
}

type ioEvent struct {
	data, obj uint64
	res, res2 int64
}

const iocbCmdPread = 0

type aioCtx struct {
	ctx  uintptr
	size int
}

func newAIO(n int) (*aioCtx, error) {
	if n < 4 {
		n = 4
	}
	for {
		var ctx uintptr
		_, _, e := syscall.Syscall(syscall.SYS_IO_SETUP, uintptr(n), uintptr(unsafe.Pointer(&ctx)), 0)
		if e == 0 {
			return &aioCtx{ctx: ctx, size: n}, nil
		}
		if e != syscall.EAGAIN || n <= 16 { // aio-max-nr is system wide; shrink if others use it
			return nil, e
		}
		n /= 2
	}
}

func (a *aioCtx) destroy() { syscall.Syscall(syscall.SYS_IO_DESTROY, a.ctx, 0, 0) }

func (a *aioCtx) submit(ptrs []*iocb) error {
	for len(ptrs) > 0 {
		n, _, e := syscall.Syscall(syscall.SYS_IO_SUBMIT, a.ctx, uintptr(len(ptrs)), uintptr(unsafe.Pointer(&ptrs[0])))
		if e == syscall.EAGAIN || e == syscall.EINTR {
			runtime.Gosched()
			continue
		}
		if e != 0 {
			return e
		}
		ptrs = ptrs[n:]
	}
	return nil
}

func (a *aioCtx) getEvents(ev []ioEvent, min int, timeout time.Duration) (int, error) {
	ts := syscall.NsecToTimespec(int64(timeout))
	n, _, e := syscall.Syscall6(syscall.SYS_IO_GETEVENTS, a.ctx, uintptr(min), uintptr(len(ev)),
		uintptr(unsafe.Pointer(&ev[0])), uintptr(unsafe.Pointer(&ts)), 0)
	if e == syscall.EINTR {
		return 0, nil
	}
	if e != 0 {
		return 0, e
	}
	return int(n), nil
}

func fillPread(cb *iocb, fd uintptr, buf []byte, off int64, data uint64) {
	*cb = iocb{data: data, opcode: iocbCmdPread, fd: uint32(fd), buf: uint64(uintptr(unsafe.Pointer(&buf[0]))),
		nbytes: uint64(len(buf)), offset: off}
}

// aioAvailable reports whether native AIO works here (a container seccomp
// profile may block it).
func aioAvailable() error {
	a, err := newAIO(64)
	if err != nil {
		return err
	}
	a.destroy()
	return nil
}
