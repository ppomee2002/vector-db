package main

// A small IVF-Flat vector index kept in memory: the CPU part of retrieval.
// Vectors are synthetic (cluster centre + noise); only the CPU cost per query
// and the IDs it returns matter here. Cost per query ~ (nlist + nprobe*nvec/nlist) * dim.

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

type Index struct {
	dim, nlist, nvec int
	cent             []float32   // nlist*dim
	lists            [][]float32 // per cluster, flattened vectors
	ids              [][]int32
	cdf              []float64 // query topic distribution over clusters (zipf or uniform)
}

func l2(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+8 <= n; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	for ; i < n; i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}

type xrng uint64

func (r *xrng) next() uint64 {
	x := uint64(*r)
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*r = xrng(x)
	return x
}

// uniform in [-1,1)
func (r *xrng) f32() float32 { return float32(int32(r.next()>>32)) / float32(1<<31) }

func BuildIndex(nvec, dim, nlist int, zipfS float64, seed int64, par int) *Index {
	ix := &Index{dim: dim, nlist: nlist, nvec: nvec}
	ix.cent = make([]float32, nlist*dim)
	r := xrng(uint64(seed)*0x9E3779B97F4A7C15 | 1)
	for i := range ix.cent {
		ix.cent[i] = r.f32()
	}
	cnt := make([]int, nlist)
	clusterOf := func(i int) int { return int(mix64(uint64(i)+uint64(seed)) % uint64(nlist)) }
	for i := 0; i < nvec; i++ {
		cnt[clusterOf(i)]++
	}
	ix.lists = make([][]float32, nlist)
	ix.ids = make([][]int32, nlist)
	for c := 0; c < nlist; c++ {
		ix.lists[c] = make([]float32, 0, cnt[c]*dim)
		ix.ids[c] = make([]int32, 0, cnt[c])
	}
	for i := 0; i < nvec; i++ {
		c := clusterOf(i)
		ix.ids[c] = append(ix.ids[c], int32(i))
	}
	var wg sync.WaitGroup
	ch := make(chan int, nlist)
	for c := 0; c < nlist; c++ {
		ch <- c
	}
	close(ch)
	for w := 0; w < par; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for c := range ch {
				rr := xrng(mix64(uint64(c)+uint64(seed)<<20) | 1)
				ce := ix.cent[c*dim : (c+1)*dim]
				l := ix.lists[c][:cnt[c]*dim]
				for j := 0; j < cnt[c]; j++ {
					v := l[j*dim : (j+1)*dim]
					for d := range v {
						v[d] = ce[d] + 0.35*rr.f32()
					}
				}
				ix.lists[c] = l
			}
		}(w)
	}
	wg.Wait()
	// query topic distribution
	ix.cdf = make([]float64, nlist)
	var s float64
	for i := 0; i < nlist; i++ {
		w := 1.0
		if zipfS > 0 {
			w = 1 / math.Pow(float64(i+1), zipfS)
		}
		s += w
		ix.cdf[i] = s
	}
	for i := range ix.cdf {
		ix.cdf[i] /= s
	}
	return ix
}

func (ix *Index) MemBytes() int64 { return int64(ix.nvec+ix.nlist) * int64(ix.dim) * 4 }

// MakeQuery writes a query vector near a (possibly popular) topic centre.
func (ix *Index) MakeQuery(r *rand.Rand, q []float32) {
	rank := sort.SearchFloat64s(ix.cdf, r.Float64())
	if rank >= ix.nlist {
		rank = ix.nlist - 1
	}
	c := int(mix64(uint64(rank)) % uint64(ix.nlist)) // popular topics are spread over clusters
	ce := ix.cent[c*ix.dim : (c+1)*ix.dim]
	xr := xrng(r.Uint64() | 1)
	for d := range q {
		q[d] = ce[d] + 0.35*xr.f32()
	}
}

type Scratch struct {
	cd    []float32
	probe []int
	pd    []float32
	topD  []float32
	topI  []int32
}

func (ix *Index) NewScratch(nprobe, k int) *Scratch {
	return &Scratch{cd: make([]float32, ix.nlist), probe: make([]int, 0, nprobe), pd: make([]float32, 0, nprobe),
		topD: make([]float32, 0, k), topI: make([]int32, 0, k)}
}

// Search returns the IDs of the k nearest vectors among the nprobe closest clusters.
func (ix *Index) Search(q []float32, nprobe, k int, s *Scratch) []int32 {
	dim := ix.dim
	s.probe, s.pd = s.probe[:0], s.pd[:0]
	for c := 0; c < ix.nlist; c++ {
		d := l2(q, ix.cent[c*dim:(c+1)*dim])
		if len(s.pd) == nprobe && d >= s.pd[nprobe-1] {
			continue
		}
		// insertion into small sorted list
		i := len(s.pd)
		if i < nprobe {
			s.pd = append(s.pd, d)
			s.probe = append(s.probe, c)
		} else {
			i = nprobe - 1
		}
		for i > 0 && s.pd[i-1] > d {
			s.pd[i], s.probe[i] = s.pd[i-1], s.probe[i-1]
			i--
		}
		s.pd[i], s.probe[i] = d, c
	}
	s.topD, s.topI = s.topD[:0], s.topI[:0]
	for _, c := range s.probe {
		l, ids := ix.lists[c], ix.ids[c]
		for j := range ids {
			d := l2(q, l[j*dim:(j+1)*dim])
			if len(s.topD) == k && d >= s.topD[k-1] {
				continue
			}
			i := len(s.topD)
			if i < k {
				s.topD = append(s.topD, d)
				s.topI = append(s.topI, ids[j])
			} else {
				i = k - 1
			}
			for i > 0 && s.topD[i-1] > d {
				s.topD[i], s.topI[i] = s.topD[i-1], s.topI[i-1]
				i--
			}
			s.topD[i], s.topI[i] = d, ids[j]
		}
	}
	return s.topI
}

// RandomDoc draws a document the way MakeQuery draws a topic (zipf or uniform
// over topics), then a random document of that topic. Used when the search is
// offloaded, so cache locality matches the search-on case.
func (ix *Index) RandomDoc(r *xrng) int32 {
	u := float64(r.next()>>11) / (1 << 53)
	rank := sort.SearchFloat64s(ix.cdf, u)
	if rank >= ix.nlist {
		rank = ix.nlist - 1
	}
	c := int(mix64(uint64(rank)) % uint64(ix.nlist))
	if l := ix.ids[c]; len(l) > 0 {
		return l[r.next()%uint64(len(l))]
	}
	return int32(r.next() % uint64(ix.nvec))
}
