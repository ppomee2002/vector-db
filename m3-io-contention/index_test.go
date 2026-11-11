package main

import (
	"math/rand"
	"sort"
	"testing"
)

// With nprobe = nlist the IVF search must equal brute force.
func TestSearchExact(t *testing.T) {
	ix := BuildIndex(3000, 32, 16, 0, 7, 2)
	r := rand.New(rand.NewSource(1))
	q := make([]float32, 32)
	s := ix.NewScratch(16, 10)
	for it := 0; it < 20; it++ {
		ix.MakeQuery(r, q)
		got := append([]int32(nil), ix.Search(q, 16, 10, s)...)
		type p struct {
			d  float32
			id int32
		}
		var all []p
		for c := range ix.ids {
			for j, id := range ix.ids[c] {
				all = append(all, p{l2(q, ix.lists[c][j*32:(j+1)*32]), id})
			}
		}
		sort.Slice(all, func(i, j int) bool { return all[i].d < all[j].d })
		for i := 0; i < 10; i++ {
			if all[i].id != got[i] {
				t.Fatalf("iter %d rank %d: want %d got %d", it, i, all[i].id, got[i])
			}
		}
	}
}
