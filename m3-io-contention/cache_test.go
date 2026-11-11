package main

import "testing"

func TestLRU(t *testing.T) {
	c := newLRU(16*4096*2, 4096) // 2 slots per shard
	b := make([]byte, 4096)
	for k := int64(0); k < 1000; k++ {
		b[0] = byte(k)
		c.put(k, b)
		if !c.get(k, b) || b[0] != byte(k) {
			t.Fatalf("just-inserted key %d missing", k)
		}
	}
	hits := 0
	for k := int64(0); k < 1000; k++ {
		if c.get(k, b) {
			hits++
		}
	}
	if hits > c.slots || hits == 0 {
		t.Fatalf("hits %d, slots %d", hits, c.slots)
	}
}
