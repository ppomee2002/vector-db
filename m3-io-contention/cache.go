package main

// Sharded LRU chunk cache (application-level document cache, like an
// in-process cache in front of the document store). Hits copy the chunk out
// under the shard lock; misses are read from the SSD and then inserted.

import "sync"

type lruShard struct {
	mu         sync.Mutex
	slot       map[int64]int32
	key        []int64
	prev, next []int32
	head, tail int32
	used       int32
	data       []byte
}

type lruCache struct {
	shards []*lruShard
	chunk  int64
	slots  int
}

func newLRU(bytes int64, chunk int64) *lruCache {
	const nsh = 16
	per := int(bytes / chunk / nsh)
	if per < 1 {
		per = 1
	}
	c := &lruCache{chunk: chunk, slots: per * nsh}
	for i := 0; i < nsh; i++ {
		s := &lruShard{slot: make(map[int64]int32, per), key: make([]int64, per), prev: make([]int32, per),
			next: make([]int32, per), head: -1, tail: -1, data: make([]byte, int64(per)*chunk)}
		c.shards = append(c.shards, s)
	}
	return c
}

func (c *lruCache) shard(k int64) *lruShard { return c.shards[mix64(uint64(k))%uint64(len(c.shards))] }

func (s *lruShard) unlink(i int32) {
	p, n := s.prev[i], s.next[i]
	if p >= 0 {
		s.next[p] = n
	} else {
		s.head = n
	}
	if n >= 0 {
		s.prev[n] = p
	} else {
		s.tail = p
	}
}

func (s *lruShard) pushFront(i int32) {
	s.prev[i], s.next[i] = -1, s.head
	if s.head >= 0 {
		s.prev[s.head] = i
	}
	s.head = i
	if s.tail < 0 {
		s.tail = i
	}
}

// get copies the chunk into dst and returns true on a hit.
func (c *lruCache) get(k int64, dst []byte) bool {
	s := c.shard(k)
	s.mu.Lock()
	i, ok := s.slot[k]
	if ok {
		s.unlink(i)
		s.pushFront(i)
		copy(dst, s.data[int64(i)*c.chunk:int64(i+1)*c.chunk])
	}
	s.mu.Unlock()
	return ok
}

func (c *lruCache) put(k int64, src []byte) {
	s := c.shard(k)
	s.mu.Lock()
	defer s.mu.Unlock()
	if i, ok := s.slot[k]; ok {
		s.unlink(i)
		s.pushFront(i)
		return
	}
	var i int32
	if int(s.used) < len(s.key) {
		i = s.used
		s.used++
	} else {
		i = s.tail
		s.unlink(i)
		delete(s.slot, s.key[i])
	}
	s.key[i] = k
	s.slot[k] = i
	copy(s.data[int64(i)*c.chunk:int64(i+1)*c.chunk], src)
	s.pushFront(i)
}
