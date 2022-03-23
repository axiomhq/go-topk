// Package topk implements the Filtered Space-Saving TopK streaming algorithm
/*

The original Space-Saving algorithm:
https://icmi.cs.ucsb.edu/research/tech_reports/reports/2005-23.pdf

The Filtered Space-Saving enhancement:
http://www.l2f.inesc-id.pt/~fmmb/wiki/uploads/Work/misnis.ref0a.pdf

This implementation follows the algorithm of the FSS paper, but not the
suggested implementation.  Specifically, we use a heap instead of a sorted list
of monitored items, and since we are also using a map to provide O(1) access on
update also don't need the c_i counters in the hash table.

Licensed under the MIT license.

*/
package topk

import (
	"container/heap"
	"fmt"
	"io"
	"sort"

	"github.com/dgryski/go-metro"
	"github.com/tinylib/msgp/msgp"
)

// Element is a TopK item
type Element struct {
	Key   string `json:"key"`
	Count int    `json:"count"`
	Error int    `json:"error"`
}

type elementsByCountDescending []Element

func (elts elementsByCountDescending) Len() int { return len(elts) }
func (elts elementsByCountDescending) Less(i, j int) bool {
	return (elts[i].Count > elts[j].Count) || (elts[i].Count == elts[j].Count && elts[i].Key < elts[j].Key)
}
func (elts elementsByCountDescending) Swap(i, j int) { elts[i], elts[j] = elts[j], elts[i] }

const nPartitions = 6

type partitions [nPartitions]keys

func (p *partitions) EncodeMsgp(w *msgp.Writer) error {
	for _, v := range *p {
		if err := v.EncodeMsgp(w); err != nil {
			return err
		}
	}
	return nil
}

func (p *partitions) DecodeMsgp(r *msgp.Reader) error {
	var (
		err error
	)
	for i := 0; i < 6; i++ {
		if err = p[i].DecodeMsgp(r); err != nil {
			return err
		}
	}
	return nil
}

type keys struct {
	m    map[string]int
	elts []Element
}

func (tk *keys) EncodeMsgp(w *msgp.Writer) error {
	if err := w.WriteMapHeader(uint32(len(tk.m))); err != nil {
		return err
	}
	for k, v := range tk.m {
		if err := w.WriteString(k); err != nil {
			return err
		}
		if err := w.WriteInt(v); err != nil {
			return err
		}
	}

	if err := w.WriteArrayHeader(uint32(len(tk.elts))); err != nil {
		return err
	}
	for _, e := range tk.elts {
		if err := w.WriteString(e.Key); err != nil {
			return err
		}
		if err := w.WriteInt(e.Count); err != nil {
			return err
		}
		if err := w.WriteInt(e.Error); err != nil {
			return err
		}
	}
	return nil
}

func (tk *keys) DecodeMsgp(r *msgp.Reader) error {
	var (
		err error
		sz  uint32
	)

	if sz, err = r.ReadMapHeader(); err != nil {
		return err
	}

	tk.m = make(map[string]int, sz)

	for i := uint32(0); i < sz; i++ {
		key, err := r.ReadString()
		if err != nil {
			return err
		}
		val, err := r.ReadInt()
		if err != nil {
			return err
		}
		tk.m[key] = val
	}

	if sz, err = r.ReadArrayHeader(); err != nil {
		return err
	}

	tk.elts = make([]Element, sz)
	for i := range tk.elts {
		if tk.elts[i].Key, err = r.ReadString(); err != nil {
			return err
		}
		if tk.elts[i].Count, err = r.ReadInt(); err != nil {
			return err
		}
		if tk.elts[i].Error, err = r.ReadInt(); err != nil {
			return err
		}
	}

	return nil
}

// Implement the container/heap interface

// Len ...
func (tk *keys) Len() int { return len(tk.elts) }

// Less ...
func (tk *keys) Less(i, j int) bool {
	return (tk.elts[i].Count < tk.elts[j].Count) || (tk.elts[i].Count == tk.elts[j].Count && tk.elts[i].Error > tk.elts[j].Error)
}
func (tk *keys) Swap(i, j int) {

	tk.elts[i], tk.elts[j] = tk.elts[j], tk.elts[i]

	tk.m[tk.elts[i].Key] = i
	tk.m[tk.elts[j].Key] = j
}

func (tk *keys) Push(x interface{}) {
	e := x.(Element)
	tk.m[e.Key] = len(tk.elts)
	tk.elts = append(tk.elts, e)
}

func (tk *keys) Pop() interface{} {
	var e Element
	e, tk.elts = tk.elts[len(tk.elts)-1], tk.elts[:len(tk.elts)-1]

	delete(tk.m, e.Key)

	return e
}

// Stream calculates the TopK elements for a stream
type Stream struct {
	n      int
	p      partitions // partitions are the different heaps
	alphas []int
}

// New returns a Stream estimating the top n most frequent elements
func New(n int) *Stream {
	p := partitions{}
	k := 1 + n/len(p)
	for i := range p {
		p[i] = keys{m: make(map[string]int, k), elts: make([]Element, 0, k)}
	}

	return &Stream{
		n:      n,
		p:      p,
		alphas: make([]int, n*nPartitions), // 6 is the multiplicative constant from the paper
	}
}

func reduce(x uint64, n int) uint32 {
	return uint32(uint64(uint32(x)) * uint64(n) >> 32)
}

// Insert adds an element to the stream to be tracked
// It returns an estimation for the just inserted element
func (s *Stream) Insert(x string, count int) Element {
	strHash := metro.Hash64Str(x, 0)
	xhash := reduce(strHash, len(s.alphas))
	i := strHash % uint64(len(s.p))

	// are we tracking this element?
	if idx, ok := s.p[i].m[x]; ok {
		s.p[i].elts[idx].Count += count
		e := s.p[i].elts[idx]
		heap.Fix(&s.p[i], idx)
		return e
	}

	// can we track more elements?
	if len(s.p[i].elts) < s.n {
		// there is free sp[i]ace
		e := Element{Key: x, Count: count}
		heap.Push(&s.p[i], e)
		return e
	}

	if s.alphas[xhash]+count < s.p[i].elts[0].Count {
		e := Element{
			Key:   x,
			Error: s.alphas[xhash],
			Count: s.alphas[xhash] + count,
		}
		s.alphas[xhash] += count
		return e
	}

	// replace the current minimum element
	minElement := s.p[i].elts[0]

	mkhash := reduce(metro.Hash64Str(minElement.Key, 0), len(s.alphas))
	s.alphas[mkhash] = minElement.Count

	e := Element{
		Key:   x,
		Error: s.alphas[xhash],
		Count: s.alphas[xhash] + count,
	}
	s.p[i].elts[0] = e

	// we're not longer monitoring minKey
	delete(s.p[i].m, minElement.Key)
	// but 'x' is as array position 0
	s.p[i].m[x] = 0

	heap.Fix(&s.p[i], 0)
	return e
}

// Merge ...
func (s *Stream) Merge(other *Stream) error {
	if s.n != other.n {
		return fmt.Errorf("expected stream of size n %d, got %d", s.n, other.n)
	}

	for i := range s.p {

		// merge the elements
		eKeys := make(map[string]struct{})
		eMap := make(map[string]Element)
		for _, e := range s.p[i].elts {
			eKeys[e.Key] = struct{}{}
		}
		for _, e := range other.p[i].elts {
			eKeys[e.Key] = struct{}{}
		}

		for k := range eKeys {
			idx1, ok1 := s.p[i].m[k]
			idx2, ok2 := other.p[i].m[k]
			xhash := reduce(metro.Hash64Str(k, 0), len(s.alphas))
			min1 := other.alphas[xhash]
			min2 := other.alphas[xhash]

			switch {
			case ok1 && ok2:
				e1 := s.p[i].elts[idx1]
				e2 := other.p[i].elts[idx2]
				eMap[k] = Element{
					Key:   k,
					Count: e1.Count + e2.Count,
					Error: e1.Error + e2.Error,
				}
			case ok1:
				e1 := s.p[i].elts[idx1]
				eMap[k] = Element{
					Key:   k,
					Count: e1.Count + min2,
					Error: e1.Error + min2,
				}
			case ok2:
				e2 := other.p[i].elts[idx2]
				eMap[k] = Element{
					Key:   k,
					Count: e2.Count + min1,
					Error: e2.Error + min1,
				}
			}

		}

		// sort the elements
		elts := make([]Element, 0, len(eMap))
		for _, v := range eMap {
			elts = append(elts, v)
		}
		sort.Sort(elementsByCountDescending(elts))

		// trim elements
		if len(elts) > s.n {
			elts = elts[:s.n]
		}

		// create heap
		tk := keys{
			m:    make(map[string]int),
			elts: make([]Element, 0, s.n),
		}
		for _, e := range elts {
			heap.Push(&tk, e)
		}

		// modify alphas
		for i, v := range other.alphas {
			s.alphas[i] += v
		}

		// replace k
		s.p[i] = tk
	}
	return nil
}

// Keys returns the current estimates for the most frequent elements
func (s *Stream) Keys() []Element {
	l := 1 + s.n/len(s.p)
	elts := make([]Element, 0, l*len(s.p))
	for _, p := range s.p {
		elts = append(elts, p.elts...)
	}
	sort.Sort(elementsByCountDescending(elts))
	if len(elts) > s.n {
		elts = elts[:s.n]
	}
	return elts
}

// Estimate returns an estimate for the item x
func (s *Stream) Estimate(x string) Element {
	strHash := metro.Hash64Str(x, 0)
	xhash := reduce(strHash, len(s.alphas))
	i := strHash % uint64(len(s.p))

	// are we tracking this element?
	if idx, ok := s.p[i].m[x]; ok {
		e := s.p[i].elts[idx]
		return e
	}

	count := s.alphas[xhash]
	e := Element{
		Key:   x,
		Error: count,
		Count: count,
	}
	return e
}

// EncodeMsgp ...
func (s *Stream) EncodeMsgp(w *msgp.Writer) error {
	if err := w.WriteInt(s.n); err != nil {
		return err
	}

	if err := w.WriteArrayHeader(uint32(len(s.alphas))); err != nil {
		return err
	}

	for _, a := range s.alphas {
		if err := w.WriteInt(a); err != nil {
			return err
		}
	}

	return s.p.EncodeMsgp(w)
}

// DecodeMsgp ...
func (s *Stream) DecodeMsgp(r *msgp.Reader) error {
	var (
		err error
		sz  uint32
	)

	if s.n, err = r.ReadInt(); err != nil {
		return err
	}

	if sz, err = r.ReadArrayHeader(); err != nil {
		return err
	}

	s.alphas = make([]int, sz)
	for i := range s.alphas {
		if s.alphas[i], err = r.ReadInt(); err != nil {
			return err
		}
	}

	return s.p.DecodeMsgp(r)
}

// Encode ...
func (s *Stream) Encode(w io.Writer) error {
	wrt := msgp.NewWriter(w)
	if err := s.EncodeMsgp(wrt); err != nil {
		return err
	}
	return wrt.Flush()
}

// Decode ...
func (s *Stream) Decode(r io.Reader) error {
	rdr := msgp.NewReader(r)
	return s.DecodeMsgp(rdr)
}
