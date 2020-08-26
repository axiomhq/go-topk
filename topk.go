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
	"bytes"
	"container/heap"
	"encoding/gob"
	"io"
	"sort"

	"github.com/dgryski/go-sip13"
	"github.com/tinylib/msgp/msgp"
)

// Element is a TopK item
type Element struct {
	Key   *string
	Count int
	Error int
}

type elementsByCountDescending []Element

func (elts elementsByCountDescending) Len() int { return len(elts) }
func (elts elementsByCountDescending) Less(i, j int) bool {
	return (elts[i].Count > elts[j].Count) || (elts[i].Count == elts[j].Count && *elts[i].Key < *elts[j].Key)
}
func (elts elementsByCountDescending) Swap(i, j int) { elts[i], elts[j] = elts[j], elts[i] }

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
		if err := w.WriteString(*e.Key); err != nil {
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

func (tk *keys) DecodeMsp(r *msgp.Reader) error {
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
		x := ""
		tk.elts[i].Key = &x
		if *tk.elts[i].Key, err = r.ReadString(); err != nil {
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

	tk.m[*tk.elts[i].Key] = i
	tk.m[*tk.elts[j].Key] = j
}

func (tk *keys) Push(x interface{}) {
	e := x.(Element)
	tk.m[*e.Key] = len(tk.elts)
	tk.elts = append(tk.elts, e)
}

func (tk *keys) Pop() interface{} {
	var e Element
	e, tk.elts = tk.elts[len(tk.elts)-1], tk.elts[:len(tk.elts)-1]

	delete(tk.m, *e.Key)

	return e
}

// Stream calculates the TopK elements for a stream
type Stream struct {
	n      int
	k      keys
	alphas []int
}

// New returns a Stream estimating the top n most frequent elements
func New(n int) *Stream {
	return &Stream{
		n:      n,
		k:      keys{m: make(map[string]int), elts: make([]Element, 0, n)},
		alphas: make([]int, n*6), // 6 is the multiplicative constant from the paper
	}
}

func reduce(x uint64, n int) uint32 {
	return uint32(uint64(uint32(x)) * uint64(n) >> 32)
}

// Insert adds an element to the stream to be tracked
// It returns an estimation for the just inserted element
func (s *Stream) Insert(x string, count int) Element {
	xhash := reduce(sip13.Sum64Str(0, 0, x), len(s.alphas))

	// are we tracking this element?
	if idx, ok := s.k.m[x]; ok {
		s.k.elts[idx].Count += count
		e := s.k.elts[idx]
		heap.Fix(&s.k, idx)
		return e
	}

	// NOTE: This is where things go wrong

	ptr := &x
	// can we track more elements?
	if len(s.k.elts) < s.n {
		// there is free space
		e := Element{Key: ptr, Count: count}
		heap.Push(&s.k, e)
		return e
	}

	if s.alphas[xhash]+count < s.k.elts[0].Count {
		e := Element{
			Key:   ptr,
			Error: s.alphas[xhash],
			Count: s.alphas[xhash] + count,
		}
		s.alphas[xhash] += count
		return e
	}

	// replace the current minimum element
	minKey := s.k.elts[0].Key

	mkhash := reduce(sip13.Sum64Str(0, 0, *minKey), len(s.alphas))
	s.alphas[mkhash] = s.k.elts[0].Count

	e := Element{
		Key:   ptr,
		Error: s.alphas[xhash],
		Count: s.alphas[xhash] + count,
	}
	s.k.elts[0] = e

	// we're not longer monitoring minKey
	delete(s.k.m, *minKey)
	// but 'x' is as array position 0
	s.k.m[x] = 0

	heap.Fix(&s.k, 0)
	return e
}

// Keys returns the current estimates for the most frequent elements
func (s *Stream) Keys() []Element {
	elts := append([]Element(nil), s.k.elts...)
	sort.Sort(elementsByCountDescending(elts))
	return elts
}

// Estimate returns an estimate for the item x
func (s *Stream) Estimate(x string) Element {
	xhash := reduce(sip13.Sum64Str(0, 0, x), len(s.alphas))

	// are we tracking this element?
	if idx, ok := s.k.m[x]; ok {
		e := s.k.elts[idx]
		return e
	}
	count := s.alphas[xhash]
	e := Element{
		Key:   &x,
		Error: count,
		Count: count,
	}
	return e
}

// GobEncode ...
func (s *Stream) GobEncode() ([]byte, error) {
	buf := bytes.Buffer{}
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(s.n); err != nil {
		return nil, err
	}
	if err := enc.Encode(s.k.m); err != nil {
		return nil, err
	}
	if err := enc.Encode(s.k.elts); err != nil {
		return nil, err
	}
	if err := enc.Encode(s.alphas); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode ...
func (s *Stream) GobDecode(b []byte) error {
	dec := gob.NewDecoder(bytes.NewBuffer(b))
	if err := dec.Decode(&s.n); err != nil {
		return err
	}
	if err := dec.Decode(&s.k.m); err != nil {
		return err
	}
	if err := dec.Decode(&s.k.elts); err != nil {
		return err
	}
	if err := dec.Decode(&s.alphas); err != nil {
		return err
	}
	return nil
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

	return s.k.EncodeMsgp(w)
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

	return s.k.DecodeMsp(r)
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
