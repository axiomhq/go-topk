package topk

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

type freqs struct {
	keys   []string
	counts map[string]int
}

func (f freqs) Len() int { return len(f.keys) }

// Actually 'Greater', since we want decreasing
func (f *freqs) Less(i, j int) bool {
	return f.counts[f.keys[i]] > f.counts[f.keys[j]] || f.counts[f.keys[i]] == f.counts[f.keys[j]] && f.keys[i] < f.keys[j]
}

func (f *freqs) Swap(i, j int) { f.keys[i], f.keys[j] = f.keys[j], f.keys[i] }

func TestTopK(t *testing.T) {

	f, err := os.Open("testdata/domains.txt")

	if err != nil {
		t.Fatal(err)
	}

	scanner := bufio.NewScanner(f)

	tk := New(100)
	exact := make(map[string]int)

	for scanner.Scan() {

		item := scanner.Text()

		exact[item]++
		e := tk.Insert(item, 1)
		if e.Count < exact[item] {
			t.Errorf("estimate lower than exact: key=%v, exact=%v, estimate=%v", e.Key, exact[item], e.Count)
		}
		if e.Count-e.Error > exact[item] {
			t.Errorf("error bounds too large: key=%v, count=%v, error=%v, exact=%v", e.Key, e.Count, e.Error, exact[item])
		}
	}

	if err := scanner.Err(); err != nil {
		log.Println("error during scan: ", err)
	}

	var keys []string

	for k := range exact {
		keys = append(keys, k)
	}

	freq := &freqs{keys: keys, counts: exact}

	sort.Sort(freq)

	top := tk.Keys()

	// at least the top 25 must be in order
	for i := 0; i < 25; i++ {
		if top[i].Key != freq.keys[i] {
			t.Errorf("key mismatch: idx=%d top=%s (%d) exact=%s (%d)", i, top[i].Key, top[i].Count, freq.keys[i], freq.counts[freq.keys[i]])
		}
	}
	for k, v := range exact {
		e := tk.Estimate(k)
		if e.Count < v {
			t.Errorf("estimate lower than exact: key=%v, exact=%v, estimate=%v", e.Key, v, e.Count)
		}
		if e.Count-e.Error > v {
			t.Errorf("error bounds too large: key=%v, count=%v, error=%v, exact=%v", e.Key, e.Count, e.Error, v)
		}
	}
	for _, k := range top {
		e := tk.Estimate(k.Key)
		if e != k {
			t.Errorf("estimate differs from top keys: key=%v, estimate=%v(-%v) top=%v(-%v)", e.Key, e.Count, e.Error, k.Count, k.Error)
		}
	}

	// msgp
	buf := bytes.NewBuffer(nil)
	if err := tk.Encode(buf); err != nil {
		t.Error(err)
	}

	decoded := New(100)
	if err := decoded.Decode(buf); err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(tk, decoded) {
		t.Error("they are not equal.")
	}
}

func TestTopKMerge(t *testing.T) {
	tk1 := New(20)
	tk2 := New(20)
	mtk := New(20)

	for i := 0; i <= 10000; i++ {
		x := rand.ExpFloat64() * 10
		word := fmt.Sprintf("word-%d", int(x))
		tk1.Insert(word, 1)
		mtk.Insert(word, 1)
	}

	for i := 0; i <= 10000; i++ {
		x := rand.ExpFloat64() * 10
		word := fmt.Sprintf("word-%d", int(x))
		tk2.Insert(word, 1)
		mtk.Insert(word, 1)
	}

	if err := tk1.Merge(tk2); err != nil {
		t.Error(err)
	}

	r1, r2 := tk1.Keys(), mtk.Keys()
	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("%v != %v", r1[i], r2[i])
		}
	}
}

func loadWords() []string {
	f, _ := os.Open("testdata/words.txt")
	defer f.Close()
	r := bufio.NewReader(f)

	res := make([]string, 0, 1024)
	for i := 0; ; i++ {
		if l, err := r.ReadString('\n'); err != nil {
			if err == io.EOF {
				return res
			}
			panic(err)
		} else {
			l = strings.Trim(l, "\r\n ")
			if len(l) > 0 {
				res = append(res, l)
			}
		}
	}
}

func exactCount(words []string) map[string]int {
	m := make(map[string]int, len(words))
	for _, w := range words {
		if _, ok := m[w]; ok {
			m[w]++
		} else {
			m[w] = 1
		}
	}

	return m
}

func exactTop(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Slice(keys, func(a, b int) bool {
		return m[keys[a]] > m[keys[b]]
	})

	return keys
}

// epsilon: count should be within exact*epsilon range
// returns: probability that a sample in the sketch lies outside the error range (delta)
func errorRate(epsilon float64, exact map[string]int, sketch map[string]Element) float64 {
	var numOk, numBad int

	for w, wc := range sketch {
		exactwc := float64(exact[w])
		lowerBound := int(math.Floor(exactwc * (1 - epsilon)))
		upperBound := int(math.Ceil(exactwc * (1 + epsilon)))

		if wc.Count-wc.Error < lowerBound || wc.Count-wc.Error > upperBound {
			numBad++
			fmt.Printf("!! %s: %d not in range [%d, %d], epsilon=%f\n", w, wc.Count-wc.Error, lowerBound, upperBound, epsilon)
		} else {
			numOk++
		}
	}

	return float64(numBad) / float64(len(sketch))
}

func resultToMap(result []Element) map[string]Element {
	res := make(map[string]Element, len(result))
	for _, lhh := range result {
		res[lhh.Key] = lhh
	}

	return res
}

func assertErrorRate(t *testing.T, exact map[string]int, result []Element, delta, epsilon float64) {
	t.Helper() // Indicates to the testing framework that this is a helper func to skip in stack traces
	sketch := resultToMap(result)
	effectiveDelta := errorRate(epsilon, exact, sketch)
	if effectiveDelta >= delta {
		t.Errorf("Expected error rate <= %f. Found %f. Sketch size: %d", delta, effectiveDelta, len(sketch))
	}
}

// split and array of strings into n slices
func split(words []string, splits int) [][]string {
	l := len(words)
	step := l / splits

	slices := make([][]string, 0, splits)
	for i := 0; i < splits; i++ {
		if i == splits-1 {
			slices = append(slices, words[i*step:])
		} else {
			slices = append(slices, words[i*step:i*step+step])
		}
	}

	sanityCheck := 0
	for _, slice := range slices {
		sanityCheck += len(slice)
	}
	if sanityCheck != l {
		panic("Internal error")
	}
	if len(slices) != splits {
		panic(fmt.Sprintf("Num splits mismatch %d/%d", len(slices), splits))
	}

	return slices
}

func TestSingle(t *testing.T) {
	delta := 0.05
	topK := 100

	words := loadWords()

	// Words in prime index positions are copied
	for _, p := range []int{2, 3, 5, 7, 11, 13, 17, 23} {
		for i := p; i < len(words); i += p {
			words[i] = words[p]
		}
	}

	sketch := New(topK)

	for _, w := range words {
		sketch.Insert(w, 1)
	}

	exact := exactCount(words)
	top := exactTop(exact)

	assertErrorRate(t, exact, sketch.Keys(), delta, 0.00001)
	//assertErrorRate(t, exact, sketch.Result(1)[:topK], delta, epsilon) // We would LOVE this to pass!

	// Assert order of heavy hitters in sub-sketch is as expected
	// TODO: by way of construction of test set we have pandemonium after #8, would like to check top[:topk]
	skTop := sketch.Keys()
	for i, w := range top[:8] {
		if w != skTop[i].Key && exact[w] != skTop[i].Count {
			fmt.Println("key", w, exact[w])
			t.Errorf("Expected top %d/%d to be '%s'(%d) found '%s'(%d)", i, topK, w, exact[w], skTop[i].Key, skTop[i].Count)
		}
	}
}

func TestTheShebang(t *testing.T) {
	words := loadWords()

	// Words in prime index positions are copied
	for _, p := range []int{2, 3, 5, 7, 11, 13, 17, 23} {
		for i := p; i < len(words); i += p {
			words[i] = words[p]
		}
	}

	cases := []struct {
		name   string
		slices [][]string
		delta  float64
		topk   int
	}{
		{
			name:   "Single slice top20 d=0.01",
			slices: split(words, 1),
			delta:  0.01,
			topk:   20,
		},
		{
			name:   "Two slices top20 d=0.01",
			slices: split(words, 2),
			delta:  0.01,
			topk:   20,
		},
		{
			name:   "Three slices top20 d=0.01",
			slices: split(words, 3),
			delta:  0.01,
			topk:   20,
		},
		{
			name:   "100 slices top20 d=0.01",
			slices: split(words, 100),
			delta:  0.01,
			topk:   20,
		},
	}

	for _, cas := range cases {
		t.Run(cas.name, func(t *testing.T) {
			caseRunner(t, cas.slices, cas.topk, cas.delta)
		})
	}
}

func caseRunner(t *testing.T, slices [][]string, topk int, delta float64) {
	var sketches []*Stream
	var corpusSize int

	// Find corpus size
	for _, slice := range slices {
		corpusSize += len(slice)
	}

	// Build sketches for each slice
	for _, slice := range slices {
		sk := New(topk)
		for _, w := range slice {
			sk.Insert(w, 1)
		}
		exact := exactCount(slice)
		top := exactTop(exact)
		skTop := sk.Keys()

		assertErrorRate(t, exact, skTop, delta, 0.001)

		// Assert order of heavy hitters in sub-sketch is as expected
		// TODO: by way of construction of test set we have pandemonium after #8, would like to check top[:topk]
		for i, w := range top[:8] {
			if w != skTop[i].Key && exact[w] != skTop[i].Count {
				t.Errorf("Expected top %d/%d to be '%s'(%d) found '%s'(%d)", i, topk, w, exact[w], skTop[i].Key, skTop[i].Count)
			}
		}

		sketches = append(sketches, sk)
	}

	if len(slices) == 1 {
		return
	}

	// Compute exact stats for entire corpus
	var allSlice []string
	for _, slice := range slices {
		allSlice = append(allSlice, slice...)
	}
	exactAll := exactCount(allSlice)

	// Merge all sketches
	mainSketch := sketches[0]
	for _, sk := range sketches[1:] {
		mainSketch.Merge(sk)
		// TODO: it would be nice to incrementally check the error rates
	}
	//assertErrorRate(t, exactAll, mainSketch.Keys(), delta, epsilon)

	// Assert order of heavy hitters in final result is as expected
	// TODO: by way of construction of test set we have pandemonium after #8, would like to check top[:topk]
	top := exactTop(exactAll)
	skTop := mainSketch.Keys()
	for i, w := range top[:8] {
		if w != skTop[i].Key {
			t.Errorf("Expected top %d/%d to be '%s'(%d) found '%s'(%d)", i, topk, w, exactAll[w], skTop[i].Key, skTop[i].Count)
		}
	}
}

func TestMarshalUnMarshal(t *testing.T) {
	topK := int(100)

	words := loadWords()

	// Words in prime index positions are copied
	for _, p := range []int{2, 3, 5, 7, 11, 13, 17, 23} {
		for i := p; i < len(words); i += p {
			words[i] = words[p]
		}
	}

	sketch := New(topK)

	for _, w := range words {
		sketch.Insert(w, 1)
	}

	b := bytes.NewBuffer(nil)
	err := sketch.Encode(b)
	assert.NoError(t, err)

	tmp := &Stream{}
	err = tmp.Decode(b)
	assert.NoError(t, err)
	assert.EqualValues(t, sketch, tmp)

}

func TestIssue(t *testing.T) {

	var (
		// use data from bug 3134
		//data = dataFromCounts(bug3134Data)
		// or use actual zipf distribution with same number of rows and distinct count
		data       = dataFromCounts(countsFromZipf(totalRecords(bug3134Data), len(bug3134Data), 1.1, 1.0))
		actual     = make(map[string]int)
		blockMax   = 16912
		sks        []*Stream
		sk         *Stream = New(10)
		blockCount int
	)

	// process data stream element at a time
	// each blockMax items, start a new stream
	for _, next := range data {
		if blockCount > blockMax {
			sks = append(sks, sk)
			sk = New(10)
			blockCount = 0
		}
		sk.Insert(next, 1)
		actual[next]++
		blockCount++
	}
	sks = append(sks, sk)

	// merge blocks
	for len(sks) > 1 {
		err := sks[0].Merge(sks[len(sks)-1])
		assert.NoError(t, err)
		sks = sks[:len(sks)-1]
	}
	sk = sks[0]

	// check results
	top := sk.Keys()
	for _, k := range top {
		fmt.Printf("%s - %d (%d) - actual %d\n", k.Key, k.Count, k.Error, actual[k.Key])
	}
}

// rowName convert integer row into string token
func rowName(i int) string {
	return fmt.Sprintf("%08d", i)
}

// countsFromZipf produces a set of token counts
// conforming to the zipf distribution with parameters s, v
// s must be > 1
// v must be >= 1
func countsFromZipf(n int, distinct int, s, v float64) []int {
	z := rand.NewZipf(rand.New(rand.NewSource(1)), s, v, uint64(distinct-1))
	counts := make([]int, distinct)
	for i := 0; i < n; i++ {
		counts[z.Uint64()]++
	}
	sort.Sort(sort.Reverse(sort.IntSlice(counts)))
	return counts
}

// dataFromCounts converts a slice of integer counts
// into a random token stream containing these
// tokens which will appear with the specified counts
// token names are generated from their position in the
// original slice, this means that if the counts are
// sorted token 00000000 will have the highest count
// this makes it easier to spot issues in the final output
func dataFromCounts(counts []int) []string {
	var rv []string
	data := make(map[string]int, len(counts))
	for i, count := range counts {
		data[rowName(i)] = count
	}
	for len(data) > 0 {
		for k, v := range data {
			v--
			if v > 0 {
				data[k] = v
			} else {
				delete(data, k)
			}
			rv = append(rv, k)
		}
	}
	return rv
}

// totalRecords is a helper to sum a slice of counts
// to produce the total number of items in the stream
func totalRecords(countSlice []int) int {
	var rv int
	for _, v := range countSlice {
		rv += v
	}
	return rv
}

var bug3134Data = []int{
	62618,
	62614,
	1977,
	1977,
	726,
	696,
	696,
	696,
	696,
	696,
	674,
	201,
	113,
	102,
	62,
	62,
	62,
	62,
	62,
	62,
	62,
	62,
	62,
	61,
	50,
	33,
	28,
	28,
	27,
	22,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	21,
	16,
	16,
	16,
	16,
	15,
	15,
	15,
	15,
	15,
	15,
	15,
	15,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	14,
	13,
	12,
	12,
	11,
	11,
	11,
	10,
	9,
	8,
	8,
	8,
	7,
	7,
	7,
	7,
	7,
	6,
	6,
	5,
	5,
	5,
	5,
	5,
	5,
	4,
	4,
	4,
	4,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
}
