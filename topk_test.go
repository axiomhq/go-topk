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
	count := 0

	for scanner.Scan() {

		item := scanner.Text()

		exact[item]++
		count++
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

	assert.Equal(t, count, tk.Count())

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
	count := 0

	for i := 0; i <= 10000; i++ {
		x := rand.ExpFloat64() * 10
		word := fmt.Sprintf("word-%d", int(x))
		tk1.Insert(word, 1)
		mtk.Insert(word, 1)
		count++
	}

	for i := 0; i <= 10000; i++ {
		x := rand.ExpFloat64() * 10
		word := fmt.Sprintf("word-%d", int(x))
		tk2.Insert(word, 1)
		mtk.Insert(word, 1)
		count++
	}

	if err := tk1.Merge(tk2); err != nil {
		t.Error(err)
	}

	r1, r2 := tk1.Keys(), mtk.Keys()
	for i := range r1 {
		fmt.Println(r1[i], r2[i])
		if r1[i] != r2[i] {
			t.Errorf("%v != %v", r1[i], r2[i])
		}
	}
	assert.Equal(t, count, mtk.Count())
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
	var sketches []*TopK
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

	fmt.Println(len(b.Bytes()))

	tmp := &TopK{}
	err = tmp.Decode(b)
	assert.NoError(t, err)
	assert.EqualValues(t, sketch, tmp)

}

func TestBug3134(t *testing.T) {
	data := bug3134Data

	// copy the original data, we will mutate this copy
	work := make(map[string]int)
	for k, v := range data {
		work[k] = v
	}

	var sk *Stream
	var sks []*Stream

	var i int
	for len(work) > 0 {
		for k, v := range work {
			if i%math.MaxUint16 == 0 {
				// append previous to list
				if sk != nil {
					sks = append(sks, sk)
				}
				// start a new one
				sk = New(10)
			}

			sk.Insert(k, 1)
			v--
			if v > 0 {
				work[k] = v
			} else {
				delete(work, k)
			}
			i++
		}
	}

	// now merge them
	sk = sks[0]
	if len(sks) > 1 {
		for _, sko := range sks[1:] {
			err := sk.Merge(sko)
			assert.NoError(t, err)
		}
	}

	top := sk.Keys()
	for _, k := range top {
		fmt.Printf("%s - %d (%d)\n", k.Key, k.Count, k.Error)
	}
}

var bug3134Data = map[string]int{
	`row000`: 62618,
	`row001`: 62614,
	`row002`: 1977,
	`row003`: 1977,
	`row004`: 726,
	`row005`: 696,
	`row006`: 696,
	`row007`: 696,
	`row008`: 696,
	`row009`: 696,
	`row010`: 674,
	`row011`: 201,
	`row012`: 113,
	`row013`: 102,
	`row014`: 62,
	`row015`: 62,
	`row016`: 62,
	`row017`: 62,
	`row018`: 62,
	`row019`: 62,
	`row020`: 62,
	`row021`: 62,
	`row022`: 62,
	`row023`: 61,
	`row024`: 50,
	`row025`: 33,
	`row026`: 28,
	`row027`: 28,
	`row028`: 27,
	`row029`: 22,
	`row030`: 21,
	`row031`: 21,
	`row032`: 21,
	`row033`: 21,
	`row034`: 21,
	`row035`: 21,
	`row036`: 21,
	`row037`: 21,
	`row038`: 21,
	`row039`: 21,
	`row040`: 21,
	`row041`: 21,
	`row042`: 21,
	`row043`: 21,
	`row044`: 21,
	`row045`: 21,
	`row046`: 16,
	`row047`: 16,
	`row048`: 16,
	`row049`: 16,
	`row050`: 15,
	`row051`: 15,
	`row052`: 15,
	`row053`: 15,
	`row054`: 15,
	`row055`: 15,
	`row056`: 15,
	`row057`: 15,
	`row058`: 14,
	`row059`: 14,
	`row060`: 14,
	`row061`: 14,
	`row062`: 14,
	`row063`: 14,
	`row064`: 14,
	`row065`: 14,
	`row066`: 14,
	`row067`: 14,
	`row068`: 14,
	`row069`: 14,
	`row070`: 14,
	`row071`: 14,
	`row072`: 14,
	`row073`: 14,
	`row074`: 14,
	`row075`: 14,
	`row076`: 14,
	`row077`: 14,
	`row078`: 13,
	`row079`: 12,
	`row080`: 12,
	`row081`: 11,
	`row082`: 11,
	`row083`: 11,
	`row084`: 10,
	`row085`: 9,
	`row086`: 8,
	`row087`: 8,
	`row088`: 8,
	`row089`: 7,
	`row090`: 7,
	`row091`: 7,
	`row092`: 7,
	`row093`: 7,
	`row094`: 6,
	`row095`: 6,
	`row096`: 5,
	`row097`: 5,
	`row098`: 5,
	`row099`: 5,
	`row100`: 5,
	`row101`: 5,
	`row102`: 4,
	`row103`: 4,
	`row104`: 4,
	`row105`: 4,
	`row106`: 3,
	`row107`: 3,
	`row108`: 3,
	`row109`: 3,
	`row110`: 3,
	`row111`: 3,
	`row112`: 3,
	`row113`: 3,
	`row114`: 2,
	`row115`: 2,
	`row116`: 2,
	`row117`: 2,
	`row118`: 2,
	`row119`: 2,
	`row120`: 2,
	`row121`: 2,
	`row122`: 2,
	`row123`: 2,
	`row124`: 2,
	`row125`: 2,
	`row126`: 2,
	`row127`: 1,
	`row128`: 1,
	`row129`: 1,
	`row130`: 1,
	`row131`: 1,
	`row132`: 1,
	`row133`: 1,
	`row134`: 1,
	`row135`: 1,
	`row136`: 1,
	`row137`: 1,
	`row138`: 1,
	`row139`: 1,
	`row140`: 1,
	`row141`: 1,
	`row142`: 1,
	`row143`: 1,
	`row144`: 1,
	`row145`: 1,
	`row146`: 1,
	`row147`: 1,
	`row148`: 1,
	`row149`: 1,
	`row150`: 1,
	`row151`: 1,
	`row152`: 1,
	`row153`: 1,
	`row154`: 1,
	`row155`: 1,
	`row156`: 1,
	`row157`: 1,
	`row158`: 1,
	`row159`: 1,
	`row160`: 1,
	`row161`: 1,
	`row162`: 1,
	`row163`: 1,
	`row164`: 1,
	`row165`: 1,
	`row166`: 1,
	`row167`: 1,
	`row168`: 1,
	`row169`: 1,
	`row170`: 1,
	`row171`: 1,
	`row172`: 1,
	`row173`: 1,
	`row174`: 1,
	`row175`: 1,
	`row176`: 1,
	`row177`: 1,
	`row178`: 1,
	`row179`: 1,
	`row180`: 1,
	`row181`: 1,
	`row182`: 1,
	`row183`: 1,
	`row184`: 1,
	`row185`: 1,
	`row186`: 1,
	`row187`: 1,
	`row188`: 1,
	`row189`: 1,
	`row190`: 1,
	`row191`: 1,
	`row192`: 1,
	`row193`: 1,
	`row194`: 1,
	`row195`: 1,
	`row196`: 1,
	`row197`: 1,
	`row198`: 1,
	`row199`: 1,
	`row200`: 1,
	`row201`: 1,
	`row202`: 1,
	`row203`: 1,
	`row204`: 1,
	`row205`: 1,
	`row206`: 1,
	`row207`: 1,
	`row208`: 1,
	`row209`: 1,
	`row210`: 1,
	`row211`: 1,
	`row212`: 1,
	`row213`: 1,
	`row214`: 1,
	`row215`: 1,
	`row216`: 1,
	`row217`: 1,
	`row218`: 1,
	`row219`: 1,
	`row220`: 1,
	`row221`: 1,
	`row222`: 1,
	`row223`: 1,
	`row224`: 1,
	`row225`: 1,
	`row226`: 1,
	`row227`: 1,
	`row228`: 1,
	`row229`: 1,
	`row230`: 1,
	`row231`: 1,
	`row232`: 1,
	`row233`: 1,
	`row234`: 1,
	`row235`: 1,
	`row236`: 1,
	`row237`: 1,
	`row238`: 1,
	`row239`: 1,
	`row240`: 1,
	`row241`: 1,
	`row242`: 1,
	`row243`: 1,
	`row244`: 1,
	`row245`: 1,
	`row246`: 1,
	`row247`: 1,
	`row248`: 1,
	`row249`: 1,
	`row250`: 1,
	`row251`: 1,
	`row252`: 1,
	`row253`: 1,
	`row254`: 1,
	`row255`: 1,
	`row256`: 1,
	`row257`: 1,
	`row258`: 1,
	`row259`: 1,
	`row260`: 1,
	`row261`: 1,
	`row262`: 1,
	`row263`: 1,
	`row264`: 1,
	`row265`: 1,
	`row266`: 1,
	`row267`: 1,
	`row268`: 1,
	`row269`: 1,
	`row270`: 1,
	`row271`: 1,
	`row272`: 1,
	`row273`: 1,
	`row274`: 1,
	`row275`: 1,
	`row276`: 1,
	`row277`: 1,
	`row278`: 1,
	`row279`: 1,
	`row280`: 1,
	`row281`: 1,
	`row282`: 1,
	`row283`: 1,
	`row284`: 1,
	`row285`: 1,
	`row286`: 1,
	`row287`: 1,
	`row288`: 1,
	`row289`: 1,
	`row290`: 1,
	`row291`: 1,
	`row292`: 1,
	`row293`: 1,
	`row294`: 1,
	`row295`: 1,
	`row296`: 1,
	`row297`: 1,
	`row298`: 1,
	`row299`: 1,
	`row300`: 1,
	`row301`: 1,
	`row302`: 1,
	`row303`: 1,
	`row304`: 1,
	`row305`: 1,
	`row306`: 1,
	`row307`: 1,
	`row308`: 1,
	`row309`: 1,
	`row310`: 1,
	`row311`: 1,
	`row312`: 1,
	`row313`: 1,
	`row314`: 1,
	`row315`: 1,
	`row316`: 1,
	`row317`: 1,
	`row318`: 1,
	`row319`: 1,
	`row320`: 1,
	`row321`: 1,
	`row322`: 1,
	`row323`: 1,
	`row324`: 1,
	`row325`: 1,
	`row326`: 1,
	`row327`: 1,
	`row328`: 1,
	`row329`: 1,
	`row330`: 1,
	`row331`: 1,
	`row332`: 1,
	`row333`: 1,
	`row334`: 1,
	`row335`: 1,
	`row336`: 1,
	`row337`: 1,
	`row338`: 1,
	`row339`: 1,
	`row340`: 1,
	`row341`: 1,
	`row342`: 1,
	`row343`: 1,
	`row344`: 1,
	`row345`: 1,
	`row346`: 1,
	`row347`: 1,
	`row348`: 1,
	`row349`: 1,
	`row350`: 1,
	`row351`: 1,
	`row352`: 1,
	`row353`: 1,
	`row354`: 1,
	`row355`: 1,
	`row356`: 1,
	`row357`: 1,
	`row358`: 1,
	`row359`: 1,
	`row360`: 1,
	`row361`: 1,
	`row362`: 1,
	`row363`: 1,
	`row364`: 1,
	`row365`: 1,
	`row366`: 1,
	`row367`: 1,
	`row368`: 1,
	`row369`: 1,
	`row370`: 1,
	`row371`: 1,
	`row372`: 1,
	`row373`: 1,
	`row374`: 1,
	`row375`: 1,
	`row376`: 1,
	`row377`: 1,
	`row378`: 1,
	`row379`: 1,
	`row380`: 1,
	`row381`: 1,
	`row382`: 1,
	`row383`: 1,
	`row384`: 1,
	`row385`: 1,
	`row386`: 1,
	`row387`: 1,
	`row388`: 1,
	`row389`: 1,
	`row390`: 1,
	`row391`: 1,
	`row392`: 1,
	`row393`: 1,
	`row394`: 1,
	`row395`: 1,
	`row396`: 1,
	`row397`: 1,
	`row398`: 1,
	`row399`: 1,
	`row400`: 1,
	`row401`: 1,
	`row402`: 1,
	`row403`: 1,
	`row404`: 1,
	`row405`: 1,
	`row406`: 1,
	`row407`: 1,
	`row408`: 1,
	`row409`: 1,
	`row410`: 1,
	`row411`: 1,
	`row412`: 1,
	`row413`: 1,
	`row414`: 1,
	`row415`: 1,
	`row416`: 1,
	`row417`: 1,
	`row418`: 1,
	`row419`: 1,
	`row420`: 1,
	`row421`: 1,
	`row422`: 1,
	`row423`: 1,
	`row424`: 1,
	`row425`: 1,
	`row426`: 1,
	`row427`: 1,
	`row428`: 1,
	`row429`: 1,
	`row430`: 1,
	`row431`: 1,
	`row432`: 1,
	`row433`: 1,
	`row434`: 1,
	`row435`: 1,
	`row436`: 1,
	`row437`: 1,
	`row438`: 1,
	`row439`: 1,
	`row440`: 1,
	`row441`: 1,
	`row442`: 1,
	`row443`: 1,
	`row444`: 1,
	`row445`: 1,
	`row446`: 1,
	`row447`: 1,
	`row448`: 1,
	`row449`: 1,
	`row450`: 1,
	`row451`: 1,
	`row452`: 1,
	`row453`: 1,
	`row454`: 1,
	`row455`: 1,
	`row456`: 1,
	`row457`: 1,
	`row458`: 1,
	`row459`: 1,
	`row460`: 1,
	`row461`: 1,
	`row462`: 1,
	`row463`: 1,
	`row464`: 1,
	`row465`: 1,
	`row466`: 1,
	`row467`: 1,
	`row468`: 1,
	`row469`: 1,
	`row470`: 1,
	`row471`: 1,
	`row472`: 1,
	`row473`: 1,
	`row474`: 1,
	`row475`: 1,
	`row476`: 1,
	`row477`: 1,
	`row478`: 1,
	`row479`: 1,
	`row480`: 1,
	`row481`: 1,
	`row482`: 1,
	`row483`: 1,
	`row484`: 1,
	`row485`: 1,
	`row486`: 1,
	`row487`: 1,
	`row488`: 1,
	`row489`: 1,
	`row490`: 1,
	`row491`: 1,
	`row492`: 1,
	`row493`: 1,
	`row494`: 1,
	`row495`: 1,
	`row496`: 1,
	`row497`: 1,
	`row498`: 1,
	`row499`: 1,
	`row500`: 1,
	`row501`: 1,
	`row502`: 1,
	`row503`: 1,
	`row504`: 1,
	`row505`: 1,
	`row506`: 1,
	`row507`: 1,
	`row508`: 1,
	`row509`: 1,
	`row510`: 1,
	`row511`: 1,
	`row512`: 1,
	`row513`: 1,
	`row514`: 1,
	`row515`: 1,
	`row516`: 1,
	`row517`: 1,
	`row518`: 1,
	`row519`: 1,
	`row520`: 1,
	`row521`: 1,
	`row522`: 1,
	`row523`: 1,
	`row524`: 1,
	`row525`: 1,
	`row526`: 1,
	`row527`: 1,
	`row528`: 1,
	`row529`: 1,
	`row530`: 1,
	`row531`: 1,
	`row532`: 1,
	`row533`: 1,
	`row534`: 1,
	`row535`: 1,
	`row536`: 1,
	`row537`: 1,
	`row538`: 1,
	`row539`: 1,
	`row540`: 1,
	`row541`: 1,
	`row542`: 1,
	`row543`: 1,
	`row544`: 1,
	`row545`: 1,
	`row546`: 1,
	`row547`: 1,
	`row548`: 1,
	`row549`: 1,
	`row550`: 1,
	`row551`: 1,
	`row552`: 1,
	`row553`: 1,
	`row554`: 1,
	`row555`: 1,
	`row556`: 1,
	`row557`: 1,
	`row558`: 1,
	`row559`: 1,
	`row560`: 1,
	`row561`: 1,
	`row562`: 1,
	`row563`: 1,
	`row564`: 1,
	`row565`: 1,
	`row566`: 1,
	`row567`: 1,
	`row568`: 1,
	`row569`: 1,
	`row570`: 1,
	`row571`: 1,
	`row572`: 1,
	`row573`: 1,
	`row574`: 1,
	`row575`: 1,
	`row576`: 1,
	`row577`: 1,
	`row578`: 1,
	`row579`: 1,
	`row580`: 1,
	`row581`: 1,
	`row582`: 1,
	`row583`: 1,
	`row584`: 1,
	`row585`: 1,
	`row586`: 1,
	`row587`: 1,
	`row588`: 1,
	`row589`: 1,
	`row590`: 1,
	`row591`: 1,
	`row592`: 1,
	`row593`: 1,
	`row594`: 1,
	`row595`: 1,
	`row596`: 1,
	`row597`: 1,
	`row598`: 1,
	`row599`: 1,
	`row600`: 1,
	`row601`: 1,
	`row602`: 1,
	`row603`: 1,
	`row604`: 1,
	`row605`: 1,
	`row606`: 1,
	`row607`: 1,
	`row608`: 1,
	`row609`: 1,
	`row610`: 1,
	`row611`: 1,
	`row612`: 1,
	`row613`: 1,
	`row614`: 1,
	`row615`: 1,
	`row616`: 1,
	`row617`: 1,
	`row618`: 1,
	`row619`: 1,
	`row620`: 1,
	`row621`: 1,
	`row622`: 1,
	`row623`: 1,
	`row624`: 1,
	`row625`: 1,
	`row626`: 1,
	`row627`: 1,
	`row628`: 1,
	`row629`: 1,
	`row630`: 1,
	`row631`: 1,
	`row632`: 1,
	`row633`: 1,
	`row634`: 1,
	`row635`: 1,
	`row636`: 1,
	`row637`: 1,
	`row638`: 1,
	`row639`: 1,
	`row640`: 1,
	`row641`: 1,
	`row642`: 1,
	`row643`: 1,
	`row644`: 1,
	`row645`: 1,
	`row646`: 1,
	`row647`: 1,
	`row648`: 1,
	`row649`: 1,
	`row650`: 1,
	`row651`: 1,
	`row652`: 1,
	`row653`: 1,
	`row654`: 1,
	`row655`: 1,
	`row656`: 1,
	`row657`: 1,
	`row658`: 1,
	`row659`: 1,
	`row660`: 1,
	`row661`: 1,
	`row662`: 1,
	`row663`: 1,
	`row664`: 1,
	`row665`: 1,
	`row666`: 1,
	`row667`: 1,
	`row668`: 1,
	`row669`: 1,
	`row670`: 1,
	`row671`: 1,
	`row672`: 1,
	`row673`: 1,
	`row674`: 1,
	`row675`: 1,
	`row676`: 1,
	`row677`: 1,
	`row678`: 1,
	`row679`: 1,
	`row680`: 1,
	`row681`: 1,
	`row682`: 1,
	`row683`: 1,
	`row684`: 1,
	`row685`: 1,
	`row686`: 1,
	`row687`: 1,
	`row688`: 1,
	`row689`: 1,
	`row690`: 1,
	`row691`: 1,
	`row692`: 1,
	`row693`: 1,
	`row694`: 1,
	`row695`: 1,
	`row696`: 1,
	`row697`: 1,
	`row698`: 1,
	`row699`: 1,
	`row700`: 1,
	`row701`: 1,
	`row702`: 1,
	`row703`: 1,
	`row704`: 1,
	`row705`: 1,
	`row706`: 1,
	`row707`: 1,
	`row708`: 1,
	`row709`: 1,
	`row710`: 1,
	`row711`: 1,
	`row712`: 1,
	`row713`: 1,
	`row714`: 1,
	`row715`: 1,
	`row716`: 1,
	`row717`: 1,
	`row718`: 1,
	`row719`: 1,
	`row720`: 1,
	`row721`: 1,
	`row722`: 1,
	`row723`: 1,
	`row724`: 1,
	`row725`: 1,
	`row726`: 1,
	`row727`: 1,
	`row728`: 1,
	`row729`: 1,
	`row730`: 1,
	`row731`: 1,
	`row732`: 1,
	`row733`: 1,
	`row734`: 1,
	`row735`: 1,
	`row736`: 1,
	`row737`: 1,
	`row738`: 1,
	`row739`: 1,
	`row740`: 1,
	`row741`: 1,
	`row742`: 1,
	`row743`: 1,
	`row744`: 1,
	`row745`: 1,
	`row746`: 1,
	`row747`: 1,
	`row748`: 1,
	`row749`: 1,
	`row750`: 1,
	`row751`: 1,
	`row752`: 1,
	`row753`: 1,
	`row754`: 1,
	`row755`: 1,
	`row756`: 1,
	`row757`: 1,
	`row758`: 1,
	`row759`: 1,
	`row760`: 1,
	`row761`: 1,
	`row762`: 1,
	`row763`: 1,
	`row764`: 1,
	`row765`: 1,
	`row766`: 1,
	`row767`: 1,
	`row768`: 1,
	`row769`: 1,
	`row770`: 1,
	`row771`: 1,
	`row772`: 1,
	`row773`: 1,
	`row774`: 1,
	`row775`: 1,
	`row776`: 1,
	`row777`: 1,
}
