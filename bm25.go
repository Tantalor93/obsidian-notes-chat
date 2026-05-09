package main

import (
	"math"
	"regexp"
	"sort"
	"strings"
)

const (
	bm25K1 = 1.5
	bm25B  = 0.75
	rrfK   = 60
)

var tokenRe = regexp.MustCompile(`[a-z0-9]+`)

func tokenize(text string) []string {
	return tokenRe.FindAllString(strings.ToLower(text), -1)
}

// BM25Index is a simple in-memory BM25 index that mirrors the VectorStore document slice.
// Documents must be added in the same order as they are added to the VectorStore.
type BM25Index struct {
	lengths []int
	tf      []map[string]int // per-document term frequencies
	df      map[string]int   // document frequency per term
	avgdl   float64
	n       int
}

func newBM25Index() *BM25Index {
	return &BM25Index{df: make(map[string]int)}
}

// Add indexes a document (must be called in the same order as VectorStore.Add).
func (idx *BM25Index) Add(text string) {
	tokens := tokenize(text)
	freq := make(map[string]int, len(tokens))
	for _, t := range tokens {
		freq[t]++
	}
	for t := range freq {
		idx.df[t]++
	}
	idx.lengths = append(idx.lengths, len(tokens))
	idx.tf = append(idx.tf, freq)
	idx.n++

	total := 0
	for _, l := range idx.lengths {
		total += l
	}
	idx.avgdl = float64(total) / float64(idx.n)
}

func (idx *BM25Index) score(docIdx int, queryTokens []string) float64 {
	if idx.n == 0 || idx.avgdl == 0 {
		return 0
	}
	dl := float64(idx.lengths[docIdx])
	freq := idx.tf[docIdx]
	var s float64
	for _, qt := range queryTokens {
		df, ok := idx.df[qt]
		if !ok {
			continue
		}
		idf := math.Log((float64(idx.n)-float64(df)+0.5)/(float64(df)+0.5) + 1)
		f := float64(freq[qt])
		tf := f * (bm25K1 + 1) / (f + bm25K1*(1-bm25B+bm25B*dl/idx.avgdl))
		s += idf * tf
	}
	return s
}

// ranks returns a slice where ranks[i] is the 1-based BM25 rank of document i.
func (idx *BM25Index) ranks(queryTokens []string) []int {
	order := make([]int, idx.n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return idx.score(order[a], queryTokens) > idx.score(order[b], queryTokens)
	})
	ranks := make([]int, idx.n)
	for rank, docIdx := range order {
		ranks[docIdx] = rank + 1
	}
	return ranks
}
