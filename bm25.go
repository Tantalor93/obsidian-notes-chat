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
	lengths  []int
	tf       []map[string]int // per-document term frequencies
	postings map[string][]int // inverted index: term → doc indices containing it
	totalLen int              // running sum of all document lengths
	avgdl    float64
	n        int
}

func newBM25Index() *BM25Index {
	return &BM25Index{postings: make(map[string][]int)}
}

// Add indexes a document (must be called in the same order as VectorStore.Add).
func (idx *BM25Index) Add(text string) {
	docIdx := idx.n
	tokens := tokenize(text)
	freq := make(map[string]int, len(tokens))
	for _, t := range tokens {
		freq[t]++
	}
	for t := range freq {
		idx.postings[t] = append(idx.postings[t], docIdx)
	}
	idx.lengths = append(idx.lengths, len(tokens))
	idx.tf = append(idx.tf, freq)
	idx.totalLen += len(tokens)
	idx.n++
	idx.avgdl = float64(idx.totalLen) / float64(idx.n)
}

func (idx *BM25Index) scoreDoc(docIdx int, qt string, idf float64) float64 {
	dl := float64(idx.lengths[docIdx])
	f := float64(idx.tf[docIdx][qt])
	tf := f * (bm25K1 + 1) / (f + bm25K1*(1-bm25B+bm25B*dl/idx.avgdl))
	return idf * tf
}

// ranks returns a slice where ranks[i] is the 1-based BM25 rank of document i.
// Only documents containing at least one query token are scored; the rest are
// ranked last in an unspecified order.
func (idx *BM25Index) ranks(queryTokens []string) []int {
	scores := make(map[int]float64, 64)
	for _, qt := range queryTokens {
		docs, ok := idx.postings[qt]
		if !ok {
			continue
		}
		df := len(docs)
		idf := math.Log((float64(idx.n)-float64(df)+0.5)/(float64(df)+0.5) + 1)
		for _, docIdx := range docs {
			scores[docIdx] += idx.scoreDoc(docIdx, qt, idf)
		}
	}

	// Build a sorted list of (docIdx, score) for scored documents.
	type pair struct {
		idx   int
		score float64
	}
	scored := make([]pair, 0, len(scores))
	for i, s := range scores {
		scored = append(scored, pair{i, s})
	}
	sort.Slice(scored, func(a, b int) bool {
		return scored[a].score > scored[b].score
	})

	ranks := make([]int, idx.n)
	// Unscored docs default to rank 0; fill in after scored ones.
	nextRank := len(scored) + 1
	for i := range ranks {
		ranks[i] = nextRank
	}
	for rank, p := range scored {
		ranks[p.idx] = rank + 1
	}
	return ranks
}
