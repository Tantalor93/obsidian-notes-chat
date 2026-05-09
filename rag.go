package main

import (
	"context"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/fatih/color"
	"github.com/ollama/ollama/api"
)

const (
	embeddingModel = "nomic-embed-text"
	docPrefix      = "search_document: "
	queryPrefix    = "search_query: "
)

type ScoredDocument struct {
	Document Document
	Score    float32
}

type Document struct {
	Content   string
	Source    string
	Embedding []float32
}

type VectorStore struct {
	docs  []Document
	norms []float32 // precomputed L2 norms for each document embedding
	bm25  *BM25Index
}

func newVectorStore() *VectorStore {
	return &VectorStore{bm25: newBM25Index()}
}

func (vs *VectorStore) Add(doc Document) {
	vs.docs = append(vs.docs, doc)
	vs.norms = append(vs.norms, l2Norm(doc.Embedding))
	vs.bm25.Add(doc.Content)
}

// Search performs hybrid retrieval using Reciprocal Rank Fusion (RRF) over
// dense cosine-similarity ranks and sparse BM25 ranks.
func (vs *VectorStore) Search(queryEmbedding []float32, query string, topK int) []ScoredDocument {
	n := len(vs.docs)
	if n == 0 {
		return nil
	}

	// Compute 1-based dense ranks (highest cosine similarity = rank 1).
	// Query norm is constant across all comparisons so compute it once.
	queryNorm := l2Norm(queryEmbedding)
	denseRanks := make([]int, n)

	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		// Cosine similarity: dot(q, d) / (|q| * |d|). Ranges 0–1; higher = more semantically similar.
		// queryNorm is precomputed once above; vs.norms[i] was precomputed at index time.
		scoreA := dotProduct(queryEmbedding, vs.docs[order[a]].Embedding) / (queryNorm * vs.norms[order[a]])
		scoreB := dotProduct(queryEmbedding, vs.docs[order[b]].Embedding) / (queryNorm * vs.norms[order[b]])
		return scoreA > scoreB
	})
	for rank, idx := range order {
		denseRanks[idx] = rank + 1
	}

	// Compute 1-based BM25 ranks.
	sparseRanks := vs.bm25.ranks(tokenize(query))

	// Fuse with Reciprocal Rank Fusion: score = 1/(k+r_dense) + 1/(k+r_sparse).
	results := make([]ScoredDocument, n)
	for i, doc := range vs.docs {
		rrf := 1.0/float64(rrfK+denseRanks[i]) + 1.0/float64(rrfK+sparseRanks[i])
		results[i] = ScoredDocument{doc, float32(rrf)}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topK > n {
		topK = n
	}
	return results[:topK]
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func l2Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

func getEmbedding(client *api.Client, text string) ([]float32, error) {
	req := api.EmbeddingRequest{
		Model:  embeddingModel,
		Prompt: text,
	}

	resp, err := client.Embeddings(context.Background(), &req)
	if err != nil {
		return nil, err
	}

	result := make([]float32, len(resp.Embedding))
	for i, v := range resp.Embedding {
		result[i] = float32(v)
	}
	return result, nil
}

func indexDirectory(client *api.Client, dir string) (*VectorStore, error) {
	store := newVectorStore()

	color.New(color.Faint).Printf("Indexing %s...\n", dir)

	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() && strings.HasPrefix(d.Name(), ".") {
			return fs.SkipDir
		}

		if d.IsDir() || !strings.EqualFold(filepath.Ext(path), ".md") {
			return nil
		}

		filename := strings.TrimSuffix(d.Name(), filepath.Ext(d.Name()))

		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		chunks := chunkTextWithOverlap(string(content), 1000, 200)
		color.New(color.Faint).Printf("\tindexing %s (%d chunks)...\n", path, len(chunks))
		for _, chunk := range chunks {
			textToEmbed := fmt.Sprintf(docPrefix+"%s\n\n%s", filename, chunk)

			embedding, err := getEmbedding(client, textToEmbed)
			if err != nil {
				return err
			}
			store.Add(Document{
				Content:   chunk,
				Source:    path,
				Embedding: embedding,
			})
		}
		return nil
	})

	return store, err
}

func chunkTextWithOverlap(text string, chunkSize int, overlap int) []string {
	var chunks []string
	lines := strings.Split(text, "\n")

	var current []string
	currentLen := 0

	for _, line := range lines {
		current = append(current, line)
		currentLen += len(line) + 1 // +1 for newline

		if currentLen >= chunkSize {
			chunks = append(chunks, strings.Join(current, "\n"))

			var overlapLines []string
			overlapLen := 0
			for i := len(current) - 1; i >= 0; i-- {
				overlapLen += len(current[i]) + 1
				overlapLines = append([]string{current[i]}, overlapLines...)
				if overlapLen >= overlap {
					break
				}
			}
			current = overlapLines
			currentLen = overlapLen
		}
	}

	if len(current) > 0 {
		chunks = append(chunks, strings.Join(current, "\n"))
	}

	return chunks
}

func enrichContext(client *api.Client, store *VectorStore, question string) (string, []ScoredDocument) {
	queryEmbedding, err := getEmbedding(client, queryPrefix+question)
	if err != nil {
		return "", nil
	}

	relevant := store.Search(queryEmbedding, question, 5)

	ragContext := "\n\nRelevant parts:\n\n"
	seen := map[string]bool{}
	var used []ScoredDocument

	for _, doc := range relevant {
		ragContext += fmt.Sprintf("// File: %s\n%s\n\n", doc.Document.Source, doc.Document.Content)
		if !seen[doc.Document.Source] {
			seen[doc.Document.Source] = true
		}
		used = append(used, doc)
	}

	if len(used) == 0 {
		return "", nil
	}

	return ragContext, used
}
