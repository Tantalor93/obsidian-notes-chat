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
	docs []Document
}

func (vs *VectorStore) Add(doc Document) {
	vs.docs = append(vs.docs, doc)
}

func (vs *VectorStore) Search(queryEmbedding []float32, topK int) []ScoredDocument {
	scores := make([]ScoredDocument, len(vs.docs))
	for i, doc := range vs.docs {
		scores[i] = ScoredDocument{doc, cosineSimilarity(queryEmbedding, doc.Embedding)}
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	k := topK
	if k > len(scores) {
		k = len(scores)
	}

	result := make([]ScoredDocument, k)
	for i := range result {
		result[i] = scores[i]
	}
	return result
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
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
	store := &VectorStore{}

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

	relevant := store.Search(queryEmbedding, 5)
	if len(relevant) == 0 {
		return "", nil
	}

	ragContext := "\n\nRelevant parts:\n\n"
	seen := map[string]bool{}

	for _, doc := range relevant {
		ragContext += fmt.Sprintf("// File: %s\n%s\n\n", doc.Document.Source, doc.Document.Content)
		if !seen[doc.Document.Source] {
			seen[doc.Document.Source] = true
		}
	}

	return ragContext, relevant
}
