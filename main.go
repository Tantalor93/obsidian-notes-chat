package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/manifoldco/promptui"
	"github.com/ollama/ollama/api"
	"github.com/schollz/progressbar/v3"
)

func main() {
	ollamaServer := flag.String("url", "http://127.0.0.1:11434", "URL of the Ollama server")
	vaultDir := flag.String("vault", ".", "vault directory")
	flag.Parse()

	parsedURL, err := url.Parse(*ollamaServer)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid URL %s\n", *ollamaServer)
		os.Exit(1)
	}
	client := api.NewClient(parsedURL, http.DefaultClient)

	model, err := selectModel(client)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error selecting model:", err)
		os.Exit(1)
	}

	// RAG initialization
	store := initRag(vaultDir, err, client)

	const systemPrompt = `You are a helpful assistant that answers questions using the provided context from an Obsidian vault.
Rules:
- Answer ONLY the question asked, using ONLY the relevant parts of the context.
- If a note is not relevant to the question, ignore it completely.
- If the context is not sufficient to answer the question, respond with "I don't know".
- Do not mix information from unrelated notes in a single answer.`

	var modelContext []api.Message = []api.Message{
		{Role: "system", Content: systemPrompt},
	}

	scanner := bufio.NewScanner(os.Stdin)
	printUserPrompt()
	for scanner.Scan() {
		input := strings.TrimSpace(scanner.Text())
		if len(input) == 0 {
			continue
		}
		fmt.Println("-----")

		ragContext := ""
		var sources []string
		if store != nil {
			ragContext, sources = enrichContext(client, store, input)
		}

		var response string
		modelContext, response, err = query(modelContext, input, model, client, ragContext)
		if err != nil {
			fmt.Println("Error:", err)
			continue
		}

		printModelPrompt()
		fmt.Println(response)
		for _, s := range sources {
			obsidianLink := sourceToObsidianLink(s, "obsidian-vault")
			color.New(color.Faint).Printf("  - %s\n", obsidianLink)
		}
		fmt.Println("-----")
		printUserPrompt()
	}
}

func sourceToObsidianLink(path string, vault string) string {
     filename := strings.TrimSuffix(filepath.Base(path), ".md")
    
    // URL encode
    encodedVault := url.PathEscape(vault)
    encodedFile := url.PathEscape(filename)
    
    return fmt.Sprintf("obsidian://open?vault=%s&file=%s", encodedVault, encodedFile)
}

func initRag(vaultDir *string, err error, client *api.Client) *VectorStore {
	var store *VectorStore
	store, err = indexDirectory(client, *vaultDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error indexing:", err)
		os.Exit(1)
	}
	color.New(color.FgGreen).Printf("Indexing complete!\n")
	return store
}

func query(modelContext []api.Message, input string, model string, client *api.Client, ragContext string) ([]api.Message, string, error) {
	progress := progressbar.NewOptions(
		-1,
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionSetWidth(10),
		progressbar.OptionThrottle(65*time.Millisecond),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionFullWidth(),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionClearOnFinish(),
	)
	defer progress.Finish()

	userMessage := input
	if ragContext != "" {
		userMessage = ragContext + "\n\nQuestion: " + input
	}

	modelContext = append(modelContext, api.Message{
		Role:    "user",
		Content: userMessage,
	})

	return queryModel(model, modelContext, client, ragContext == "")
}

func printUserPrompt() {
	color.New(color.FgGreen).Print("> ")
}

func printModelPrompt() {
	color.New(color.FgYellow).Print("< ")
}

// selectModel let user select model from the list of models available on the Ollama server. It returns the name of the selected model.
func selectModel(client *api.Client) (string, error) {
	ctx := context.Background()

	resp, err := client.List(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to list models: %w", err)
	}

	if len(resp.Models) == 0 {
		return "", fmt.Errorf("no models available on the server")
	}

	var names []string
	embeddingAvailable := false
	for _, m := range resp.Models {
		if strings.HasPrefix(m.Name, embeddingModel) {
			embeddingAvailable = true
		} else {
			names = append(names, m.Name)
		}
	}
	if !embeddingAvailable {
		return "", fmt.Errorf("embedding model %q is not available — run: ollama pull %s", embeddingModel, embeddingModel)
	}
	if len(names) == 0 {
		return "", fmt.Errorf("no chat models available on the server")
	}

	prompt := promptui.Select{
		Label: "Select model",
		Items: names,
	}

	_, selected, err := prompt.Run()
	if err != nil {
		return "", err
	}

	return selected, nil
}
