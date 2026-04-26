package main

import (
	"context"

	"github.com/ollama/ollama/api"
)


func queryModel(model string, modelContext []api.Message, client *api.Client) ([]api.Message, string, error) {
	req := api.ChatRequest{
		Model:    model,
		Messages: modelContext,
		Stream:   new(bool),
	}

	var resp api.ChatResponse
	err := client.Chat(context.Background(), &req, func(r api.ChatResponse) error {
		resp = r
		return nil
	})
	if err != nil {
		return modelContext, "", err
	}

	modelContext = append(modelContext, resp.Message)


	return modelContext, resp.Message.Content, nil
}

