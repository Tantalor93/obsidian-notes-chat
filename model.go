package main

import (
	"context"
	"time"

	"github.com/ollama/ollama/api"
)


func queryModel(model string, modelContext []api.Message, client *api.Client, useTools bool) ([]api.Message, string, error) {
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

	if len(resp.Message.ToolCalls) > 0 {
		return handleToolCalls(resp.Message.ToolCalls, modelContext, client, model)
	}

	return modelContext, resp.Message.Content, nil
}

func handleToolCalls(calls []api.ToolCall, modelContext []api.Message, client *api.Client, model string) ([]api.Message, string, error) {
	for _, call := range calls {
		var result string

		switch call.Function.Name {
		case "get_time":
			result = time.Now().Format("15:04:05")
		case "get_date":
			result = time.Now().Format("Monday 2006-01-02")
		default:
			panic("unknown tool: " + call.Function.Name)
		}

		modelContext = append(modelContext, api.Message{
			Role:    "tool",
			Content: result,
		})
	}

	return queryModel(model, modelContext, client, true)
}
