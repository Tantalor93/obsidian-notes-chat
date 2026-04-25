# Notes chatbot

CLI chatbot to provide LLM capabilities over Obsidian notes.

## Why?

I have too many notes and it is getting difficult to find anything

## Local development

### Run Ollama server

install Ollama

```
brew install Ollama
```

run Ollama
```
ollama serve
```

pull model
```
ollama pull llama3.2
```


### Build & Run chatbot

```
go build
```

```
./notes-chat
```
