# Ollama CLI Quick Reference

This file summarizes the output of `ollama help` for convenience.
Ollama is a local LLM server and model manager. The CLI accepts the
following top level commands:

```
ollama help        Show general help or command specific help
ollama run MODEL   Run a model locally
ollama pull MODEL  Download a model from the registry
ollama push MODEL  Upload a model to the registry
ollama create NAME Create a model from a Modelfile
ollama list        Show installed models
ollama show MODEL  Display details about a model
ollama cp SRC DST  Copy or rename a model
ollama rm MODEL    Remove a model
ollama serve       Start the Ollama API server
ollama login       Authenticate with the Ollama registry
```

Common flags include:

```
--host <addr>         API server address (default http://localhost:11434)
--verbose             Enable debug logs
--help                Show help for a command
--version             Display the Ollama version
```

For more information see the [official Ollama documentation](https://github.com/ollama/ollama/tree/main/docs).

