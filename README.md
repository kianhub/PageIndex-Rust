# PageIndex-Rust

A Rust rewrite of [PageIndex](https://github.com/VectifyAI/PageIndex), the vectorless,
reasoning-based RAG indexer. It turns a PDF or Markdown document into a table-of-contents
tree (`title`, `node_id`, `start_index`, `end_index`, `summary`, `nodes`) that an LLM can
navigate instead of a vector index.

- **Flash** (default): builds the tree from PDF layout statistics, with no LLM involved.
  The parser reads each character with its font, size, and position, then groups text into
  lines and blocks. It then detects headers, footers, contents pages, the title, and
  headings, and assigns heading levels. When the PDF's embedded bookmarks are trustworthy,
  they frame the tree.
- **Optimize**: a deterministic merge driven by worst-case search cost, then an optional
  LLM expand pass.
- **Summaries**: optional LLM node summaries, written bottom-up.
- **Markdown**: `#` headings become nodes, with optional thinning and summaries.

The workspace has three crates:

| Crate | Contents |
|---|---|
| `pageindex-pdf` | Character-level PDF extraction (pure Rust, built on `lopdf`) and bookmark reading |
| `pageindex` | The library: Flash, tree optimization, summaries, Markdown, config, LLM routing, local client and store |
| `pageindex-cli` | The `pageindex` binary, equivalent to upstream's `run_pageindex.py` |

## Build and run

```bash
cargo build --release
# Structure only, no API key needed:
./target/release/pageindex --pdf_path paper.pdf --no-summary --optimize off
# Default run: merge, LLM expand, and summaries (needs a model key):
./target/release/pageindex --pdf_path paper.pdf
./target/release/pageindex --md_path notes.md
```

Results are written to `./results/<name>_structure.json`. The flags match
`run_pageindex.py`; run `pageindex --help` for the full list.

Library use:

```rust
let tree = pageindex::page_index_flash("paper.pdf", &pageindex::FlashOptions::structure_only()).await?;
```

## Configuration

Defaults live in `crates/pageindex/config.yaml`, the same keys and values as upstream.
Model names follow LiteLLM conventions: a bare name is an OpenAI model, and any other
provider is written as `provider/model` (for example `anthropic/claude-sonnet-5-5`,
`openrouter/...`, or `ollama/llama3`).

Secrets come only from the environment or a local `.env` file, which is never committed:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` (legacy: `CHATGPT_API_KEY`) | OpenAI models (the default) |
| `OPENAI_BASE_URL` | Any OpenAI-compatible server |
| `ANTHROPIC_API_KEY` | `anthropic/...` |
| `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `TOGETHERAI_API_KEY`, `FIREWORKS_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY` | Their providers |
| `OLLAMA_API_BASE` | Local Ollama (no key needed) |

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Benchmarks

Placeholder: Python-vs-Rust numbers on the same inputs will be filled in when the
benchmark suite lands.

## License

MIT, like upstream.
