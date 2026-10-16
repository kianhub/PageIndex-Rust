//! `pageindex`: index a PDF or Markdown document and write
//! `./results/<name>_structure.json` (the `run_pageindex.py` equivalent).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use pageindex::chatgpt::{ChatGptClient, chatgpt_model, open_browser, redact_authorize_url};
use pageindex::classic::{ClassicOptions, page_index};
use pageindex::config::parse_yes_no;
use pageindex::flash::{FlashOptions, flash_rejection_reason, page_index_flash};
use pageindex::markdown::{MdOptions, md_to_tree};
use pageindex::summary::SUMMARY_MAX_WORDS;
use pageindex::tree_optimize::EXPAND_CONCURRENCY;
use pageindex::{Config, ConfigLoader, ConfigOverrides, LiteLlm, LlmClient, OptimizeMode};

#[derive(Debug, Parser)]
#[command(name = "pageindex", about = "Process PDF or Markdown document and generate structure")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    index: IndexArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Opt-in Sign in with ChatGPT (plan usage via the Responses API)
    #[command(subcommand)]
    Chatgpt(ChatgptAction),
}

#[derive(Debug, Subcommand)]
enum ChatgptAction {
    /// Continue with ChatGPT (PKCE loopback login)
    Login,
    /// Revoke the session and clear local tokens
    Logout,
    /// Show whether ChatGPT plan usage is connected
    Status,
}

#[derive(Debug, clap::Args)]
struct IndexArgs {
    /// Path to the PDF file
    #[arg(long = "pdf_path", alias = "pdf-path")]
    pdf_path: Option<PathBuf>,
    /// Path to the Markdown file
    #[arg(long = "md_path", alias = "md-path")]
    md_path: Option<PathBuf>,
    /// Processing mode
    #[arg(long, value_parser = ["flash", "standard"], default_value = "flash")]
    mode: String,
    #[arg(long, hide = true)]
    flash: bool,

    /// Use the PDF's embedded bookmarks when trustworthy (default: on in flash mode)
    #[arg(long = "embedded-toc", overrides_with = "no_embedded_toc")]
    embedded_toc: bool,
    #[arg(long = "no-embedded-toc", hide = true)]
    no_embedded_toc: bool,
    /// Generate node summaries with an LLM (default: on in flash mode)
    #[arg(long, overrides_with = "no_summary")]
    summary: bool,
    #[arg(long = "no-summary", hide = true)]
    no_summary: bool,
    /// Refine the tree for search cost (default: full in flash mode); `merge` for deterministic merge only; `off` to disable
    #[arg(long, num_args = 0..=1, default_missing_value = "full", value_parser = ["full", "merge", "off"])]
    optimize: Option<String>,

    /// Model used to index the document (overrides config.yaml)
    #[arg(long = "index-model")]
    index_model: Option<String>,
    /// (legacy) Same as --index-model
    #[arg(long)]
    model: Option<String>,
    /// Model for node summaries (falls back to config.yaml summary_model, then --index-model, then --model)
    #[arg(long = "summary-model")]
    summary_model: Option<String>,
    /// Word cap for each model-written node summary (flash mode)
    #[arg(long = "summary-max-words", help = format!("Word cap for each model-written node summary; a leaf node shorter than that keeps its own text (flash mode; default {SUMMARY_MAX_WORDS})"))]
    summary_max_words: Option<usize>,
    #[arg(long = "summary-concurrency", help = format!("Cap on simultaneous indexing model calls per lane (flash mode; default 64, expand tops out at {EXPAND_CONCURRENCY})"))]
    summary_concurrency: Option<usize>,
    /// Reasoning effort sent with each node-summary call (flash mode; default none)
    #[arg(long = "summary-reasoning-effort")]
    summary_reasoning_effort: Option<String>,

    /// Number of pages to check for table of contents (PDF only)
    #[arg(long = "toc-check-pages")]
    toc_check_pages: Option<usize>,
    /// Maximum number of pages per node (PDF only)
    #[arg(long = "max-pages-per-node")]
    max_pages_per_node: Option<usize>,
    /// Maximum number of tokens per node (PDF only)
    #[arg(long = "max-tokens-per-node")]
    max_tokens_per_node: Option<usize>,

    /// Whether to add node id to the node
    #[arg(long = "if-add-node-id")]
    if_add_node_id: Option<String>,
    /// Whether to add summary to the node
    #[arg(long = "if-add-node-summary")]
    if_add_node_summary: Option<String>,
    /// Whether to add doc description to the doc
    #[arg(long = "if-add-doc-description")]
    if_add_doc_description: Option<String>,
    /// Whether to add text to the node
    #[arg(long = "if-add-node-text")]
    if_add_node_text: Option<String>,

    /// Whether to apply tree thinning for markdown (markdown only)
    #[arg(long = "if-thinning", default_value = "no")]
    if_thinning: String,
    /// Minimum token threshold for thinning (markdown only)
    #[arg(long = "thinning-threshold", default_value_t = 5000)]
    thinning_threshold: usize,
    /// Token threshold for generating summaries (markdown only)
    #[arg(long = "summary-token-threshold", default_value_t = 200)]
    summary_token_threshold: usize,
    /// Use Sign in with ChatGPT (`chatgpt/` models) instead of an API key
    #[arg(long)]
    chatgpt: bool,
}

impl IndexArgs {
    fn flag(on: bool, off: bool) -> Option<bool> {
        match (on, off) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        }
    }

    fn embedded_toc(&self) -> Option<bool> {
        Self::flag(self.embedded_toc, self.no_embedded_toc)
    }

    fn summary(&self) -> Option<bool> {
        Self::flag(self.summary, self.no_summary)
    }

    fn overrides(&self) -> ConfigOverrides {
        ConfigOverrides {
            model: self.model.clone(),
            index_model: self.index_model.clone(),
            summary_model: self.summary_model.clone(),
            toc_check_page_num: self.toc_check_pages,
            max_page_num_each_node: self.max_pages_per_node,
            max_token_num_each_node: self.max_tokens_per_node,
            if_add_node_id: self.if_add_node_id.as_deref().and_then(parse_yes_no),
            if_add_node_summary: self.if_add_node_summary.as_deref().and_then(parse_yes_no),
            if_add_doc_description: self.if_add_doc_description.as_deref().and_then(parse_yes_no),
            if_add_node_text: self.if_add_node_text.as_deref().and_then(parse_yes_no),
            ..Default::default()
        }
    }

    /// The same refusals `run_pageindex.py` raises before doing any work.
    fn validate(&mut self) -> Result<(), String> {
        if self.flash {
            self.mode = "flash".into();
        }
        match (&self.pdf_path, &self.md_path) {
            (None, None) => return Err("Either --pdf_path or --md_path must be specified".into()),
            (Some(_), Some(_)) => return Err("Only one of --pdf_path or --md_path can be specified".into()),
            _ => {}
        }
        let flash_pdf = self.pdf_path.is_some() && self.mode == "flash";
        let flash_only = [
            ("--optimize", self.optimize.is_some()),
            ("--embedded-toc", self.embedded_toc().is_some()),
            ("--summary", self.summary().is_some()),
            ("--summary-max-words", self.summary_max_words.is_some()),
            ("--summary-concurrency", self.summary_concurrency.is_some()),
            ("--summary-reasoning-effort", self.summary_reasoning_effort.is_some()),
        ];
        if let Some((flag, _)) = flash_only.iter().find(|(_, set)| *set && !flash_pdf) {
            return Err(format!("{flag} requires Flash mode with --pdf_path"));
        }
        if flash_pdf {
            let standard_only = [
                ("--toc-check-pages", self.toc_check_pages.is_some()),
                ("--max-pages-per-node", self.max_pages_per_node.is_some()),
                ("--max-tokens-per-node", self.max_tokens_per_node.is_some()),
                ("--if-add-node-id", self.if_add_node_id.is_some()),
                ("--if-add-node-summary", self.if_add_node_summary.is_some()),
                ("--if-add-doc-description", self.if_add_doc_description.is_some()),
                ("--if-add-node-text", self.if_add_node_text.is_some()),
            ];
            if let Some((flag, _)) = standard_only.iter().find(|(_, set)| *set) {
                return Err(format!("{flag} is not supported in flash mode; use --mode standard"));
            }
        }
        for (flag, value) in
            [("--summary-max-words", self.summary_max_words), ("--summary-concurrency", self.summary_concurrency)]
        {
            if value == Some(0) {
                return Err(format!("{flag} must be a positive int"));
            }
        }
        Ok(())
    }
}

fn output_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    PathBuf::from("./results").join(format!("{stem}_structure.json"))
}

fn write_result(input: &Path, value: &impl serde::Serialize) -> Result<PathBuf, String> {
    let out = output_path(input);
    std::fs::create_dir_all("./results").map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(&out, json).map_err(|e| e.to_string())?;
    Ok(out)
}

fn apply_chatgpt(config: &mut Config) -> Result<(), String> {
    config.index_model = chatgpt_model(&config.index_model).map_err(|e| e.to_string())?;
    config.summary_model = chatgpt_model(&config.summary_model).map_err(|e| e.to_string())?;
    Ok(())
}

async fn chatgpt_cmd(action: ChatgptAction) -> Result<(), String> {
    let client = ChatGptClient::from_env();
    match action {
        ChatgptAction::Status => {
            println!("{}", client.status().map_err(|e| e.to_string())?.render());
            Ok(())
        }
        ChatgptAction::Logout => {
            let out = client.logout().await.map_err(|e| e.to_string())?;
            println!("{}", out.message);
            Ok(())
        }
        ChatgptAction::Login => {
            println!("Continue with ChatGPT");
            let flow = client.begin_login().await.map_err(|e| e.to_string())?;
            let printable = redact_authorize_url(&flow.authorize_url);
            if open_browser(&flow.authorize_url).is_err() {
                println!("Open this URL in your browser:\n{printable}");
            } else {
                println!("If the browser does not open, visit:\n{printable}");
            }
            println!("Waiting for the loopback callback on {} …", flow.redirect_uri);
            let account = flow.finish(&client, Duration::from_secs(600)).await.map_err(|e| e.to_string())?;
            let who = account.email.as_deref().unwrap_or("ChatGPT");
            println!("Signed in as {who}.");
            if !account.plan_usage_enabled() {
                println!("ChatGPT plan usage was not granted. Re-run login and approve it, or set OPENAI_API_KEY.");
            }
            Ok(())
        }
    }
}

async fn run_pdf(args: &IndexArgs, pdf: &Path) -> Result<(), String> {
    let has_pdf_ext = pdf.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
    if !has_pdf_ext {
        return Err("PDF file must have .pdf extension".into());
    }
    if !pdf.is_file() {
        return Err(format!("PDF file not found: {}", pdf.display()));
    }
    if args.mode == "standard" {
        let mut config = ConfigLoader::new().load(&args.overrides());
        if args.chatgpt {
            apply_chatgpt(&mut config)?;
        }
        let llm = LiteLlm::new();
        let result = page_index(
            pdf,
            ClassicOptions {
                model: Some(config.index_model),
                toc_check_page_num: args.toc_check_pages,
                max_page_num_each_node: args.max_pages_per_node,
                max_token_num_each_node: args.max_tokens_per_node,
                if_add_node_id: args.if_add_node_id.as_deref().and_then(parse_yes_no),
                if_add_node_summary: args.if_add_node_summary.as_deref().and_then(parse_yes_no),
                if_add_doc_description: args.if_add_doc_description.as_deref().and_then(parse_yes_no),
                if_add_node_text: args.if_add_node_text.as_deref().and_then(parse_yes_no),
            },
            &llm,
        )
        .await
        .map_err(|e| e.to_string())?;
        println!("Parsing done, saving to file...");
        let out = write_result(pdf, &result)?;
        println!("Tree structure saved to: {}", out.display());
        return Ok(());
    }
    let mut config = ConfigLoader::new().load(&ConfigOverrides {
        summary_model: args.summary_model.clone(),
        index_model: args.index_model.clone(),
        model: args.model.clone(),
        ..Default::default()
    });
    if args.chatgpt {
        apply_chatgpt(&mut config)?;
    }
    let optimize = match args.optimize.as_deref() {
        None | Some("full") => OptimizeMode::Full,
        Some("merge") => OptimizeMode::Merge,
        _ => OptimizeMode::Off,
    };
    let opts = FlashOptions {
        summary: args.summary().unwrap_or(true),
        summary_model: Some(config.summary_model.clone()),
        optimize,
        optimize_model: Some(config.summary_model),
        summary_concurrency: args.summary_concurrency,
        use_embedded_toc: args.embedded_toc().unwrap_or(true),
        summary_max_words: args.summary_max_words,
        summary_reasoning_effort: args.summary_reasoning_effort.clone(),
        llm: Some(Arc::new(LiteLlm::new())),
    };
    let result = page_index_flash(pdf, &opts).await.map_err(|e| e.to_string())?;
    if let Some(reason) = flash_rejection_reason(&result.structure, result.toc_source, "--mode standard") {
        return Err(reason);
    }
    if let Some(report) = &result.optimize {
        println!(
            "Optimize: merges={} expands={}, worst-case search cost {} -> {} pages",
            report.merges,
            report.expands,
            report.before.worst_case_search_complexity,
            report.after.worst_case_search_complexity
        );
    }
    println!("Parsing done, saving to file...");
    let out = write_result(pdf, &result)?;
    println!("Tree structure saved to: {}", out.display());
    Ok(())
}

async fn run_markdown(args: &IndexArgs, md: &Path) -> Result<(), String> {
    let ext = md.extension().map(|e| e.to_string_lossy().to_lowercase());
    if !matches!(ext.as_deref(), Some("md" | "markdown")) {
        return Err("Markdown file must have .md or .markdown extension".into());
    }
    if !md.is_file() {
        return Err(format!("Markdown file not found: {}", md.display()));
    }
    println!("Processing markdown file...");
    let mut config = ConfigLoader::new().load(&args.overrides());
    if args.chatgpt {
        apply_chatgpt(&mut config)?;
    }
    let yes = |v: &Option<String>| v.as_deref().is_some_and(|s| s.eq_ignore_ascii_case("yes"));
    let opts = MdOptions {
        if_thinning: args.if_thinning.eq_ignore_ascii_case("yes"),
        min_token_threshold: args.thinning_threshold,
        if_add_node_summary: yes(&args.if_add_node_summary),
        summary_token_threshold: args.summary_token_threshold,
        summary_model: config.summary_model,
        if_add_doc_description: yes(&args.if_add_doc_description),
        if_add_node_text: yes(&args.if_add_node_text),
        if_add_node_id: yes(&args.if_add_node_id),
    };
    let llm = LiteLlm::new();
    let llm: Option<&dyn LlmClient> = opts.if_add_node_summary.then_some(&llm as &dyn LlmClient);
    let result = md_to_tree(md, &opts, llm).await.map_err(|e| e.to_string())?;
    println!("Parsing done, saving to file...");
    let out = write_result(md, &result)?;
    println!("Tree structure saved to: {}", out.display());
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = if let Some(Command::Chatgpt(action)) = cli.command {
        chatgpt_cmd(action).await
    } else {
        let mut args = cli.index;
        match args.validate() {
            Err(err) => Err(err),
            Ok(()) => match (args.pdf_path.clone(), args.md_path.clone()) {
                (Some(pdf), _) => run_pdf(&args, &pdf).await,
                (_, Some(md)) => run_markdown(&args, &md).await,
                _ => unreachable!("validate requires one input"),
            },
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Error: {err}");
            ExitCode::FAILURE
        }
    }
}
