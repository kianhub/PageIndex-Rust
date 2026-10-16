"""Python side of the PageIndex vs Rust bench. Invoked by compare.rs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def median_ms(fn, repeats: int) -> float:
    samples = []
    fn()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def fake_completion(model, prompt, chat_history=None, return_finish_reason=False):
    del model, chat_history
    text = (
        '{"subsections": []}'
        if "subsection" in (prompt or "").lower()
        else '{"summary": "Deterministic bench summary."}'
    )
    if return_finish_reason:
        return text, "finished"
    return text


async def fake_acompletion(model, prompt, **params):
    del model, params
    return fake_completion(None, prompt)


def count_tokens_cl100k(text, model=None):
    del model
    if not text:
        return 0
    return len(enc.encode(text))


def install_llm_stubs():
    import pageindex.utils as utils

    utils.llm_completion = fake_completion
    utils.llm_acompletion = fake_acompletion
    utils.count_tokens = count_tokens_cl100k


def parse_pdf(path: str):
    from pageindex.flash.parser_pdfium_charlevel import parse_charlevel_meta

    return parse_charlevel_meta(path)


def tree_pdf(path: str):
    from pageindex.flash.api import page_index_flash

    return page_index_flash(path, summary=False, optimize=False)


def e2e_pdf(path: str):
    from pageindex.flash.api import page_index_flash

    return page_index_flash(
        path, summary=True, optimize="full", summary_model="gpt-bench", summary_reasoning_effort="none"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    docs = json.loads(Path(args.docs).read_text())
    token_text = Path(args.tokens).read_text()
    install_llm_stubs()

    results = {
        "token_count": len(enc.encode(token_text)),
        "token_ms": median_ms(lambda: len(enc.encode(token_text)), args.repeats * 4),
        "docs": {},
    }
    for doc in docs:
        name, path = doc["name"], doc["path"]
        parse_ms = median_ms(lambda p=path: parse_pdf(p), args.repeats)
        tree = None

        def run_tree(p=path):
            nonlocal tree
            tree = tree_pdf(p)
            return tree

        tree_ms = median_ms(run_tree, args.repeats)
        e2e_ms = median_ms(lambda p=path: e2e_pdf(p), max(3, args.repeats - 2))
        structure = (tree or {}).get("structure") or []
        results["docs"][name] = {
            "parse_ms": parse_ms,
            "tree_ms": tree_ms,
            "e2e_ms": e2e_ms,
            "pages": len((tree or {}).get("page_texts") or [])
            or sum(1 for _ in structure),
            "nodes": len(structure),
        }
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
