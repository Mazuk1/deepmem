from deepmem.extraction_filter import (
    is_pure_code_block,
    prepare_messages_for_fact_extraction,
    strip_large_code_blocks,
)


RUST_CODE = """```rust
fn solve_n_queens(n: usize) -> Vec<Vec<String>> {
    let mut board = vec![vec!['.'; n]; n];
    let mut cols = vec![false; n];
    let mut diag1 = vec![false; 2 * n];
    let mut diag2 = vec![false; 2 * n];
    let mut out = Vec::new();
    fn dfs(row: usize) {
        // lots of implementation details
    }
    out
}
```"""


def test_pure_large_code_block_is_skipped():
    assert is_pure_code_block(RUST_CODE) is True

    prepared, should_skip, stripped = prepare_messages_for_fact_extraction([
        {"role": "user", "content": RUST_CODE},
    ])

    assert prepared == []
    assert should_skip is True
    assert stripped is True


def test_mixed_text_and_code_is_sanitized_not_skipped():
    text = f"""From now on I want all algorithm problems written in Rust, here is an example:

{RUST_CODE}
"""

    prepared, should_skip, stripped = prepare_messages_for_fact_extraction([
        {"role": "user", "content": text},
    ])

    assert should_skip is False
    assert stripped is True
    assert "all algorithm problems written in Rust" in prepared[0]["content"]
    assert "[code block omitted]" in prepared[0]["content"]
    assert "solve_n_queens" not in prepared[0]["content"]


def test_small_code_snippet_is_not_treated_as_large_code_pollution():
    snippet = """```python
print('hello')
```"""
    assert is_pure_code_block(snippet) is False

    cleaned, changed = strip_large_code_blocks(snippet)
    assert changed is False
    assert "print('hello')" in cleaned


def test_multiple_user_messages_skip_only_when_all_user_messages_are_code_only():
    prepared, should_skip, stripped = prepare_messages_for_fact_extraction([
        {"role": "user", "content": RUST_CODE},
        {"role": "assistant", "content": "I see this code."},
        {"role": "user", "content": "I prefer all algorithm examples in Rust."},
    ])

    assert should_skip is False
    assert stripped is True
    text = "\n".join(m["content"] for m in prepared)
    assert "all algorithm examples in Rust" in text
    assert "solve_n_queens" not in text
