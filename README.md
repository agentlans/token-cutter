# TokenCutter

A stateful, high-performance, and boundary-safe Python text utility designed to slide, slice, count, and abridge text directly based on LLM token boundaries rather than arbitrary character steps.

Powered by `tiktoken`, **TokenCutter** caches text-to-token translations internally. This design pattern completely avoids cross-boundary decoding bugs (broken characters/corrupted emojis) when trimming context down for Large Language Model prompt windows.

## 🚀 Features

- **Boundary-Safe Truncation:** Drops, slices, or merges items at the token identifier level before execution, eliminating corrupted string multi-byte splits.
- **Internal Lazy-Caching:** Encodes strings on demand and safely maintains a token array cache. Consecutive counting, slicing, and trimming actions cost no additional compute.
- **Smart Abridging:** Supports structured context truncation (`left`, `right`, or `centre`) while strictly budgeting for the token size of custom trailing/leading ellipses (`...`).
- **Sliding Windows:** Includes memory-efficient generator processing loops to chunk long documentation strings using programmatic token sizes and custom overlap margins.

## 🛠️ Installation

You can install `token-cutter` directly from GitHub using `pip`. 

### Standard Installation

Ensure you have Python 3.8+ installed along with `tiktoken`.

To install the latest stable version of this package, run the following command in your terminal:

```bash
pip install git+https://github.com/agentlans/token-cutter.git
```

## 💡 Quick Start Guide

### Basic Token Operations & Caching

```python
from token_cutter import TokenCutter

# Initialize targeting a model (automatically falls back to cl100k_base if missing)
cutter = TokenCutter("Hello world! Artificial intelligence is fascinating.", model_name="gpt-4")

# Quick length/counting check (Caches tokens under-the-hood)
print(f"Token Length: {len(cutter)}")  # Output: 9
print(f"Token Count: {cutter.count()}") # Output: 9

# Dynamic mutations invalidate cache only when data actually mutates
cutter.text = "Hello world! Artificial intelligence is fascinating." # No re-encoding (same text)
cutter.text = "New context string to track."                        # Cache automatically cleared

```

### Context Truncation (`abridge`)

Safely guarantee your prompt text stays within a hard token threshold without breaking unicode structures:

```python
long_text = "The quick brown fox jumps over the lazy dog repeatedly until the context windows explode entirely."
cutter = TokenCutter(long_text, model_name="gpt-4")

# Trim right side with custom absolute token limit
print(cutter.abridge(max_tokens=10, position="right", ellipses="..."))
# "The quick brown fox jumps over the lazy dog..."

# Keep the beginning and end intact, squeezing out tokens from the middle
print(cutter.abridge(max_tokens=12, position="centre", ellipses="---"))
# "The quick brown fox jumps--- the context windows explode entirely."

```

### Sliding Windows for RAG

Chunk text records safely into overlapping arrays before indexing them inside Vector databases:

```python
cutter = TokenCutter("Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa", model_name="gpt-4")

# Eagerly generate chunks of size 4 with a 1-token overlap between boundaries
chunks = cutter.slide(size=4, overlap=1)
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx}: '{chunk}'")

```

## 📊 API Reference

### Properties

* `cutter.text`: Read/Write string content managed by the instance. Setting a unique value flushes the underlying cache automatically.
* `cutter.tokens`: Returns the calculated list of token integers (`List[int]`). Lazy loaded.

### Core Methods

* `count() -> int` / `__len__() -> int`: Returns total count of tokens.
* `head(num_tokens: int) -> str`: Decodes and returns the first $N$ tokens.
* `tail(num_tokens: int) -> str`: Decodes and returns the trailing $N$ tokens.
* `slice(start: int, end: int) -> str`: Extract any targeted ranges using integer indices.
* `slide_generator(size: int, overlap: int) -> Generator`: Yields window strings on demand (Memory safe).
* `slide(size: int, overlap: int) -> List[str]`: Evaluates sliding arrays eagerly into a list.
* `abridge(max_tokens: int, ellipses: str, position: str) -> str`: Enforces strict maximum token configurations on strings safely.

## 📄 Licence

This utility is available as open-source software under the MIT License

