import tiktoken
from typing import List, Optional, Generator, Literal

class TokenCutter:
    """A stateful, memory-efficient utility for tokenizing, slicing, sliding, 
    and abridging text strings based on LLM token counts rather than character counts.

    By maintaining state and internal token caching, this class avoids redundant 
    encoding operations, making it highly suitable for applications processing large 
    prompts, context windows, or continuous text fragments through tiktoken.

    Attributes:
        encoder: The tiktoken Encoding instance used for tokenization.
    """

    def __init__(self, text: str = "", model_name: str = "gpt-4"):
        """Initializes the TokenCutter with a target text and a designated tokenizer model.

        Args:
            text: The initial string content to manage and manipulate. Defaults to "".
            model_name: The OpenAI model name used to resolve the correct tokenizer encoding. 
                Falls back to 'cl100k_base' if the exact model name is unrecognized. 
                Defaults to "gpt-4".
        """
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            
        self._text = text
        self._tokens: Optional[List[int]] = None
        
        # Micro-cache for ellipsis sequences
        self._cached_ellipsis: str = ""
        self._cached_ellipsis_tokens: List[int] = []

    @property
    def text(self) -> str:
        """Gets the current raw text managed by the cutter.

        Returns:
            str: The underlying string content.
        """
        return self._text

    @text.setter
    def text(self, new_text: str) -> None:
        """Sets new raw text content, safely invalidating the token cache only if the text changes.

        Args:
            new_text: The new string content to manage.
        """
        if self._text != new_text:
            self._text = new_text
            self._tokens = None  # Clear cache only if text actually changes

    @property
    def tokens(self) -> List[int]:
        """Lazy-loads, caches, and returns the tokenized array of the text.

        Returns:
            List[int]: A list of integer token IDs corresponding to the text string.
        """
        if self._tokens is None:
            self._tokens = self.encoder.encode(self._text)
        return self._tokens

    def clear_cache(self) -> None:
        """Manually flushes the internal token array cache from memory while preserving the text.

        Use this method to optimize memory usage when handling exceptionally large texts 
        that no longer require slicing or counting modifications.
        """
        self._tokens = None

    def __len__(self) -> int:
        """Returns the total token count of the underlying text.

        Enables standard Pythonic length syntax: `len(cutter)`.

        Returns:
            int: The total count of tokens.
        """
        return len(self.tokens)

    def count(self) -> int:
        """Returns the total token count of the target text.

        Returns:
            int: The total count of tokens.
        """
        return len(self.tokens)

    def head(self, num_tokens: int) -> str:
        """Extracts a slice of text containing the first N tokens.

        Args:
            num_tokens: The exact number of tokens to include from the beginning of the text.

        Returns:
            str: The decoded string representation of the leading tokens. Returns an 
                empty string if `num_tokens` <= 0.
        """
        if num_tokens <= 0:
            return ""
        return self.encoder.decode(self.tokens[:num_tokens])

    def tail(self, num_tokens: int) -> str:
        """Extracts a slice of text containing the last N tokens safely.

        Args:
            num_tokens: The exact number of tokens to include from the end of the text.

        Returns:
            str: The decoded string representation of the trailing tokens. Returns an 
                empty string if `num_tokens` <= 0.
        """
        if num_tokens <= 0:
            return ""
        return self.encoder.decode(self.tokens[-num_tokens:])

    def slice(self, start: int, end: int) -> str:
        """Extracts a specific slice of the text based on token index ranges.

        Args:
            start: The starting token index (inclusive).
            end: The ending token index (exclusive).

        Returns:
            str: The decoded string slice mapped from the given token range.
        """
        return self.encoder.decode(self.tokens[start:end])

    def slide_generator(self, size: int, overlap: int = 0) -> Generator[str, None, None]:
        """A memory-efficient generator that yields text chunks using a token-based sliding window.

        Args:
            size: The maximum number of tokens contained within each sliding chunk window.
            overlap: The number of tokens to duplicate across overlapping contiguous chunks. 
                Must be non-negative and strictly smaller than `size`. Defaults to 0.

        Yields:
            Generator[str, None, None]: String chunks representing successive views of 
                the sliding window.

        Raises:
            ValueError: If `size` is 0 or negative, or if `overlap` is negative or 
                greater than or equal to `size`.
        """
        if size <= 0:
            raise ValueError("Window size must be greater than 0.")
        if overlap >= size or overlap < 0:
            raise ValueError("Overlap must be non-negative and strictly less than window size.")

        tokens = self.tokens
        total = len(tokens)
        if total == 0:
            return

        step = size - overlap
        decode_tokens = self.encoder.decode
        
        for i in range(0, total, step):
            yield decode_tokens(tokens[i : i + size])
            if i + size >= total:
                break

    def slide(self, size: int, overlap: int = 0) -> List[str]:
        """Eagerly evaluates the sliding window generator into a concrete list of string chunks.

        Args:
            size: The maximum number of tokens contained within each sliding chunk window.
            overlap: The number of tokens to duplicate across consecutive chunks. Defaults to 0.

        Returns:
            List[str]: A list containing all generated text chunk strings.
        """
        return list(self.slide_generator(size, overlap))

    def abridge(
        self, 
        max_tokens: int, 
        ellipses: str = "...", 
        position: Literal["centre", "left", "right"] = "centre"
    ) -> str:
        """Trims text down to fit securely within a strict maximum token budget.

        This method acts directly on the underlying integer tokens before executing a final 
        re-decoding pass, removing cross-boundary multi-byte character corruption bugs 
        commonly found when truncating texts purely via raw character slicing.

        Args:
            max_tokens: The total absolute budget allowed for the output token stream, 
                inclusive of the ellipsis sequence tokens.
            ellipses: The visual indicator string representing truncated content. Defaults to "...".
            position: Where to apply truncation and insert the ellipsis markers. 
                Options include:
                - 'right': Truncates trailing tokens and appends the ellipsis to the end.
                - 'left': Truncates leading tokens and prepends the ellipsis to the start.
                - 'centre': Splices out tokens from the middle, flanking the ellipsis 
                  evenly with balanced remaining head and tail tokens.
                Defaults to "centre".

        Returns:
            str: The structural safely truncated text string. Returns the unmodified 
                source text if its natural token footprint already satisfies the `max_tokens` budget.

        Raises:
            ValueError: If `max_tokens` budget is too small to even accommodate the encoded 
                ellipsis footprint, or if an unrecognized string literal is passed into `position`.
        """
        tokens = self.tokens
        if len(tokens) <= max_tokens:
            return self._text
            
        # Manage ellipsis cache
        if self._cached_ellipsis != ellipses:
            self._cached_ellipsis = ellipses
            self._cached_ellipsis_tokens = self.encoder.encode(ellipses)
            
        ellipsis_tokens = self._cached_ellipsis_tokens
        ellipsis_len = len(ellipsis_tokens)
        
        if max_tokens <= ellipsis_len:
            raise ValueError("Budget (max_tokens) is too small to fit the ellipsis.")

        content_budget = max_tokens - ellipsis_len

        if position == "right":
            final_tokens = tokens[:content_budget] + ellipsis_tokens
        elif position == "left":
            final_tokens = ellipsis_tokens + tokens[-content_budget:]
        elif position == "centre":
            front_budget = content_budget // 2
            back_budget = content_budget - front_budget
            
            front_part = tokens[:front_budget]
            back_part = tokens[-back_budget:] if back_budget > 0 else []
            
            final_tokens = front_part + ellipsis_tokens + back_part
        else:
            raise ValueError(f"Invalid position argument: {position}")

        return self.encoder.decode(final_tokens)
