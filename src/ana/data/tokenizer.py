"""One joint SentencePiece vocabulary per corpus, shared by the source and the target side.

The model ties its embedding table to its output head, so a single vocabulary is the natural
fit. The tokenizer is trained once and reused; if the file is already there it is loaded, so
a rerun cannot quietly train a different vocabulary and change every parameter count with it.
"""

from __future__ import annotations

import os
import tempfile

import sentencepiece as spm

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class Tokenizer:
    def __init__(self, model_path: str) -> None:
        self.processor = spm.SentencePieceProcessor(model_file=model_path)

    def __len__(self) -> int:
        return self.processor.get_piece_size()

    def encode(self, text: str, max_length: int) -> list[int]:
        """Source and target are both wrapped in start and end symbols."""
        pieces = self.processor.encode(text, out_type=int)[: max_length - 2]
        return [BOS_ID, *pieces, EOS_ID]

    def encode_target(self, text: str, max_length: int) -> list[int]:
        """The decoder is fed the start symbol by the model, so labels only need the end."""
        pieces = self.processor.encode(text, out_type=int)[: max_length - 1]
        return [*pieces, EOS_ID]

    def decode(self, ids: list[int]) -> str:
        kept = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.processor.decode(kept)


def train_or_load(
    model_path: str,
    sentences: list[str],
    vocab_size: int,
    character_coverage: float,
    model_type: str = "bpe",
) -> Tokenizer:
    """`model_type` is "bpe" for the translation corpora and "word" for COGS.

    COGS is scored by exact match against a logical form whose tokens are words. The papers
    that report 96 percent in-distribution and around 80 percent on the generalization split
    all tokenise it on whitespace, and its whole vocabulary is under 900 types. Splitting those
    words into subwords would give a model a different task from the one the published numbers
    describe, and nothing to compare against.
    """
    if os.path.exists(model_path):
        return Tokenizer(model_path)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    prefix = model_path[: -len(".model")] if model_path.endswith(".model") else model_path

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for sentence in sentences:
            handle.write(sentence.replace("\n", " ") + "\n")
        corpus_file = handle.name

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=prefix,
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            # A small corpus may not contain enough distinct pieces to fill the requested
            # vocabulary. Without this, training raises rather than settling for a smaller
            # one, and every smoke run on a truncated split fails before it starts. The
            # model reads its vocabulary size back off the tokenizer, so a short table is
            # harmless; a crash is not.
            hard_vocab_limit=False,
            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            unk_id=UNK_ID,
            pad_piece="<pad>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            unk_piece="<unk>",
            minloglevel=1,
        )
    finally:
        os.unlink(corpus_file)

    return Tokenizer(model_path)
