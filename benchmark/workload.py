"""Builds prompts with an exact, controlled token count.

Prompt *length* is one of the axes we sweep (it drives prefill cost), so we
need precise control over it rather than an approximation like word count.
We tokenize a long source passage with the model's real tokenizer, tile it
if needed, then slice to the exact token count and decode back to text.
Decoding a sliced token sequence and re-tokenizing it server-side can drift
by a token or two at BPE merge boundaries; this is accurate to within ~1%,
which is fine for a controlled sweep axis.
"""

from transformers import AutoTokenizer

_SOURCE_TEXT = """
The history of computing is a story of abstraction. Early machines were
programmed by rewiring physical circuits, one instruction encoded directly
in the arrangement of switches and cables. The invention of stored-program
architecture changed this: instructions became data, sitting in the same
memory as the numbers they operated on, and could be written, copied, and
modified with the same tools used for any other computation. This single
idea underlies nearly everything that followed, from compilers that
translate human-readable code into machine instructions, to operating
systems that multiplex a single processor across many programs, to virtual
machines that simulate entire computers inside other computers. Each layer
of abstraction traded some performance for a large gain in expressiveness
and reliability, and the net effect over decades has been an extraordinary
compounding of capability built on layers most programmers never think
about directly.
""".strip()


def build_prompt(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    ids = tokenizer.encode(_SOURCE_TEXT, add_special_tokens=False)
    while len(ids) < target_tokens:
        ids += tokenizer.encode(_SOURCE_TEXT, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def load_tokenizer(model: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model)
