# Product Atlas

Product Atlas is a memory-centric inference experiment: precompute exact
products for bounded BF16 value domains, retrieve them by compact IDs, and
measure whether avoided multiplication can outweigh indexing and memory costs.

The project is intentionally a negative-results-first engineering study. Its
main lesson is that removing arithmetic does not guarantee a faster kernel:
address generation, irregular reads, cache behavior, and accumulation can cost
more than the multiplication they replace.

## What is included

- A dependency-free C++17 Product Atlas implementation.
- Exact FP32 products of BF16 operands stored in a dense atlas.
- Scalar and four-way blocked lookup dot-product kernels.
- A direct-address prototype that packs eight bounded parameters into a
  64-bit key, maps high bits to a 4 KiB block, and maps six low bits to one of
  sixty-four 64-byte slots.
- Exactness tests and a reproducible synthetic benchmark.
- Recovered experimental findings and the next research questions.

This repository is a privacy-safe reconstruction. Original large model files,
raw traces, chat attachments, machine-specific paths, and personal information
are deliberately excluded.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/product_atlas_bench 4096 2000
```

Benchmark numbers vary by compiler and hardware. The benchmark verifies exact
equality before timing and reports lookup/native ratios rather than presenting
one machine's output as universal.

## Architecture

```text
activation ID + weight ID
          |
          v
  dense product address
          |
          v
 exact precomputed FP32 product
          |
          v
 ordered FP32 accumulation
```

The direct-address experiment explores a storage-scale variant:

```text
8 bounded parameters -> packed key -> block + slot -> fixed-size result
```

With 64-byte records in a 4 KiB page:

```text
slot         = key & 63
block_number = key >> 6
byte_offset  = block_number << 12
```

The address calculation becomes the index. Whether it wins depends on how much
useful computation each storage fetch replaces and how well requests coalesce.

## Key lessons

1. Memory traffic and routing can dominate arithmetic.
2. Cache-dependent reuse is fragile when the observed hit rate is low.
3. Contiguous blocked retrieval is more promising than scalar random lookup.
4. Microkernel improvements must be validated end to end.
5. One retrieval should replace a large partial computation, not one multiply.
6. For batch-one decoding, lossless weight compression plus fused decode/GEMV
   may be more promising than a scalar multiplication table.

See [docs/experiment-notes.md](docs/experiment-notes.md) for recovered results,
limitations, and follow-up experiments.

## Privacy

Run the lightweight repository audit before publishing changes:

```bash
./scripts/privacy-audit.sh
```

## License

MIT
