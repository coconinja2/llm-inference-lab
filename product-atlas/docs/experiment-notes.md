# Experiment notes

## Scope

The historical Product Atlas work investigated exact lookup-based inference,
primarily for BF16 matrix-vector operations during batch-one decoding. These
notes preserve approximate recovered measurements while avoiding model files,
raw traces, local paths, and personal or employer information.

## Recovered findings

- A historical atlas occupied approximately 2.6 GB; an active product row was
  approximately 39 KB.
- A representative one-token comparison was approximately 8.71 seconds for a
  native path and 13.76 seconds for the lookup path.
- A four-column end-to-end experiment improved approximately 45.5 seconds to
  40.3 seconds while preserving matching tokens and logit hashes.
- Four-column processing improved one representative projection microkernel by
  approximately 1.27x; eight columns did not preserve the gain.
- Packed 14-bit weight IDs improved another representative projection by
  approximately 1.34x.
- One earlier formulation removed approximately 77% of counted scalar
  multiplications but remained approximately 9.5x slower overall because
  lookup and routing overhead dominated.
- A measured cache hit rate near 10% made caching-style reuse an unattractive
  primary strategy.

These are recovered exploratory measurements, not portable benchmark claims.
The original artifacts are not included, so this repository does not attempt
to reproduce those exact numbers.

## Kernel work explored

- Scalar exact-product lookup.
- Two-, four-, and eight-column blocking.
- Packed weight identifiers.
- Loop unrolling and prefetching.
- Fused lookup and accumulation.
- Exact token/logit validation in end-to-end experiments.
- Profiling address generation, lookup traffic, and accumulation separately.

No surviving evidence establishes a completed production Metal or GPU kernel.
A custom Metal/ARM kernel was identified as follow-up work.

## Next experiments

1. True 2-D lookup tiles, such as four activations by four weight columns.
2. Physical coalescing of nearby requests, not merely loop blocking.
3. Packed-parameter direct addressing into fixed 4 KiB storage blocks.
4. A minimal/perfect hash variant when the parameter space is sparse.
5. Larger precomputed partial products that replace hundreds or thousands of
   multiply-accumulates per retrieval.
6. Lossless 10-12 bit BF16 weight encodings with fused decode and GEMV.

The deciding metric remains end-to-end tokens per second against the strongest
native baseline, not multiplication count in isolation.
