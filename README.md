# Pareton (SN10)

**Inference optimization campaigns.** Seeded from [Cacheon](https://github.com/latent-to/cacheon); this repo is the Pareton product on Bittensor SN10.

Cacheon (SN14) code remains in-tree as legacy reference while Stage 0 is built. New work lives under `pareton/`.

## Stage 0 (in progress)

Campaign pipeline: profile → pinned campaign manifest → miner patch commitment → provenance & build gate → content-addressed engine image.

See:

- [`Pareton_Engineering_Architecture_v0.pdf`](Pareton_Engineering_Architecture_v0.pdf)
- [`Pareton_Optimization_Profile.pdf`](Pareton_Optimization_Profile.pdf)

## Legacy Cacheon layout

Frozen Cacheon (SN14) code — not the Pareton runtime path:

- `validator/`, `cacheon_db/`, `api/`, `miner/commit.py`, `example-miner/`, `scripts/`

## License

MIT
