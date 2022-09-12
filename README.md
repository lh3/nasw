## Getting Started
```sh
git clone https://github.com/lh3/nasw
cd nasw && make
./nasw test/test-n.fa test/test-p.fa
./nasw test/DPP3-hs.gen.fa.gz test/DPP3-mm.pep.fa.gz
```

## Introduction

nasw provides a implementation of dynamic programming (DP)
for protein-to-genome alignment with affine-gap penalty, splicing and
frameshifts. The DP involves 6 states and 20 transitions, similar to the
[GeneWise][genewise] model. Different from GeneWise, nasw explicitly implements
the DP recursion with SSE2 or NEON intrinsics and is tens of times faster.
Please see [nasw.h](nasw.h) for the brief API documentation.

## Limitations

1. The initial condition disallows gap opens at the beginning of the sequences.
2. Simplistic splice site model
3. The 32-bit mode uses excessive memory for traceback

[genewise]: https://pubmed.ncbi.nlm.nih.gov/15123596/
