# Paired bootstrap results

- Input: `results/post_annotation/episode_metrics.csv`
- Resampling unit: program (paired, n=20)
- Bootstrap: percentile 95% CI, 1,000 draws, seed 42
- The same resampling indices are used for all comparisons and metrics.
- Difference is `Integrated - other`; negative is favorable for Pk and WindowDiff.

| Comparison | Metric | Integrated | Other | Difference [95% CI] | W/T/L |
|---|---:|---:|---:|---:|---:|
| Integrated - No VLM | F1@5 | 0.678 | 0.611 | +0.067 [+0.030, +0.104] | 14/1/5 |
| Integrated - No VLM | F1@30 | 0.881 | 0.839 | +0.042 [+0.011, +0.074] | 12/2/6 |
| Integrated - No VLM | Pk | 0.074 | 0.086 | -0.012 [-0.021, -0.003] | 12/2/6 |
| Integrated - No VLM | WindowDiff | 0.130 | 0.154 | -0.024 [-0.038, -0.008] | 15/0/5 |
| Integrated - No VLM | Start F1@5 | 0.724 | 0.642 | +0.083 [+0.039, +0.130] | 15/4/1 |
| Integrated - No VLM | End F1@5 | 0.643 | 0.613 | +0.030 [-0.010, +0.068] | 11/3/6 |
| Integrated - No VLM | Start F1@30 | 0.896 | 0.848 | +0.048 [+0.017, +0.083] | 11/7/2 |
| Integrated - No VLM | End F1@30 | 0.899 | 0.872 | +0.027 [-0.006, +0.063] | 10/5/5 |
| Integrated - No transcript | F1@5 | 0.678 | 0.371 | +0.307 [+0.213, +0.395] | 19/0/1 |
| Integrated - No transcript | F1@30 | 0.881 | 0.872 | +0.009 [-0.045, +0.056] | 10/2/8 |
| Integrated - No transcript | Pk | 0.074 | 0.118 | -0.044 [-0.076, -0.010] | 17/0/3 |
| Integrated - No transcript | WindowDiff | 0.130 | 0.172 | -0.041 [-0.082, +0.002] | 14/0/6 |
| Integrated - No transcript | Start F1@5 | 0.724 | 0.380 | +0.345 [+0.235, +0.446] | 18/0/2 |
| Integrated - No transcript | End F1@5 | 0.643 | 0.309 | +0.333 [+0.248, +0.422] | 20/0/0 |
| Integrated - No transcript | Start F1@30 | 0.896 | 0.891 | +0.004 [-0.058, +0.056] | 10/3/7 |
| Integrated - No transcript | End F1@30 | 0.899 | 0.871 | +0.028 [-0.029, +0.077] | 11/2/7 |
| Integrated - No RapidOCR | F1@5 | 0.678 | 0.647 | +0.031 [-0.017, +0.074] | 10/2/8 |
| Integrated - No RapidOCR | F1@30 | 0.881 | 0.868 | +0.013 [-0.012, +0.039] | 11/3/6 |
| Integrated - No RapidOCR | Pk | 0.074 | 0.079 | -0.005 [-0.014, +0.004] | 12/0/8 |
| Integrated - No RapidOCR | WindowDiff | 0.130 | 0.135 | -0.005 [-0.024, +0.017] | 14/0/6 |
| Integrated - No RapidOCR | Start F1@5 | 0.724 | 0.674 | +0.051 [-0.004, +0.103] | 11/2/7 |
| Integrated - No RapidOCR | End F1@5 | 0.643 | 0.613 | +0.030 [-0.021, +0.078] | 9/4/7 |
| Integrated - No RapidOCR | Start F1@30 | 0.896 | 0.882 | +0.014 [-0.012, +0.045] | 6/8/6 |
| Integrated - No RapidOCR | End F1@30 | 0.899 | 0.882 | +0.017 [-0.007, +0.044] | 8/8/4 |

## Direct comparison of VLM gains at starts versus ends

A positive contrast means that removing VLM annotations harms starts more than ends.

| Tolerance | Start gain | End gain | Start-minus-end gain [95% CI] |
|---:|---:|---:|---:|
| 5s | +0.083 | +0.030 | +0.053 [+0.024, +0.081] |
| 30s | +0.048 | +0.027 | +0.021 [-0.011, +0.055] |
