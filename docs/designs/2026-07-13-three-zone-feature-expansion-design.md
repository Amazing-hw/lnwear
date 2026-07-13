# Three-Zone Feature Expansion Design

## Goal

Expand the governed Stage2 feature pool with complementary pulse-shape and motion-artifact evidence while preserving the hardware contract that both green-light layouts are normalized to three rotationally symmetric optical zones.

## Geometry contract

- `get_channels_from_window` remains the only raw-layout adapter.
- Downstream feature extraction consumes three zone signals named `g1`, `g2`, and `g3`; these names do not encode an absolute watch orientation.
- New spatial features must be invariant under every permutation of the three zones.
- No feature may use a raw channel identifier, clockwise direction, signed zone difference, or dominant-zone index.
- The robust top-two composite remains the optical input for ACC–PPG coupling.

## New governed candidates

1. `GTOP2_ROBUST_SKEWNESS`: quantile skewness `(P90 + P10 - 2*P50)/(P90-P10)` of the top-two pulse.
2. `GTOP2_SPECTRAL_ENTROPY`: normalized Shannon entropy of 0.5–5 Hz spectral power.
3. `ACC_JERK_TAIL_MEAN_REL`: mean of the largest 10% absolute acceleration-magnitude differences divided by mean magnitude.
4. `ACC_GREEN_MAX_LAG_CORR`: maximum absolute normalized ACC-motion/top-two-pulse correlation over ±0.4 seconds.
5. `ACC_GREEN_PSD_SIMILARITY`: cosine similarity of ACC-motion and top-two-pulse spectral power over 0.5–5 Hz.
6. `G_2OF3_PERIODICITY`: median of the three zone autocorrelation peaks, a smooth two-of-three periodic support score.
7. `G_ZONE_LAG_RMS_SEC`: RMS of absolute bounded cross-correlation peak lags across all three zone pairs.

All candidates are finite on flat, missing-ACC, short, and non-finite inputs. The pool version becomes `stage2_interpretable_v5`, with 91 ordered candidates.

## Verification

- Prove both raw layouts normalize to equivalent three-zone signals.
- Prove new spatial features are invariant under all zone permutations.
- Verify expected response on clean periodic, broadband/noisy, impulsive-motion, and lagged-motion synthetic windows.
- Verify catalog order, formulas, C metadata, CSV manual selection, deployment replay, figures, and complete regression suite.
