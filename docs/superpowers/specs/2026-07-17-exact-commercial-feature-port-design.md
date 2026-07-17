# Exact Commercial Feature Port Design

## Goal

Replace the calculation of the eight existing commercial Stage2 candidates with
the supplied float32 Python port of `hr_liveness_detect.c` (`PC_TEST` path).
Keep the existing candidate names and mapping, while making training and the
standalone deployment extractor produce the same C-compatible values.

This is an intentional breaking semantic change to the governed feature pool.
Existing Stage2 feature artifacts, rankings, trained models, and deployment
packages are stale after this change and must be regenerated.

## Scope

The change covers:

- the exact eight-feature PPG and `g_acc` calculation;
- construction of the commercial inputs from raw 25 Hz or raw 100 Hz windows;
- replacement of the eight existing Stage2 candidate values;
- governed catalog formulas, units, operators, tolerances, and feature-pool
  version;
- embedding the same implementation in the standalone deployment extractor;
- tests and documentation that describe the commercial candidates.

The change does not:

- add new feature candidates;
- rename the existing candidates;
- change the non-commercial Stage2 feature calculations;
- restore the removed commercial comparison pipeline;
- retrain a model or change the selected-feature CSV as part of implementation.

## Existing Fields To Replace

`COMMERCIAL_8_FEATURE_MAPPING` keeps its current external names and target
fields. The supplied port output replaces the value of each target field in
this exact order:

| Port output | Existing Stage2 field | New governed meaning |
| --- | --- | --- |
| `green_corr` | `GREEN_CORR` | beat-pattern correlation from filtered green |
| `green_ac` | `COMM_GREEN_AC` | median paired peak-to-valley green amplitude |
| `amb_ac` | `COMM_AMB_AC` | median paired peak-to-valley ambient amplitude |
| `acc_ysum` | `ACC_MAG_MEAN` | C-compatible `g_acc` rolling acceleration energy |
| `green_dc` | `GREEN_DC_MEDIAN` | absolute arithmetic mean of repaired raw green |
| `amb_dc` | `AMBX_DC_MEDIAN` | absolute arithmetic mean of repaired raw ambient |
| `green_xcorr` | `GREEN_AUTO_CORR_PEAK` | beat-pattern correlation after 125-point xcorr |
| `fft_peak_med` | `GREEN_FFT_PEAK_MEDIAN_RATIO` | FFT peak-to-median ratio after xcorr |

The legacy formulas formerly associated with these eight target fields are not
retained under other names. Other candidates that use their own separately
computed intermediates remain unchanged.

## Commercial Input Contract

The commercial algorithm always receives a five-second, 25 Hz window:

- PPG shape: `(125, 4)`, integer semantics;
- ACC shape: `(125, 3)`, raw sensor-count semantics;
- PPG slot 0: the commercial green signal;
- PPG slot 3: ambient;
- PPG slots 1 and 2: unused and initialized deterministically to zero.

The commercial green signal is constructed as follows:

1. Resolve `g1`, `g2`, and `g3` with the existing `ppg_config` channel mapping.
2. Compute `(g1 + g2 + g3) / 3` for each sample.
3. Convert to `int32` by truncation toward zero, matching a C integer
   assignment. Do not round to nearest.

Ambient is converted to `int32`. ACC remains in raw count units and is passed
to `compute_g_acc`, which performs the supplied `/4096.0` conversion internally.
No earlier ACC scaling is allowed.

### Native 25 Hz Input

Use the raw PPG and raw ACC samples directly. Both must contain exactly 125
aligned samples for the window.

### Native 100 Hz Input

Build the commercial input before the existing `resample_poly` path:

- commercial PPG: `raw_ppg_window[::4]`;
- commercial ACC: `raw_acc_window[::4]`.

Each raw window contains 500 aligned samples, so both slices contain exactly
125 samples. The commercial path must not interpolate, filter, average, or
round these sampled values before the port receives them.

The existing resampled 25 Hz path remains in place for all non-commercial
Stage2 candidates.

### Alignment And Failure Behavior

Commercial PPG and ACC data are required. The extractor raises a clear
`ValueError` instead of silently filling a commercial feature when:

- PPG or ACC is missing;
- a native 25 Hz window is not exactly 125 samples;
- a native 100 Hz window is not exactly 500 samples before decimation;
- PPG and ACC windows are not aligned;
- PPG channel layout does not satisfy `ppg_config`;
- the constructed commercial PPG or ACC shape is invalid.

The Stage2 diagnostics may still describe other invalid values, but they do not
turn a commercial input-contract violation into eight fallback values.

## Implementation Boundary

Create a focused commercial-port module containing the supplied constants and
functions, including `cal_ppg_feature`, `compute_g_acc`, and their private
helpers. Preserve the supplied float32 casts, operation order, padding,
filtering, peak/valley logic, FFT implementation, and feature order. Avoid
rewriting the math in terms of higher-level SciPy or NumPy operations whose
rounding or edge semantics differ from the reference.

Expose a small adapter that accepts the resolved raw `ambient`, `g1`, `g2`,
`g3`, raw ACC, and source frequency, validates the contract, constructs the
four-column PPG buffer, and returns a name-to-float mapping for the eight
existing Stage2 fields.

Training extraction calls this adapter while assembling the governed candidate
dictionary. It overwrites the eight existing fields after the ordinary optical
and ACC candidate families have been computed. This keeps non-commercial
feature logic unchanged while making the commercial mapping authoritative.

The raw-window extraction path must carry a dedicated commercial input alongside
the current resampled Stage2 input. Passing only the current float64 25 Hz arrays
is insufficient for 100 Hz sources because the required original integer
samples have already been replaced by `resample_poly` output.

## Standalone Deployment

The deployment extractor remains self-contained. Its generated source embeds
the exact commercial-port implementation from the same checked-in module used
by training; it must not maintain a manually copied second implementation.

The public raw deployment entry point follows the same frequency rules:

- `frequency=25`: validate and use the 125 raw samples;
- `frequency=100`: select the 500-sample raw PPG and ACC buffers with `::4` for
  the commercial path, while retaining the existing resampling path for other
  candidates.

The lower-level deployment API that accepts already separated 25 Hz signals
requires raw-count ACC and exactly 125 samples when any governed extraction is
performed, because all eight commercial fields remain part of the governed
candidate dictionary even if the trained model selects only a subset.

Golden-vector export must generate raw integer-count ACC data, not synthetic
values in units of `g`.

## Catalog And Version Migration

Increment `FEATURE_POOL_VERSION` from `stage2_interpretable_v8` to
`stage2_interpretable_v9`. Candidate names and order remain unchanged, so the
candidate count remains 126, but the version bump is mandatory because eight
values have new semantics.

Update the eight catalog records so their preprocessing, formula, C operators,
units, accumulator type, numerical guard, buffer size, and tolerances describe
the supplied port. In particular:

- all eight use the commercial C-compatible preprocessing path;
- float32 accumulation is declared;
- `ACC_MAG_MEAN` is documented as `g_acc`, not mean acceleration magnitude;
- the DC fields are arithmetic absolute means after burr/step repair, not
  medians;
- the correlation, xcorr, and FFT fields describe the port algorithms rather
  than the previous generic implementations;
- tolerances reflect the validated C-reference agreement rather than the
  previous float64 generic-feature tolerance.

Update the feature interpretability guide and any feature-pool overview that
states the old formulas or version. Old artifacts are rejected through the
existing feature-pool version checks and must be regenerated from Stage 3
onward.

## Verification

Testing follows red-green-refactor and includes:

1. A fixed raw-integer PPG/ACC golden vector covering all eight outputs, output
   order, dtype, and numeric tolerance.
2. Focused adapter tests proving:
   - native 100 Hz uses exactly indices `0, 4, ..., 496`;
   - native 25 Hz uses the input samples unchanged;
   - three-zone green averaging truncates toward zero to `int32`;
   - ambient remains raw `int32` counts;
   - ACC remains raw counts until the port divides by 4096.
3. Failure tests for missing data, wrong lengths, misalignment, and invalid
   channel layout.
4. Stage2 integration tests showing the eight existing fields equal the port
   outputs and that representative non-commercial fields are unchanged.
5. Catalog tests for the unchanged mapping/order/count, new formulas, and new
   feature-pool version.
6. Training-versus-standalone deployment parity tests for both native 25 Hz and
   native 100 Hz raw inputs.
7. Export smoke, compilation, feature-contract, and end-to-end tests affected by
   the stricter five-second raw-count input contract.

Tests that currently construct ACC in `g` units or use three-second deployment
windows must be updated to the governed five-second, raw-count contract when
they exercise the full Stage2 candidate extractor.

## Acceptance Criteria

- The eight mapped Stage2 fields are calculated only by the supplied port.
- The implementation preserves float32 C semantics and the stated feature
  order.
- A 100 Hz commercial input uses raw `::4` samples for both PPG and ACC.
- Green is the truncated integer mean of the three resolved physical zones.
- ACC is raw count data and is divided by 4096 only inside `compute_g_acc`.
- Invalid or missing commercial input fails explicitly.
- Non-commercial candidate calculations retain their current behavior.
- Training and standalone deployment outputs match for all eight fields.
- Candidate names, mapping, order, and total count remain unchanged.
- The feature-pool version and all relevant documentation identify the semantic
  migration.
