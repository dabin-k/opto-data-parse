# M150605_ICTP1 data-file conjectures and paper-derived notes

_Last updated: 2026-08-06_

## Goal

Identify the most processed representation of the optogenetic extracellular recordings, ideally something close to:

```python
responses.shape == (n_trials_or_conditions, n_units_or_populations, n_times)
stimulus.shape  == (n_trials_or_conditions, n_times)
```

The immediate priority is to avoid reprocessing the `.ns5` continuous voltage data unless no higher-level representation exists.

---

## Important conclusion from the paper

The desired representation may **not** have a conventional `n_cells` dimension for V1.

The paper says that the primary V1 analysis used a **clusterless algorithm**, starting from SpikeDetekt detections and PCA features rather than conventional sorted units. Each detected spike event was classified individually as either:

- a wide spike, interpreted as putative excitatory activity; or
- a narrow spike, interpreted as putative fast-spiking inhibitory activity.

The resulting analysis therefore appears to have focused on two population-level signals rather than a matrix of identified single neurons. A realistic target may be:

```python
responses.shape == (n_trials, 2, n_times)
# population axis:
# 0 = wide-spiking / putative excitatory
# 1 = narrow-spiking / putative inhibitory
```

or a trial-averaged version:

```python
responses.shape == (n_conditions, 2, n_times)
```

For LGN, the paper did use conventional spike sorting with KiloSort and Phy, then pooled selected units into multi-unit activity. That is a different pipeline from the main cortical V1 E/I analysis.

---

## Paper-derived acquisition and processing facts

### Acquisition

The optogenetic extracellular recordings used 32-channel NeuroNexus silicon probes:

- Buzsaki32 probes; or
- A1x32-Edge-5mm-20-177 ("Edge32") probes.

Signals were:

- sampled at 30 kHz;
- amplified and acquired with a Blackrock Cerebus system; and
- stored for offline analysis.

This strongly supports the interpretation that:

- `.ns5` contains original continuous Blackrock data;
- `.nev` contains Blackrock event/digital-timestamp data;
- `.ccf` contains a Cerebus channel/configuration file.

### V1 spike processing

The primary V1 processing pipeline described in Supplemental Information S1.6 was approximately:

```text
continuous extracellular voltage
    ↓
SpikeDetekt spike-event detection
    ↓
PCA features from SpikeDetekt
    ↓
locality-sensitive hashing to find up to 100 similar spikes
    ↓
average similar waveforms to denoise each target spike
    ↓
artifact and quality-control rejection
    ↓
classify each retained event as wide or narrow
    ↓
construct E- and I-population PSTHs
```

Relevant details:

- The custom method did not require explicit clustering.
- Similar spikes were identified in SpikeDetekt PCA-feature space.
- Only about 40% of detected SpikeDetekt events were ultimately classified as wide or narrow; the rest were excluded.
- The paper states that optogenetic and movement artifacts were rejected partly by detecting waveforms that were too similar across recording sites.

### LGN spike processing

For LGN recordings, the paper says that the authors:

1. spike-sorted with KiloSort;
2. curated or inspected with Phy;
3. identified units with rhythmic responses to reversing checkerboards; and
4. pooled spikes from those selected units into LGN multi-unit activity.

Thus, KiloSort/Phy output is highly relevant for LGN but is not necessarily the source of the published V1 wide/narrow population signals.

### Published PSTH processing

For extracellular V1 recordings, the paper describes:

- PSTHs computed separately for each stimulus condition;
- smoothing with a 40 ms Hamming window;
- averaging across trials of the same stimulus condition;
- normalization by mean baseline activity from 500 to 100 ms before the first laser pulse.

Therefore, the most publication-ready representation might be:

```python
condition × population(E/I) × time
```

rather than trial-by-trial spikes.

### Stimulus design

The optogenetic experiment used:

- 445 nm blue stimulation;
- 561 nm green stimulation;
- single pulses;
- paired pulses in all four E/I orders;
- pulse durations around 0.5–4 ms;
- interpulse intervals ranging from 5 to 800 ms;
- randomized trial blocks;
- 10–68 blocks per session;
- randomized intervals of roughly 1.6–5 s between stimuli.

This implies that a useful trial table should ideally contain:

```text
trial index
first pulse type / wavelength
second pulse type / wavelength, if present
first onset time
second onset time, if present
pulse duration(s)
interpulse interval
condition identifier
block identifier
```

---

## Files observed

Top-level/session directory:

```text
M150605_ICTP1_s1_V1.mat
M150605_ICTP1_s1_not9.mat
M150605_ICTP1_s1_V1_CAR.dat
M150605_ICTP1_s1_not9.dat
M150605_ICTP1_s1_not9.kwik
M150605_ICTP1_s1_not9.kwx
M150605_ICTP1_s1_not9.raw.kwd
klustakwik/
```

Experiment/acquisition directory:

```text
1_1_M150605_ICTP1_1_1.mat
1_1_M150605_ICTP1_Timeline.mat
1_1_M150605_ICTP1_eye.mat
1_1_M150605_ICTP1_hardwareInfo.mat
M150605_ICTP1_1_1.ccf
M150605_ICTP1_1_1.nev
M150605_ICTP1_1_1.ns5
M150605_ICTP1_1_1.p
Protocol.mat
```

---

## Updated interpretation after inspecting the top-level MAT variables

### `M150605_ICTP1_s1_V1.mat`

Observed variables:

```text
inputStruct            struct, 1 × 1
ns5files2unify         struct, 1 × 13
CHANNELS_ORDER         double, 32 × 1
lims                   double, 1 × 13
SELECTED_CHANNELS      double, 32 × 1
INTERNAL_REF_CONTACT   empty
SELECTED_EXPERIMENTS   double, 1 × 13
```

### Current conjecture

**This is probably not the final response array.**

The variables look more like a preprocessing recipe, manifest, or saved set of arguments for combining 13 Blackrock recordings:

- `ns5files2unify`: likely identifies the 13 source `.ns5` files or recording segments;
- `lims`: likely records segment limits, sample boundaries, or concatenation endpoints;
- `SELECTED_EXPERIMENTS`: likely the experiment numbers included in the combined session;
- `SELECTED_CHANNELS`: the 32 channels retained for the V1 version;
- `CHANNELS_ORDER`: channel remapping/order for probe geometry or binary export;
- `INTERNAL_REF_CONTACT`: empty, suggesting no single internal reference electrode was specified.

The filename and fields suggest this file may have been used by a script that created:

```text
M150605_ICTP1_s1_V1_CAR.dat
```

from several `.ns5` segments.

**Confidence: high** that it is configuration/provenance rather than a final PSTH file.

---

### `M150605_ICTP1_s1_not9.mat`

Observed variables:

```text
inputStruct            struct, 1 × 1
ns5files2unify         struct, 1 × 13
CHANNELS_ORDER         double, 36 × 1
lims                   double, 1 × 13
SELECTED_CHANNELS      double, 36 × 1
INTERNAL_REF_CONTACT   empty
SELECTED_EXPERIMENTS   double, 1 × 13
```

### Current conjecture

This appears to be another preprocessing manifest for the same 13 source recordings, but with 36 selected channels rather than 32.

Possible interpretations of `not9` include:

1. experiment/segment 9 was excluded;
2. channel/contact 9 was excluded;
3. a named probe group or processing variant called `not9`;
4. 32 neural channels plus several auxiliary channels were retained.

The filename alone is not enough to decide which interpretation is correct.

Because the associated SpikeDetekt/Klusta files are also named `not9`, this was probably the variant used to produce:

```text
M150605_ICTP1_s1_not9.dat
M150605_ICTP1_s1_not9.raw.kwd
M150605_ICTP1_s1_not9.kwik
M150605_ICTP1_s1_not9.kwx
```

**Confidence: high** that it is preprocessing configuration; **low** on the meaning of `not9`.

---

### `1_1_M150605_ICTP1_Timeline.mat`

Observed top-level variable:

```text
Timeline   struct, 1 × 1
```

### Current conjecture

This is one of the strongest candidates for measured stimulus timing.

Likely contents include some combination of:

- laser TTL or analog control traces;
- blue- and green-laser command signals;
- visual stimulus timing;
- photodiode or synchronization signals;
- timestamps and sampling rate;
- possibly camera, wheel, or behavioral signals.

The top-level `whosmat` output is not enough because all useful information is nested inside `Timeline`.

This file may provide the most reliable stimulus onset times, while `Protocol.mat` provides condition labels and intended trial order.

**Confidence: medium-high**, based on naming and common experimental organization.

---

### `Protocol.mat`

Observed top-level variable:

```text
Protocol   struct, 1 × 1
```

### Current conjecture

This likely stores intended experimental conditions and randomization, such as:

- pulse identities;
- pulse durations;
- paired-pulse order;
- interpulse intervals;
- trial/block ordering;
- parameter values for each condition.

A useful working model is:

```text
Protocol.mat = what the experiment intended to present
Timeline.mat = when the hardware actually presented it
```

The two may need to be aligned by trial or event order.

**Confidence: medium-high**.

---

### `1_1_M150605_ICTP1_1_1.mat`

The attempted path was:

```text
/mnt/scratch/M150605_ICTP1/1/1/1_1_M150605_ICTP1_1_1.mat
```

but the directory listing suggests the file is probably located at:

```text
/mnt/scratch/M150605_ICTP1/1/1_1_M150605_ICTP1_1_1.mat
```

There appears to be one extra `/1/` in the failed path.

### Current conjecture

This could be a key session or experiment file and should be inspected before falling back to `.kwik`.

Possible contents:

- experiment metadata;
- trial/event records;
- processed online spike information;
- references to Timeline and Protocol;
- acquisition file paths;
- stimulus-response structures.

Its role cannot be determined until the corrected path is inspected recursively.

**Confidence: low-medium**, but priority is high because it may be the missing higher-level file.

---

## Interpretation of the non-MAT files

### `.ns5`

Example:

```text
M150605_ICTP1_1_1.ns5
```

Likely original Blackrock continuous voltage data sampled at 30 kHz.

**Processing level: raw acquisition.**

Avoid unless higher-level data are incomplete.

---

### `.nev`

Example:

```text
M150605_ICTP1_1_1.nev
```

Likely Blackrock event data:

- digital input transitions;
- hardware event timestamps;
- possibly comments or spike-event packets.

Useful for synchronization and validating stimulus timing, but probably not the easiest starting point if `Timeline.mat` already contains aligned signals.

**Processing level: acquisition events.**

---

### `.ccf`

Example:

```text
M150605_ICTP1_1_1.ccf
```

Likely Blackrock/Cerebus channel configuration.

Useful for:

- channel labels;
- gains;
- filters;
- enabled inputs;
- electrode mappings.

Not likely to contain responses.

---

### `*_CAR.dat`

Example:

```text
M150605_ICTP1_s1_V1_CAR.dat
```

`CAR` almost certainly means common-average referenced.

This is likely a headerless continuous binary voltage file after:

- selecting channels;
- ordering channels;
- concatenating recordings; and
- applying common-average referencing.

It is more processed than `.ns5`, but still far from the desired trial-aligned response array.

**Processing level: preprocessed continuous voltage.**

---

### `*.dat`

Example:

```text
M150605_ICTP1_s1_not9.dat
```

Likely a flat binary continuous-voltage file prepared for SpikeDetekt/Klusta.

It may be concatenated and channel-selected, but it is still continuous voltage rather than spike times or PSTHs.

**Processing level: preprocessed continuous voltage / spike-detector input.**

---

### `*.raw.kwd`

Example:

```text
M150605_ICTP1_s1_not9.raw.kwd
```

Likely HDF5/Klusta continuous data.

It may contain:

- continuous traces;
- recording boundaries;
- channel metadata;
- sample rates.

Despite the word `raw`, it may already be converted from Blackrock format, but it is not the desired final representation.

**Processing level: converted continuous voltage.**

---

### `*.kwik`

Example:

```text
M150605_ICTP1_s1_not9.kwik
```

Likely the most useful non-MAT file.

Potentially contains:

- spike-event sample indices;
- recording/segment identifiers;
- cluster assignments;
- channel-group metadata;
- event metadata;
- references to features in `.kwx`.

For this paper, the `.kwik` file may contain the SpikeDetekt detections that fed the custom clusterless classification.

However, the final wide/narrow label for every spike might be stored elsewhere, because this custom classification is not necessarily a standard Klusta cluster field.

**Processing level: detected spikes and spike metadata.**

This is probably the next fallback if no final response/PSTH arrays are found in MAT files.

---

### `*.kwx`

Example:

```text
M150605_ICTP1_s1_not9.kwx
```

Likely contains high-dimensional spike features and masks, including PCA-derived features.

This matches the paper's statement that the clusterless algorithm used PCA features produced by SpikeDetekt.

It may be necessary to reproduce the wide/narrow classifier, but it is not desirable if the objective is to avoid preprocessing.

**Processing level: spike features, intermediate.**

---

### `klustakwik/`

Likely contains KlustaKwik clustering output or temporary clustering files.

This may represent an alternate or validation sorting pipeline. The paper says the custom clusterless results were cross-checked against traditionally spike-sorted data on a subset of recordings.

It is unlikely to be the simplest path to the published V1 population responses.

---

### `*_eye.mat`

Likely eye-camera or pupil data.

Probably irrelevant for the immediate goal except for excluding artifacts or reproducing behavioral-state criteria.

---

### `*_hardwareInfo.mat`

Likely acquisition hardware metadata and channel/device information.

Useful for decoding Timeline channels and sample rates, but probably not a response array.

---

### `*.p`

Unknown. Possibilities include:

- a parameter file;
- a serialized experiment-control object;
- a probe or protocol helper file.

Inspect with `file`, `head`, or a hex viewer before assuming its format.

---

## Current ranking of useful files

### Priority 1: corrected session-level MAT file

Inspect:

```text
/mnt/scratch/M150605_ICTP1/1/1_1_M150605_ICTP1_1_1.mat
```

This may be the best chance of finding an already assembled trial/session structure.

### Priority 2: `Timeline.mat` plus `Protocol.mat`

Together, these are likely the best route to:

- measured stimulus onsets;
- trial condition labels;
- pulse durations;
- pulse order;
- interpulse interval.

They may be enough to construct the stimulus array without touching `.nev`.

### Priority 3: search for additional MAT files

Search recursively for MAT files containing names such as:

```text
psth
rate
spike
wide
narrow
clusterless
stim
trial
response
population
```

There may be publication-analysis output elsewhere in the dataset tree.

### Priority 4: `.kwik`

Use `.kwik` for detected spike times if no final wide/narrow PSTHs are stored.

### Priority 5: `.kwx`

Only use if the custom classification must be reconstructed.

### Avoid initially

```text
.ns5
.dat
.raw.kwd
```

These require substantially more raw-data processing.

---

## What “most processed” probably means in this dataset

There are several possible endpoints:

### Best case

A MAT file already stores something like:

```python
E_rate      # condition × time
I_rate      # condition × time
time_axis
stimulus_conditions
```

or:

```python
trial_rates  # trial × 2 × time
trial_info
```

### Intermediate case

A file stores:

```python
spike_times
wide_or_narrow_label
trial_onsets
trial_conditions
```

Then only binning/alignment is needed.

### Less desirable case

The `.kwik` contains detected spike times, but the custom wide/narrow labels are not saved. Reconstructing the paper's classifier would then require `.kwx` PCA features and the custom locality-sensitive-hashing code.

### Important limitation

A conventional:

```python
n_trials × n_cells × n_times
```

array may be incompatible with the paper's principal V1 analysis, because the authors deliberately avoided assigning every detected event to a stable single-neuron cluster.

A more faithful representation is likely:

```python
n_trials × 2_populations × n_times
```

---

## Immediate next inspections

### 1. Correct the missing path

```python
Path("/mnt/scratch/M150605_ICTP1/1/1_1_M150605_ICTP1_1_1.mat")
```

### 2. Recursively inspect MATLAB structs

`scipy.io.whosmat()` only shows top-level variables. The useful fields are nested inside `Timeline`, `Protocol`, and possibly `inputStruct`.

A compact recursive Python inspector:

```python
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def summarize(value, indent=0, name="root", max_depth=5):
    prefix = " " * indent
    print(f"{prefix}{name}: type={type(value).__name__}", end="")

    if isinstance(value, np.ndarray):
        print(f", shape={value.shape}, dtype={value.dtype}")
    else:
        print()

    if max_depth <= 0:
        return

    # scipy MATLAB struct object
    if hasattr(value, "_fieldnames"):
        for field in value._fieldnames:
            summarize(
                getattr(value, field),
                indent=indent + 2,
                name=field,
                max_depth=max_depth - 1,
            )
        return

    # Object arrays often contain MATLAB structs/cells.
    if isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in np.ndenumerate(value):
            summarize(
                item,
                indent=indent + 2,
                name=f"{name}{index}",
                max_depth=max_depth - 1,
            )


paths = [
    Path("/mnt/scratch/M150605_ICTP1/1/1_1_M150605_ICTP1_1_1.mat"),
    Path("/mnt/scratch/M150605_ICTP1/1/1_1_M150605_ICTP1_Timeline.mat"),
    Path("/mnt/scratch/M150605_ICTP1/1/Protocol.mat"),
    Path("/mnt/scratch/M150605_ICTP1/1/M150605_ICTP1_s1_V1.mat"),
]

for path in paths:
    print(f"\n{'=' * 20} {path} {'=' * 20}")
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"Could not load: {exc}")
        continue

    for key, value in data.items():
        if not key.startswith("__"):
            summarize(value, name=key)
```

### 3. Inspect the KWIK hierarchy without loading arrays

```python
import h5py

path = "/mnt/scratch/M150605_ICTP1/1/M150605_ICTP1_s1_not9.kwik"

with h5py.File(path, "r") as handle:
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)

    handle.visititems(visitor)
```

Useful paths may resemble:

```text
/channel_groups/.../spikes/time_samples
/channel_groups/.../spikes/recording
/channel_groups/.../spikes/clusters/...
/recordings/...
/event_types/...
```

Exact paths depend on the Klusta/Kwik version.

---

## Timing and synchronization cautions

- Spike times in `.kwik` may be stored as sample indices at 30 kHz rather than seconds.
- Recording segments may have been concatenated; `lims` and `ns5files2unify` may be needed to map concatenated sample numbers back to individual experiments.
- `Protocol.mat` may describe intended timing, whereas `Timeline.mat` or `.nev` may contain measured timing.
- Always check whether pulse onset is represented by a rising edge, falling edge, analog threshold crossing, or logged software timestamp.
- Paired-pulse trials require preserving both onsets, not only the first event.
- The paper warns that extracellular spike detection severely underestimates activity during roughly the first 20 ms after strong excitatory optogenetic stimulation because of overlapping spikes and high-frequency field-potential fluctuations. The earliest part of a trial-aligned response should therefore be interpreted cautiously.

---

## Current working hypothesis

The most likely route to a usable analysis-ready dataset is:

```text
corrected session MAT file
        +
Timeline.mat
        +
Protocol.mat
        ↓
trial table and measured laser onsets
        +
either:
    precomputed E/I responses from another MAT file
or:
    wide/narrow spike events from a processed file
        ↓
trial × 2 populations × time
```

The existing `M150605_ICTP1_s1_V1.mat` and `M150605_ICTP1_s1_not9.mat` appear to be preprocessing manifests, not final response arrays.

If no publication-level MAT output exists, `.kwik` is the best next source because it should avoid returning to continuous `.ns5` voltage. The critical unknown is where the custom per-event wide/narrow labels were saved.
