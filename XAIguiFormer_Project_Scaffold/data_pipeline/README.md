# Data Preprocessing Pipeline

This preprocessing pipeline is designed to transform raw EEG data from the TD-BRAIN dataset into graph representations suitable for machine learning models, such as graph neural networks. It uses the MNE and MNE-Connectivity toolkits along with ICLabel for artifact detection.

## Overview of Steps

1. **File Loading**
   - The script searches for `.vhdr` files matching the pattern `*task-restEC_eeg.vhdr` within each subject's session folder.
   - If no EEG file is found, the subject is skipped.

2. **Channel Cleanup**
   - Known problematic non-EEG channels are dropped (`['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']`).
   - Only EEG channels are retained.
   - A standard 10-20 montage is applied (`standard_1020`), and channels with missing locations are dropped to ensure compatibility with ICLabel.

3. **Preprocessing**
   - EEG data is re-referenced to the average.
   - Bandpass filtering is applied between 1–45 Hz.
   - The signal is resampled to 250 Hz.

4. **Artifact Removal with ICA**
   - ICA is performed with 15 components.
   - ICLabel is used to automatically identify components related to eye, muscle, or heart activity.
   - These components are excluded from the signal.

5. **Demographic Information**
   - Each subject's age and gender are retrieved from `participants.tsv`.
   - If demographic data is missing, the subject is skipped.

6. **Epoching**
   - The continuous signal is segmented into fixed-length epochs of 30 seconds.
   - If no epochs are extracted, the subject is skipped.

7. **Functional Connectivity Computation**
   - For each frequency band (delta to low gamma), the following metrics are computed:
     - **Coherence (COH)**: for node feature representation.
     - **Weighted Phase Lag Index (wPLI)**: for edge feature representation.
   - Frequencies used are:
     - Delta: 2–4 Hz
     - Theta: 4–8 Hz
     - Low Alpha: 8–10 Hz
     - High Alpha: 10–12 Hz
     - Low Beta: 12–18 Hz
     - Mid Beta: 18–21 Hz
     - High Beta: 21–30 Hz
     - Low Gamma: 30–45 Hz
     - Theta/Beta Ratio: computed as the ratio between mean theta (4–8 Hz) and high beta (21–30 Hz) power from the coherence matrices. This ratio is added as a 9th band in the output and is a known biomarker of ADHD (Angelidis et al., 2016). The corresponding wPLI matrix for this band is set to zeros.

8. **Averaging and Saving**
   - All metrics are averaged across epochs.
   - The following files are saved:
     - `[subject_id]_EC_coherence.npy`
     - `[subject_id]_EC_wpli.npy`
     - `[subject_id]_EC_label.npy` containing an integer label corresponding to the subject’s condition. Subjects with missing (`NaN`) or empty `indication` values are automatically labeled as healthy (label `8`). The mapping is:
       - 0: ADHD
       - 1: MDD
       - 2: OCD
       - 3: Dyslexia
       - 4: Chronic Pain
       - 5: Burnout
       - 6: SMC
       - 7: Insomnia
       - 8: Healthy (label `"n/a"`)
       - 9: Replication
     - `[subject_id]_EC_demographics.npy`

## Output Structure

Each subject's folder contains:
- A 3D numpy array for coherence (shape: [9, channels, channels], including theta/beta ratio as last band)
- A 3D numpy array for wPLI (shape: [9, channels, channels], last band is all zeros for theta/beta ratio)
- A label file: integer-coded condition as described above
- A demographics file: `[[age, gender]]`, where gender is 1 for male, 0 for female

# Usage

To run the preprocessing on all subjects, execute the script from the command line:

```bash
python preprocess_tdbrain_to_npy.py
```
Make sure you have the following directory structure:

```bash
data/
└── TD-BRAIN-SAMPLE/
    ├── participants.tsv
    └── sub-XXXXXX/
        └── ses-1/
            └── eeg/
                └── sub-XXXXXX_ses-1_task-restEC_eeg.vhdr
```

You can modify the output directory or task name inside the __main__ block of the script:

```bash
output_dir = Path("/your/output/path")
task = "restEC"  # change if needed
```
Dependencies:
	•	mne
	•	mne-connectivity
	•	mne-icalabel
	•	numpy
	•	pandas

see `pyproject.toml`