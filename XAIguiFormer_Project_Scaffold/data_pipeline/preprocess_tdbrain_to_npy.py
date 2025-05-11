import os
import mne
import numpy as np
import json
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.io import read_raw_brainvision
from pathlib import Path

def preprocess_subject(sub_path, output_dir, subject_id, session="ses-1", task="restEC"):
    raw_dir = sub_path / session / "eeg"
    raw_files = list(raw_dir.glob(f"*task-{task}_eeg.vhdr"))
    if not raw_files:
        print(f"⚠️ No EEG file found for {subject_id} {task}")
        return

    raw_file = raw_files[0]
    print(f"▶️ Loading: {raw_file.name}")
    raw = read_raw_brainvision(raw_file, preload=True)
    # Supprimer manuellement les canaux non EEG connus qui posent problème
    bad_channels = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']
    raw.drop_channels([ch for ch in bad_channels if ch in raw.ch_names])
    # Garder uniquement les canaux EEG
    raw.pick('eeg')
    # Appliquer le montage standard aux canaux EEG
    raw.set_montage("standard_1020", on_missing="ignore")
    # Supprimer les canaux sans position (nécessaire pour ICLabel)
    raw = raw.copy().pick_channels([ch for ch in raw.info['ch_names']
                                    if raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3].any()])
    raw.set_eeg_reference("average", projection=False)

    # 1. Bandpass filter
    raw.filter(1., 45., fir_design='firwin', skip_by_annotation='edge')

    # 2. Resample
    raw.resample(250)

    # 3. ICA + ICLabel
    ica = ICA(n_components=15, random_state=97, max_iter='auto')
    ica.fit(raw)
    labels = label_components(raw, ica, method='iclabel')
    print(f"🔍 Available keys in ICLabel output: {labels.keys()}")
    y_pred = labels.get('y_pred_auto', labels.get('labels', []))
    ica.exclude = [i for i, comp in enumerate(y_pred) if comp in ['eye', 'muscle', 'heart']]
    raw = ica.apply(raw)

    # 4. Get demographic info
    age, gender = get_demographics(subject_id)
    if age is None or gender is None:
        print(f"❌ Demographics missing for {subject_id}")
        return

    # 5. Segment into 30s chunks
    epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)
    if len(epochs) == 0:
        print(f"⚠️ No 30s epochs extracted from {subject_id}")
        return

    # 6. Compute coherence & wPLI with MNE-Connectivity
    from mne_connectivity import spectral_connectivity_epochs

    con_methods = ["coh", "wpli"]
    fmin = [2, 4, 8, 10, 12, 18, 21, 30]
    fmax = [4, 8, 10, 12, 18, 21, 30, 45]
    bands = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'mid_beta', 'high_beta', 'low_gamma']
    save_dir = output_dir / subject_id
    save_dir.mkdir(exist_ok=True, parents=True)

    coherence = []
    wpli = []
    for band, lo, hi in zip(bands, fmin, fmax):
        conn = spectral_connectivity_epochs(
            epochs, method=con_methods, mode='fourier',
            fmin=lo, fmax=hi, sfreq=raw.info['sfreq'],
            faverage=True, verbose=False
        )
        coherence.append(conn[0].get_data(output='dense'))  # cohérence
        wpli.append(conn[1].get_data(output='dense'))       # wPLI

    # 7. Average over epochs
    coherence = np.stack(coherence).mean(axis=1)
    wpli = np.stack(wpli).mean(axis=1)

    # 8. Save as .npy
    base = f"{subject_id}_EC"
    np.save(save_dir / f"{base}_coherence.npy", coherence)
    np.save(save_dir / f"{base}_wpli.npy", wpli)
    np.save(save_dir / f"{base}_label.npy", np.array([1]))  # dummy label for now
    np.save(save_dir / f"{base}_demographics.npy", np.array([[age, gender]]))

    print(f"✅ Saved processed data for {subject_id}")

def get_demographics(subject_id):
    demo_path = Path("/XAIguiFormer_Project_Scaffold/data/TD-BRAIN-SAMPLE/participants.tsv")
    if not demo_path.exists():
        return None, None
    import pandas as pd
    df = pd.read_csv(demo_path, sep='\t')

    print(f"🔍 subject_id recherché = '{subject_id}'")
    print("📋 IDs dispo :", df['participant_id'].head(10).tolist())

    row = df[df['participant_id'] == subject_id]
    if row.empty and not subject_id.startswith("sub-"):
        row = df[df['participant_id'] == f"sub-{subject_id.strip()}"]

    if row.empty:
        print(f"⚠️ Aucun match trouvé pour {subject_id}")
        return None, None

    age = row.iloc[0]['age']
    gender = row.iloc[0]['gender']
    return age, gender

if __name__ == "__main__":
    data_root = Path("XAIguiFormer_Project_Scaffold/data/TD-BRAIN-SAMPLE")
    output_dir = Path("XAIguiFormer_Project_Scaffold/data/TDBRAIN-PRE-PROCESSED/raw/train")  # adapte pour val/test
    subjects = [p.name for p in data_root.iterdir() if p.name.startswith("sub-")]

    for subject_id in subjects:
        preprocess_subject(data_root / subject_id, output_dir, subject_id)