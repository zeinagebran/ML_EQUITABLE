import os
import mne
import numpy as np
import json
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.io import read_raw_brainvision
from pathlib import Path

def restore_full_matrix(vec, n_channels):
    if vec.shape[0] == n_channels and vec.shape[1] == n_channels:
        return vec  # matrice déjà carrée
    elif vec.shape[0] == (n_channels * (n_channels - 1)) // 2:
        mat = np.zeros((n_channels, n_channels))
        upper_indices = np.triu_indices(n_channels, k=1)
        mat[upper_indices] = vec[:, 0]
        mat += mat.T  # rendre symétrique
        return mat
    else:
        raise ValueError(f"Format inattendu pour la matrice : shape={vec.shape}")

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
    fmin = [4, 8, 13, 30, 4, 10, 2, 18, 13]
    fmax = [8, 10, 18, 45, 8, 12, 4, 21, 30]
    bands = ['theta', 'low_alpha', 'low_beta', 'gamma', 'theta_duplicate', 'high_alpha', 'delta', 'mid_beta', 'beta_total']
    save_dir = output_dir / subject_id
    save_dir.mkdir(exist_ok=True, parents=True)

    n_channels = raw.info['nchan']  # nombre de canaux EEG après nettoyage

    coherence = []
    wpli = []
    for band, lo, hi in zip(bands, fmin, fmax):
        conn = spectral_connectivity_epochs(
            epochs, method=con_methods, mode='fourier',
            fmin=lo, fmax=hi, sfreq=raw.info['sfreq'],
            faverage=True, verbose=False
        )

        coh_matrix = restore_full_matrix(conn[0].get_data(output='dense').squeeze(), n_channels)
        wpli_matrix = restore_full_matrix(conn[1].get_data(output='dense').squeeze(), n_channels)

        coherence.append(coh_matrix)
        wpli.append(wpli_matrix)

    # 7. Stack over bands
    coherence = np.stack(coherence)  # shape: (n_bands, n_channels, n_channels)
    wpli = np.stack(wpli)
    base = f"{subject_id}_EC"

    # 7bis. Compute theta/beta ratio from coherence
    def band_power(mat):
        return np.mean(mat)

    theta_idx = bands.index('theta')
    beta_idx = bands.index('beta_total')

    theta_power = band_power(coherence[theta_idx])
    beta_power = band_power(coherence[beta_idx])
    theta_beta_ratio = theta_power / beta_power if beta_power != 0 else 0

    np.save(save_dir / f"{base}_theta_beta_ratio.npy", np.array([theta_beta_ratio]))

    # 8. Save as .npy
    np.save(save_dir / f"{base}_coherence.npy", coherence)
    np.save(save_dir / f"{base}_wpli.npy", wpli)
    label = get_label(subject_id)
    if label is None:
        print(f"❌ Label manquant ou inconnu pour {subject_id}")
        return
    np.save(save_dir / f"{base}_label.npy", np.array([label]))
    np.save(save_dir / f"{base}_demographics.npy", np.array([[age, gender]]))

    print(f"✅ Saved processed data for {subject_id}")

def get_demographics(subject_id):
    demo_path = Path("/Users/ghalia/Desktop/Telecom_IA/projet_XAI/data/TD-BRAIN-SAMPLE/participants.tsv")
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

def get_label(subject_id):
    demo_path = Path("/Users/ghalia/Desktop/Telecom_IA/projet_XAI/data/TD-BRAIN-SAMPLE/participants.tsv")
    if not demo_path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(demo_path, sep='\t')

    row = df[df['participant_id'] == subject_id]
    if row.empty and not subject_id.startswith("sub-"):
        row = df[df['participant_id'] == f"sub-{subject_id.strip()}"]

    if row.empty:
        print(f"⚠️ Aucun label trouvé pour {subject_id}")
        return None

    label_text = str(row.iloc[0]['indication']).strip().lower()
    label_map = {
        "adhd": 0,
        "mdd": 1,
        "ocd": 2,
        "dyslexia": 3,
        "chronic pain": 4,
        "burnout": 5,
        "smc": 6,
        "insomnia": 7,
        "n/a": 8,
        "replication":9
    }

    label_encoded = label_map.get(label_text)
    if label_encoded is None:
        print(f"⚠️ Label inconnu : {label_text}")
    return label_encoded

if __name__ == "__main__":
    data_root = Path("/Users/ghalia/Desktop/Telecom_IA/projet_XAI/data/TD-BRAIN-SAMPLE")
    output_dir = Path("/Users/ghalia/Desktop/Telecom_IA/projet_XAI/data/TDBRAIN-PRE-PROCESSED/raw/train")  # adapte pour val/test
    subjects = [p.name for p in data_root.iterdir() if p.name.startswith("sub-")]

    for subject_id in subjects:
        preprocess_subject(data_root / subject_id, output_dir, subject_id)