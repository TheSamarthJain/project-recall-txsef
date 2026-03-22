import librosa
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, entropy
import warnings

warnings.filterwarnings('ignore')


def extract_advanced_features(audio_path):
    """Extract 200+ advanced acoustic features for Alzheimer's detection"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        features = {}

        # Helper function to ensure scalar values
        def to_float(val):
            if isinstance(val, (list, np.ndarray)):
                return float(np.mean(val)) if len(val) > 0 else 0.0
            return float(val)

        # ============ 1. MFCC FEATURES (78 features) ============
        # Standard MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = to_float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = to_float(np.std(mfccs[i]))
            features[f'mfcc_{i}_max'] = to_float(np.max(mfccs[i]))
            features[f'mfcc_{i}_min'] = to_float(np.min(mfccs[i]))
            features[f'mfcc_{i}_median'] = to_float(np.median(mfccs[i]))
            features[f'mfcc_{i}_skew'] = to_float(skew(mfccs[i]))

        # MFCC Deltas (first derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f'mfcc_delta_{i}_mean'] = to_float(np.mean(mfcc_delta[i]))
            features[f'mfcc_delta_{i}_std'] = to_float(np.std(mfcc_delta[i]))

        # MFCC Delta-Deltas (second derivative - acceleration)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        for i in range(13):
            features[f'mfcc_delta2_{i}_mean'] = to_float(np.mean(mfcc_delta2[i]))

        # ============ 2. PITCH FEATURES (15 features) ============
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 10:
            features['pitch_mean'] = to_float(np.mean(pitch_values))
            features['pitch_std'] = to_float(np.std(pitch_values))
            features['pitch_median'] = to_float(np.median(pitch_values))
            features['pitch_min'] = to_float(np.min(pitch_values))
            features['pitch_max'] = to_float(np.max(pitch_values))
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_iqr'] = to_float(np.percentile(pitch_values, 75) - np.percentile(pitch_values, 25))
            features['pitch_cv'] = features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0
            pitch_diffs = np.abs(np.diff(pitch_values))
            features['pitch_jitter'] = to_float(
                np.mean(pitch_diffs) / features['pitch_mean'] if features['pitch_mean'] > 0 else 0)
            features['pitch_jitter_std'] = to_float(np.std(pitch_diffs))
            features['pitch_skew'] = to_float(skew(pitch_values))
            features['pitch_kurtosis'] = to_float(kurtosis(pitch_values))
            # Pitch contour slope
            features['pitch_slope'] = to_float(np.polyfit(range(len(pitch_values)), pitch_values, 1)[0])
            features['pitch_entropy'] = to_float(entropy(np.histogram(pitch_values, bins=20)[0] + 1e-10))
            # Pitch variability over time
            features['pitch_variation_rate'] = to_float(np.sum(np.abs(np.diff(pitch_values)) > 10) / len(pitch_values))
        else:
            for key in ['pitch_mean', 'pitch_std', 'pitch_median', 'pitch_min',
                        'pitch_max', 'pitch_range', 'pitch_iqr', 'pitch_cv', 'pitch_jitter',
                        'pitch_jitter_std', 'pitch_skew', 'pitch_kurtosis', 'pitch_slope',
                        'pitch_entropy', 'pitch_variation_rate']:
                features[key] = 0.0

        # ============ 3. ENERGY/LOUDNESS FEATURES (12 features) ============
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = to_float(np.mean(rms))
        features['energy_std'] = to_float(np.std(rms))
        features['energy_max'] = to_float(np.max(rms))
        features['energy_min'] = to_float(np.min(rms))
        features['energy_range'] = features['energy_max'] - features['energy_min']
        features['energy_median'] = to_float(np.median(rms))
        energy_diffs = np.abs(np.diff(rms))
        features['energy_shimmer'] = to_float(
            np.mean(energy_diffs) / features['energy_mean'] if features['energy_mean'] > 0 else 0)
        features['energy_shimmer_std'] = to_float(np.std(energy_diffs))
        features['energy_skew'] = to_float(skew(rms))
        features['energy_kurtosis'] = to_float(kurtosis(rms))
        features['energy_entropy'] = to_float(entropy(np.histogram(rms, bins=20)[0] + 1e-10))
        features['energy_dynamic_range'] = to_float(
            20 * np.log10(features['energy_max'] / (features['energy_min'] + 1e-10)))

        # ============ 4. PAUSE & RHYTHM FEATURES (20 features) ============
        frame_length = 2048
        hop_length = 512
        energy_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        silence_threshold = np.mean(energy_frames) * 0.15
        is_silence = energy_frames < silence_threshold
        is_voiced = ~is_silence

        features['silence_ratio'] = to_float(np.sum(is_silence) / len(is_silence))
        features['voiced_ratio'] = to_float(np.sum(is_voiced) / len(is_voiced))

        # Extract pauses and voiced segments
        pauses = []
        voiced_segments = []
        current_pause = 0
        current_voiced = 0

        for i, silent in enumerate(is_silence):
            if silent:
                if current_voiced > 0:
                    voiced_segments.append(current_voiced)
                    current_voiced = 0
                current_pause += 1
            else:
                if current_pause > 0:
                    pauses.append(current_pause)
                    current_pause = 0
                current_voiced += 1

        if len(pauses) > 0:
            pause_durations = [p * (hop_length / sr) for p in pauses]
            features['pause_count'] = to_float(len(pauses))
            features['pause_rate'] = to_float(len(pauses) / (len(y) / sr))
            features['pause_duration_mean'] = to_float(np.mean(pause_durations))
            features['pause_duration_std'] = to_float(np.std(pause_durations))
            features['pause_duration_max'] = to_float(np.max(pause_durations))
            features['pause_duration_min'] = to_float(np.min(pause_durations))
            features['pause_duration_median'] = to_float(np.median(pause_durations))
            features['pause_duration_total'] = to_float(np.sum(pause_durations))
            long_pauses = [p for p in pause_durations if p > 0.5]
            features['long_pause_ratio'] = to_float(len(long_pauses) / len(pauses))
            features['long_pause_count'] = to_float(len(long_pauses))
            features['pause_duration_skew'] = to_float(skew(pause_durations))
            features['pause_duration_kurtosis'] = to_float(kurtosis(pause_durations))
            features['pause_variability'] = to_float(
                np.std(pause_durations) / np.mean(pause_durations) if np.mean(pause_durations) > 0 else 0)
        else:
            for key in ['pause_count', 'pause_rate', 'pause_duration_mean', 'pause_duration_std',
                        'pause_duration_max', 'pause_duration_min', 'pause_duration_median',
                        'pause_duration_total', 'long_pause_ratio', 'long_pause_count',
                        'pause_duration_skew', 'pause_duration_kurtosis', 'pause_variability']:
                features[key] = 0.0

        if len(voiced_segments) > 0:
            voiced_durations = [v * (hop_length / sr) for v in voiced_segments]
            features['voiced_segment_count'] = to_float(len(voiced_segments))
            features['voiced_segment_mean'] = to_float(np.mean(voiced_durations))
            features['voiced_segment_std'] = to_float(np.std(voiced_durations))
            features['voiced_segment_max'] = to_float(np.max(voiced_durations))
            features['voiced_segment_min'] = to_float(np.min(voiced_durations))
            features['voiced_segment_rate'] = to_float(len(voiced_segments) / (len(y) / sr))
            features['voiced_segment_cv'] = features['voiced_segment_std'] / features['voiced_segment_mean'] if \
            features['voiced_segment_mean'] > 0 else 0
        else:
            for key in ['voiced_segment_count', 'voiced_segment_mean', 'voiced_segment_std',
                        'voiced_segment_max', 'voiced_segment_min', 'voiced_segment_rate', 'voiced_segment_cv']:
                features[key] = 0.0

        # Speech rate estimate
        peaks, _ = find_peaks(energy_frames, distance=int(0.1 * sr / hop_length))
        features['speech_rate_estimate'] = to_float(len(peaks) / (len(y) / sr))

        # ============ 5. SPECTRAL FEATURES (35 features) ============
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = to_float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = to_float(np.std(spectral_centroids))
        features['spectral_centroid_median'] = to_float(np.median(spectral_centroids))
        features['spectral_centroid_skew'] = to_float(skew(spectral_centroids))
        features['spectral_centroid_kurtosis'] = to_float(kurtosis(spectral_centroids))

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = to_float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = to_float(np.std(spectral_rolloff))
        features['spectral_rolloff_median'] = to_float(np.median(spectral_rolloff))

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = to_float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = to_float(np.std(spectral_bandwidth))
        features['spectral_bandwidth_median'] = to_float(np.median(spectral_bandwidth))

        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = to_float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = to_float(np.std(spectral_flatness))
        features['spectral_flatness_median'] = to_float(np.median(spectral_flatness))

        # Spectral contrast (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        for i in range(min(7, spectral_contrast.shape[0])):
            features[f'spectral_contrast_{i}_mean'] = to_float(np.mean(spectral_contrast[i]))
            features[f'spectral_contrast_{i}_std'] = to_float(np.std(spectral_contrast[i]))

        # Spectral flux
        spec = np.abs(librosa.stft(y))
        spectral_flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
        features['spectral_flux_mean'] = to_float(np.mean(spectral_flux))
        features['spectral_flux_std'] = to_float(np.std(spectral_flux))

        # ============ 6. ZERO CROSSING RATE (6 features) ============
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = to_float(np.mean(zcr))
        features['zcr_std'] = to_float(np.std(zcr))
        features['zcr_max'] = to_float(np.max(zcr))
        features['zcr_median'] = to_float(np.median(zcr))
        features['zcr_skew'] = to_float(skew(zcr))
        features['zcr_kurtosis'] = to_float(kurtosis(zcr))

        # ============ 7. CHROMA FEATURES (24 features) ============
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = to_float(np.mean(chroma[i]))
            features[f'chroma_{i}_std'] = to_float(np.std(chroma[i]))

        # ============ 8. FORMANT ESTIMATES (6 features) ============
        freqs = librosa.fft_frequencies(sr=sr)
        avg_spectrum = np.mean(spec, axis=1)
        spectral_peaks, peak_props = find_peaks(avg_spectrum, height=np.max(avg_spectrum) * 0.1)

        if len(spectral_peaks) >= 2:
            features['formant_f1'] = to_float(freqs[spectral_peaks[0]])
            features['formant_f2'] = to_float(freqs[spectral_peaks[1]])
            if len(spectral_peaks) >= 3:
                features['formant_f3'] = to_float(freqs[spectral_peaks[2]])
            else:
                features['formant_f3'] = 0.0
            if len(spectral_peaks) >= 4:
                features['formant_f4'] = to_float(freqs[spectral_peaks[3]])
            else:
                features['formant_f4'] = 0.0

            # Formant dispersion and spacing
            formant_freqs = [freqs[p] for p in spectral_peaks[:4]]
            features['formant_dispersion'] = to_float(np.std(formant_freqs))
            features['formant_mean_spacing'] = to_float(np.mean(np.diff(formant_freqs))) if len(
                formant_freqs) > 1 else 0.0
        else:
            for key in ['formant_f1', 'formant_f2', 'formant_f3', 'formant_f4',
                        'formant_dispersion', 'formant_mean_spacing']:
                features[key] = 0.0

        # ============ 9. TEMPORAL FEATURES (5 features) ============
        features['duration'] = to_float(len(y) / sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = to_float(tempo)
        features['beat_count'] = to_float(len(beats))
        features['duration_voiced'] = features['voiced_ratio'] * features['duration']
        features['duration_silent'] = features['silence_ratio'] * features['duration']

        # ============ 10. HARMONICITY (5 features) ============
        harmonic, percussive = librosa.effects.hpss(y)
        features['harmonic_ratio'] = to_float(np.sum(harmonic ** 2) / (np.sum(y ** 2) + 1e-10))
        features['percussive_ratio'] = to_float(np.sum(percussive ** 2) / (np.sum(y ** 2) + 1e-10))
        features['harmonic_percussive_ratio'] = features['harmonic_ratio'] / (features['percussive_ratio'] + 1e-10)
        features['harmonic_mean_energy'] = to_float(np.mean(np.abs(harmonic)))
        features['percussive_mean_energy'] = to_float(np.mean(np.abs(percussive)))

        # ============ 11. MEL SPECTROGRAM FEATURES (10 features) ============
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        features['mel_mean'] = to_float(np.mean(mel_spec_db))
        features['mel_std'] = to_float(np.std(mel_spec_db))
        features['mel_max'] = to_float(np.max(mel_spec_db))
        features['mel_min'] = to_float(np.min(mel_spec_db))
        features['mel_median'] = to_float(np.median(mel_spec_db))
        features['mel_skew'] = to_float(skew(mel_spec_db.flatten()))
        features['mel_kurtosis'] = to_float(kurtosis(mel_spec_db.flatten()))

        # Mel band energies
        mel_band_energies = np.sum(mel_spec, axis=1)
        features['mel_energy_low'] = to_float(np.sum(mel_band_energies[:43]))  # 0-4kHz
        features['mel_energy_mid'] = to_float(np.sum(mel_band_energies[43:86]))  # 4-6kHz
        features['mel_energy_high'] = to_float(np.sum(mel_band_energies[86:]))  # 6-8kHz

        # ============ 12. PROSODY FEATURES (8 features) ============
        # Rate of speech-related features
        features['syllable_rate'] = features['speech_rate_estimate']
        features['pause_to_speech_ratio'] = features['silence_ratio'] / (features['voiced_ratio'] + 1e-10)
        features['articulation_rate'] = features['speech_rate_estimate'] / (features['voiced_ratio'] + 1e-10)

        # Rhythm variability
        if len(voiced_segments) > 1:
            voiced_durs = [v * (hop_length / sr) for v in voiced_segments]
            features['rhythm_variability'] = to_float(np.std(voiced_durs) / np.mean(voiced_durs))
            features['rhythm_entropy'] = to_float(entropy(np.histogram(voiced_durs, bins=10)[0] + 1e-10))
        else:
            features['rhythm_variability'] = 0.0
            features['rhythm_entropy'] = 0.0

        # Intensity dynamics
        features['intensity_range'] = features['energy_dynamic_range']
        features['intensity_variation'] = features['energy_std'] / (features['energy_mean'] + 1e-10)
        features['intensity_slope'] = to_float(np.polyfit(range(len(rms)), rms, 1)[0])

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def process_all_corpora():
    """Process all available corpora"""

    all_features = []

    # List of corpora to process
    corpora = [
        ('pitt', 'data/pitt_labels.csv', 'data/pitt_alzheimers/', 'data/pitt_healthy/'),
        ('pitt_orig', 'data/pitt_orig_labels.csv', 'data/pitt_orig_alzheimers/', 'data/pitt_orig_healthy/'),
        ('vas', 'data/vas_labels.csv', 'data/vas_alzheimers/', 'data/vas_healthy/'),
    ]

    for corpus_name, labels_file, ad_folder, hc_folder in corpora:
        try:
            print(f"\n{'=' * 70}")
            print(f"Processing {corpus_name.upper()} Corpus...")
            print(f"{'=' * 70}")
            labels_df = pd.read_csv(labels_file)
            total_files = len(labels_df)
            print(f"Total files to process: {total_files}")

            successful = 0
            failed = 0

            for idx, row in labels_df.iterrows():
                filename = row['filename']
                label = row['label']
                participant_id = f"{corpus_name}_{row['participant_id']}"

                if label == 'AD':
                    audio_path = os.path.join(ad_folder, filename)
                else:
                    audio_path = os.path.join(hc_folder, filename)

                if not os.path.exists(audio_path):
                    print(f"  ⚠️ File not found: {audio_path}")
                    failed += 1
                    continue

                # Show progress every file (so you know it's working)
                print(f"  [{idx + 1}/{total_files}] Processing {filename}...", end='\r')

                features = extract_advanced_features(audio_path)
                if features:
                    features['filename'] = filename
                    features['label'] = label
                    features['participant_id'] = participant_id
                    features['corpus'] = corpus_name
                    all_features.append(features)
                    successful += 1
                else:
                    failed += 1

                # Show milestone progress
                if (idx + 1) % 25 == 0:
                    print(
                        f"  ✅ Processed {idx + 1}/{total_files} files ({successful} successful, {failed} failed)     ")

            print(f"\n  ✅ {corpus_name.upper()} Complete: {successful}/{total_files} successful")

        except FileNotFoundError:
            print(f"  ❌ {corpus_name} data not found, skipping...")
            continue

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Save
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/all_features_advanced.csv', index=False)

    print(f"\n{'=' * 70}")
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total samples extracted: {len(df)}")
    print(f"Total features per sample: {len(df.columns) - 4}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nCorpus distribution:")
    print(df['corpus'].value_counts())
    print(f"\nSaved to: results/all_features_advanced.csv")

    return df

if __name__ == "__main__":
    df = process_all_corpora()