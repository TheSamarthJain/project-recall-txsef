import os
import pandas as pd


def create_labels_csv(ad_folder, hc_folder, output_csv):
    """Create labels CSV from folder structure"""

    labels = []

    # Process AD files
    if os.path.exists(ad_folder):
        ad_files = [f for f in os.listdir(ad_folder) if f.endswith(('.mp3', '.wav'))]
        print(f"Found {len(ad_files)} AD files in {ad_folder}")

        for filename in ad_files:
            # Extract participant ID from filename (e.g., "001-0.mp3" -> "001")
            participant_id = filename.split('-')[0] if '-' in filename else filename.split('.')[0]

            labels.append({
                'filename': filename,
                'label': 'AD',
                'participant_id': participant_id
            })

    # Process HC files
    if os.path.exists(hc_folder):
        hc_files = [f for f in os.listdir(hc_folder) if f.endswith(('.mp3', '.wav'))]
        print(f"Found {len(hc_files)} HC files in {hc_folder}")

        for filename in hc_files:
            # Extract participant ID from filename
            participant_id = filename.split('-')[0] if '-' in filename else filename.split('.')[0]

            labels.append({
                'filename': filename,
                'label': 'HC',
                'participant_id': participant_id
            })

    # Create DataFrame
    df = pd.DataFrame(labels)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"✅ Created {output_csv}")
    print(f"   Total files: {len(df)}")
    if 'label' in df.columns:
        print(f"   AD: {len(df[df['label'] == 'AD'])}")
        print(f"   HC: {len(df[df['label'] == 'HC'])}")
    else:
        print("   (no files found — skipping label breakdown)")
    print()

    return df


if __name__ == "__main__":
    print("Creating label CSV files...\n")

    # Create Pitt labels (if not already exists)
    if not os.path.exists('data/pitt_labels.csv'):
        create_labels_csv(
            'data/pitt_alzheimers',
            'data/pitt_healthy',
            'data/pitt_labels.csv'
        )
    else:
        print("✅ pitt_labels.csv already exists, skipping...\n")

    # Create Pitt-orig labels
    create_labels_csv(
        'data/pitt_orig_alzheimers',
        'data/pitt_orig_healthy',
        'data/pitt_orig_labels.csv'
    )

    # Create VAS labels
    create_labels_csv(
        'data/vas_alzheimers',
        'data/vas_healthy' if os.path.exists('data/vas_healthy') else 'data/vas_alzheimers',
        # VAS might not have healthy controls
        'data/vas_labels.csv'
    )

    print("\n✅ All label files created!")