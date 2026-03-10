#!/usr/bin/env python3
"""
Script to update oasis_data.csv with new subjects found in nifti_converted folder.
Requires the original OASIS cross-sectional CSV for age metadata.
"""

import os
import csv
import argparse
from pathlib import Path
import pandas as pd
from typing import Set, Dict


def get_existing_subjects(csv_path: Path) -> Set[str]:
    """Read the current CSV and return set of existing subject IDs."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} does not exist yet. Will create new file.")
        return set()
    
    df = pd.read_csv(csv_path)
    return set(df['subject_id'].tolist())


def get_nifti_subjects(nifti_dir: Path) -> Set[str]:
    """Scan nifti_converted folder and return set of all subject IDs."""
    subjects = set()
    
    if not nifti_dir.exists():
        raise FileNotFoundError(f"Directory not found: {nifti_dir}")
    
    # Find all OAS1_*_MR* directories
    for item in nifti_dir.iterdir():
        if item.is_dir() and item.name.startswith('OAS1_'):
            subjects.add(item.name)
    
    return subjects


def load_oasis_metadata(metadata_path: Path) -> Dict[str, int]:
    """
    Load age information from the original OASIS cross-sectional metadata.
    Supports CSV and Excel files. Returns dict mapping subject_id to age.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"OASIS metadata file not found: {metadata_path}\n"
            f"Please download oasis_cross-sectional.csv from the OASIS website:\n"
            f"https://www.oasis-brains.org/"
        )

    suffix = metadata_path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        df = pd.read_excel(metadata_path)
    else:
        df = pd.read_csv(metadata_path)

    # Expect columns like 'ID' and 'Age'
    if 'ID' not in df.columns or 'Age' not in df.columns:
        raise ValueError(
            "Metadata file must contain 'ID' and 'Age' columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Map ID to Age
    age_map = {}
    for _, row in df.iterrows():
        subject_id = str(row['ID']).strip()  # Format: OAS1_0001_MR1
        age = row['Age']
        age_map[subject_id] = age

    return age_map


def update_csv(csv_path: Path, age_map: Dict[str, int], new_subjects: Set[str]):
    """Update the oasis_data.csv with new subjects."""
    # Read existing data
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['subject_id', 'age'])
    
    # Prepare new rows
    new_rows = []
    missing_ages = []
    
    for subject_id in sorted(new_subjects):
        if subject_id in age_map:
            new_rows.append({
                'subject_id': subject_id,
                'age': age_map[subject_id]
            })
        else:
            missing_ages.append(subject_id)
    
    if missing_ages:
        print(f"\nWarning: Could not find age information for {len(missing_ages)} subjects:")
        for subj in missing_ages[:10]:  # Show first 10
            print(f"  - {subj}")
        if len(missing_ages) > 10:
            print(f"  ... and {len(missing_ages) - 10} more")
    
    # Append new rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Sort by subject_id for consistency
        df = df.sort_values('subject_id').reset_index(drop=True)
        
        # Save
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Successfully added {len(new_rows)} new subjects to {csv_path}")
    else:
        print("\nNo new subjects with age information to add.")


def main():
    parser = argparse.ArgumentParser(
        description='Update oasis_data.csv with new subjects from nifti_converted folder'
    )
    parser.add_argument(
        '--nifti-dir',
        type=Path,
        default=Path.home() / 'data' / 'OASIS' / 'nifti_converted',
        help='Path to nifti_converted directory'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path(__file__).parent / 'oasis_data.csv',
        help='Path to oasis_data.csv to update'
    )
    parser.add_argument(
        '--metadata-csv',
        type=Path,
        default= '/home/lukas/3Ddinov3/dinov3/eval/medical/oasis_cross-sectional-5708aa0a98d82080.xlsx',
        help='Path to original OASIS metadata file (CSV or Excel)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without modifying files'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OASIS Data CSV Updater")
    print("=" * 70)
    
    # Get subjects from filesystem
    print(f"\n1. Scanning nifti_converted directory: {args.nifti_dir}")
    nifti_subjects = get_nifti_subjects(args.nifti_dir)
    print(f"   Found {len(nifti_subjects)} subjects in nifti_converted")
    
    # Get existing subjects from CSV
    print(f"\n2. Reading existing CSV: {args.csv_path}")
    existing_subjects = get_existing_subjects(args.csv_path)
    print(f"   Found {len(existing_subjects)} subjects already in CSV")
    
    # Find new subjects
    new_subjects = nifti_subjects - existing_subjects
    print(f"\n3. New subjects to add: {len(new_subjects)}")
    
    if not new_subjects:
        print("\n✓ All subjects are already in the CSV. Nothing to update.")
        return
    
    # Show some examples
    print("\n   Examples of new subjects:")
    for subj in sorted(list(new_subjects))[:5]:
        print(f"     - {subj}")
    if len(new_subjects) > 5:
        print(f"     ... and {len(new_subjects) - 5} more")
    
    # Load metadata
    print(f"\n4. Loading age metadata from: {args.metadata_csv}")
    age_map = load_oasis_metadata(args.metadata_csv)
    print(f"   Loaded age information for {len(age_map)} subjects")
    
    # Update CSV
    if args.dry_run:
        print("\n[DRY RUN] Would update CSV with new subjects (no changes made)")
        matched = sum(1 for s in new_subjects if s in age_map)
        print(f"   Would add {matched} subjects with age information")
    else:
        print(f"\n5. Updating {args.csv_path}...")
        update_csv(args.csv_path, age_map, new_subjects)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
