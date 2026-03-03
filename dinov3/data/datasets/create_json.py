path_to_data = "/home/lukas/data/brain-t1-dataset"

import os
import json
import nibabel as nib

def create_nifti_json(input_folder, output_json):
    samples = []
    for fname in os.listdir(input_folder):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            print(f"Processing {fname}...")
            fpath = os.path.join(input_folder, fname)
            img = nib.load(fpath)
            shape = list(img.shape)
            spacing = img.header.get_zooms()
            samples.append({
                "shape": shape,
                "image": fpath,
                "spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],

            })
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {output_json}")

if __name__ == "__main__":

    input_folder = path_to_data
    output_json = "/home/lukas/data/braint1.json"
    create_nifti_json(input_folder, output_json)