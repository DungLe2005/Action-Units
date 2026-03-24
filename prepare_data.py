import os
import pandas as pd
from tqdm import tqdm

def parse_au_file(file_path):
    """
    Parse DISFA AU text file format.
    Format: "000.jpg     0" or "000    0" depending on sub-version
    Returns a dictionary: {frame_name: intensity}
    """
    data = {}
    if not os.path.exists(file_path):
        return data
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('~'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                frame_id = parts[0]
                # Ensure frame_id has .jpg extension if it doesn't
                if not frame_id.endswith('.jpg'):
                    frame_id = f"{int(frame_id):03d}.jpg"
                try:
                    intensity = int(parts[1])
                    data[frame_id] = intensity
                except ValueError:
                    continue
    return data

def main():
    root_dir = "AUs_DATA"
    labels_dir = os.path.join(root_dir, "Labels")
    images_dir = os.path.join(root_dir, "Images")
    output_csv = os.path.join(root_dir, "labels.csv")
    
    AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    threshold = 2
    
    all_data = []
    
    subjects = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_label_path = os.path.join(labels_dir, subject)
        trial_folders = [d for d in os.listdir(subject_label_path) if os.path.isdir(os.path.join(subject_label_path, d))]
        
        for trial in trial_folders:
            trial_label_path = os.path.join(subject_label_path, trial)
            trial_image_path = os.path.join(images_dir, subject, trial)
            
            if not os.path.exists(trial_image_path):
                print(f"Warning: Image folder {trial_image_path} not found. Skipping.")
                continue
            
            # Load all AU files for this trial
            au_data = {}
            for au in AU_LIST:
                au_file = os.path.join(trial_label_path, f"AU{au}.txt")
                au_data[au] = parse_au_file(au_file)
            
            # Get all frames in image folder
            frames = [f for f in os.listdir(trial_image_path) if f.endswith('.jpg')]
            
            for frame in frames:
                row = {
                    "image_path": f"{subject}/{trial}/{frame}"
                }
                
                # Check binary activation for each AU
                for au in AU_LIST:
                    intensity = au_data[au].get(frame, 0)
                    row[f"AU{au}"] = 1 if intensity >= threshold else 0
                
                all_data.append(row)
                
    df = pd.DataFrame(all_data)
    # Reorder columns to have image_path first, then AU order
    cols = ["image_path"] + [f"AU{au}" for au in AU_LIST]
    df = df[cols]
    
    df.to_csv(output_csv, index=False)
    print(f"Finished! Total frames: {len(df)}")
    print(f"CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()
