import os
import pandas as pd
from tqdm import tqdm

def prepare_data(data_dir="AUs_DATA"):
    labels_dir = os.path.join(data_dir, "Labels")
    images_dir = os.path.join(data_dir, "Images")
    
    # Expected AUs based on DISFA / typical set
    expected_aus = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]
    
    all_records = []
    
    # Iterate through subject folders (e.g., SN001, SN002)
    subjects = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d)) and d.startswith("SN")]
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_label_dir = os.path.join(labels_dir, subject)
        
        # Iterate through trials (e.g., A1_AU1_TrailNo_1)
        trials = [d for d in os.listdir(subject_label_dir) if os.path.isdir(os.path.join(subject_label_dir, d))]
        
        for trial in trials:
            trial_label_dir = os.path.join(subject_label_dir, trial)
            trial_image_dir = os.path.join(images_dir, subject, trial)
            
            # Dictionary to store AU intensities per image: { "000.jpg": {"AU1": 0, "AU2": 1, ...} }
            trial_data = {}
            
            # Read each AU file
            for au in expected_aus:
                au_file = os.path.join(trial_label_dir, f"{au}.txt")
                if not os.path.exists(au_file):
                    continue
                    
                with open(au_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    # Skip header or empty lines
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        intensity = float(parts[1])
                        
                        # Convert intensity > 0 to 1 (binary classification)
                        # If you want multi-class intensity, remove this binarization.
                        binary_label = 1 if intensity > 0 else 0
                        
                        if img_name not in trial_data:
                            trial_data[img_name] = {au_col: 0 for au_col in expected_aus}
                            
                        trial_data[img_name][au] = binary_label
            
            # Convert dictionary to list of records
            for img_name, au_labels in trial_data.items():
                # Define relative path like "SN001/A1_AU1_TrailNo_1/000.jpg"
                rel_img_path = f"{subject}/{trial}/{img_name}"
                
                # Check if image actually exists (optional, uncomment if needed)
                # full_img_path = os.path.join(images_dir, subject, trial, img_name)
                # if not os.path.exists(full_img_path):
                #     continue
                
                record = {"image_path": rel_img_path}
                record.update(au_labels)
                all_records.append(record)
                
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Ensure correct column order
    columns = ["image_path"] + expected_aus
    # If some AUs were completely missing, they might not be in the df, so only keep ones that exist
    columns = [c for c in columns if c in df.columns]
    df = df[columns]
    
    # Save to CSV
    output_path = os.path.join(data_dir, "labels.csv")
    df.to_csv(output_path, index=False)
    print(f"\nGenerated {output_path} with {len(df)} records and {len(columns)-1} AUs.")
    
if __name__ == "__main__":
    prepare_data()
