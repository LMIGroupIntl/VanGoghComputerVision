import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import random

def prepare_dataset(excel_path, source_images_folder, output_folder, sheet_name='vg_cv_data_july31', random_seed=42):
   
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Loading Excel file from sheet '{sheet_name}'...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"Total rows in Excel: {len(df)}")
 
    df['run_1'] = 0
    
    # Define criteria for authentic images
    authentic_criteria = (
        (df['is_wikiart_vangogh_oil_painting'] == 1) & 
        (df['exclude_from_training'] == 0)
    )
    
    # Define criteria for imitation images 
    imitation_artists = ['Henri Matisse', 'Henri Toulouse-Lautrec', 'Maurice Prendergast', 'Vik Muniz', 'Paul Cezanne']
    imitation_criteria = df['artist'].isin(imitation_artists)
    
    # Filter the data
    authentic_df = df[authentic_criteria].copy()
    imitation_df = df[imitation_criteria].copy()
    
    print(f"Found {len(authentic_df)} authentic Van Gogh oil paintings (before undersampling)")
    print(f"Found {len(imitation_df)} imitation images (before undersampling)")
    print(f"Imitation breakdown by artist:")
    for artist in imitation_artists:
        count = len(imitation_df[imitation_df['artist'] == artist])
        print(f"  - {artist}: {count}")
    
    print(f"Using all {len(authentic_df)} authentic Van Gogh paintings")
    
    target_imitation_count = 755
    if len(imitation_df) > target_imitation_count:
        print(f"Undersampling imitation images from {len(imitation_df)} to {target_imitation_count}...")
        imitation_df = imitation_df.sample(n=target_imitation_count, random_state=random_seed).copy()
    else:
        print(f"Imitation images ({len(imitation_df)}) <= {target_imitation_count}, using all available")
    
    print(f"Final counts: {len(authentic_df)} authentic, {len(imitation_df)} imitation")

    output_path = Path(output_folder)
    authentic_path = output_path / 'authentic'
    imitation_path = output_path / 'imitation'
    
 
    authentic_path.mkdir(parents=True, exist_ok=True)
    imitation_path.mkdir(parents=True, exist_ok=True)
    
   
    print("Copying authentic images...")
    copied_authentic = 0
    missing_authentic = []
    
    for idx, row in authentic_df.iterrows():
        source_file = Path(source_images_folder) / row['image']
        dest_file = authentic_path / row['image']
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied_authentic += 1
                # Mark as used in run_1
                df.loc[idx, 'run_1'] = 1
            except Exception as e:
                print(f"Error copying {source_file}: {e}")
                missing_authentic.append(row['image'])
        else:
            missing_authentic.append(row['image'])
    
   
    print("Copying imitation images...")
    copied_imitation = 0
    missing_imitation = []
    
    for idx, row in imitation_df.iterrows():
        source_file = Path(source_images_folder) / row['image']
        dest_file = imitation_path / row['image']
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied_imitation += 1
                # Mark as used in run_1
                df.loc[idx, 'run_1'] = 1
            except Exception as e:
                print(f"Error copying {source_file}: {e}")
                missing_imitation.append(row['image'])
        else:
            missing_imitation.append(row['image'])
    
  
    print(f"\n=== COPY RESULTS ===")
    print(f"Authentic images copied: {copied_authentic}/{len(authentic_df)}")
    print(f"Imitation images copied: {copied_imitation}/{len(imitation_df)}")
    print(f"Total images copied: {copied_authentic + copied_imitation}")
    
    if missing_authentic:
        print(f"\nMissing authentic images ({len(missing_authentic)}):")
        for img in missing_authentic[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_authentic) > 10:
            print(f"  ... and {len(missing_authentic) - 10} more")
    
    if missing_imitation:
        print(f"\nMissing imitation images ({len(missing_imitation)}):")
        for img in missing_imitation[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_imitation) > 10:
            print(f"  ... and {len(missing_imitation) - 10} more")
    
    updated_excel_path = excel_path.replace('.xlsx', '_updated.xlsx')
    df.to_excel(updated_excel_path, sheet_name=sheet_name, index=False)
    print(f"\nUpdated Excel saved to: {updated_excel_path}")
    print(f"Total rows marked with run_1=1: {df['run_1'].sum()}")
    

    print(f"\n=== FINAL DATASET SUMMARY ===")
    print(f"Dataset location: {output_folder}")
    print(f"Authentic images: {copied_authentic}")
    print(f"Imitation images: {copied_imitation}")
    print(f"Total images: {copied_authentic + copied_imitation}")
    print(f"Class balance: {copied_authentic/(copied_authentic + copied_imitation):.1%} authentic")
    
    print(f"\nFinal imitation breakdown by artist:")
    final_imitation_df = df[df['run_1'] == 1][df[df['run_1'] == 1]['artist'].isin(imitation_artists)]
    for artist in imitation_artists:
        count = len(final_imitation_df[final_imitation_df['artist'] == artist])
        if count > 0:
            print(f"  - {artist}: {count}")
    
    return {
        'authentic_copied': copied_authentic,
        'imitation_copied': copied_imitation,
        'missing_authentic': missing_authentic,
        'missing_imitation': missing_imitation,
        'updated_excel_path': updated_excel_path
    }


if __name__ == "__main__":
    excel_path = r"C:\Users\majit\Downloads\vg_cv_data_july31.xlsx"
    source_images_folder = r"C:\Users\majit\Downloads\images_saved"
    output_folder = r"C:\Users\majit\Downloads\AR_model_data\run_1"
    sheet_name = 'vg_cv_data_july31'  
    
  
    results = prepare_dataset(
        excel_path=excel_path,
        source_images_folder=source_images_folder,
        output_folder=output_folder,
        sheet_name=sheet_name,
        random_seed=42  
    )
    
    print("\nDataset preparation complete!")
    print("You can now update the training script with:")
    print(f"root_dir = '{output_folder}'")