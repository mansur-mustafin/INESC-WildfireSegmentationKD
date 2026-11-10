# Adapted from pereira-gha/activefire, src/utils/unzip_patches.py (CC BY 4.0).
# Source: https://github.com/pereira-gha/activefire/blob/main/src/utils/unzip_patches.py

import zipfile
from glob import glob
import os
import shutil

DELETE_ZIP_AFTER_EXTRACTION = True

# This 2 folders cannot be the same:
FULL_DATASET_ZIPS_PATH = './datasets/Landsat8-ActiveFire/compressed/'  # where the continent ZIPs are stored
FULL_DATASET_UNZIP_PATH = './datasets/Landsat8-ActiveFire/'  # where extracted files will be stored

patches_output_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'images', 'patches')
masks_output_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'masks', 'patches')
voting_output_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'masks', 'voting')
intersection_output_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'masks', 'intersection')

for directory in [patches_output_dir, masks_output_dir, voting_output_dir, intersection_output_dir]:
    os.makedirs(directory, exist_ok=True)

print(f'Unzipping images to {patches_output_dir}')
print(f'Unzipping masks to {masks_output_dir}')

total_files = 0
zips_continents = glob(os.path.join(FULL_DATASET_ZIPS_PATH, '*.zip'))

for zip_continent in zips_continents:
    print(f'Unzipping: {zip_continent}')

    with zipfile.ZipFile(zip_continent, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f'Num of files in ZIP: {len(file_list)}')

        zip_ref.extractall(FULL_DATASET_UNZIP_PATH)

    if DELETE_ZIP_AFTER_EXTRACTION:
        os.remove(zip_continent)

    patches_zips = glob(os.path.join(FULL_DATASET_UNZIP_PATH, '*.zip'))
    print(f'Num of ZIPs unpacked: {len(patches_zips)}')

    print('Unzipping patches...')
    num_files = 0

    for patches_zip in patches_zips:
        output_dir = patches_output_dir

        if patches_zip.endswith('masks_derivates.zip'):
            with zipfile.ZipFile(patches_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                zip_ref.extractall(FULL_DATASET_UNZIP_PATH)
                num_files += len(file_list)

            for derivate_patch in glob(os.path.join(FULL_DATASET_UNZIP_PATH, '*.tif')):
                if '_voting_' in derivate_patch.lower():
                    shutil.move(derivate_patch, os.path.join(voting_output_dir, os.path.basename(derivate_patch)))
                elif '_intersection_' in derivate_patch.lower():
                    shutil.move(derivate_patch, os.path.join(intersection_output_dir, os.path.basename(derivate_patch)))

            os.remove(patches_zip)
            continue

        if patches_zip.endswith('masks.zip'):
            output_dir = masks_output_dir

        with zipfile.ZipFile(patches_zip, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(output_dir)
            num_files += len(file_list)

        os.remove(patches_zip)

    total_files += num_files
    print(f'Processed: {zip_continent} - {num_files} files extracted')

print(f'Total files unzipped: {total_files}')
print('Done!')
