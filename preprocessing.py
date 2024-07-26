import os
import shutil
import tqdm

from multiprocessing import Pool
from pathlib import Path

from biomedmbz_glioma.dataset_preprocessing import preprocessing_and_save , collect_cropped_image_sizes

n_jobs=8

if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    source_directory=...
    target_directory=...
    
    print('Preprocessing:')
    print('Source dir:', source_directory)
    print('Target dir:', target_directory)
    print('Patch size:', patch_size)
    
    if os.path.exists(target_directory) and os.path.isdir(target_directory):
        shutil.rmtree(target_directory)
    Path(target_directory).mkdir(parents=True, exist_ok=False)
    example_ids = os.listdir(source_directory)
    # get the patch size (median)
    median_patch_size = collect_cropped_image_sizes(source_directory,example_ids)
    def fn(x):
        preprocessing_and_save(target_directory, source_directory, x, median_patch_size)
    with Pool(n_jobs) as p:
        r = list(tqdm.tqdm(p.imap(fn, example_ids), total=len(example_ids)))
    # ---------------------------------------------------------------------------
