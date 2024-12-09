import os
import subprocess
from tqdm import tqdm

frame_rate = 30 # fps

nfs_coded_dir = '/Users/kelly/Documents/tcig-coded-sensor/NfS_coded_'+str(frame_rate)
nfs_files = sorted(os.listdir(nfs_coded_dir))
nfs_files.remove('.DS_Store')

for file in tqdm(nfs_files):
    groups = sorted(os.listdir(os.path.join(nfs_coded_dir, file)))
    if '.DS_Store' in groups:
        groups.remove('.DS_Store')
    for group in groups:
        group_dir = os.path.join(nfs_coded_dir, file, group)
        output_fname = os.path.join(group_dir, f'{file}_{group}')
        
        subprocess.run(['python', 'simulate.py', 
                        '--params', 'data/params.mat',
                        '--mask', 'masks/coded_exposure_2x2.bmp',
                        '--input_imgs', f"{group_dir}/?.jpg",
                        '--output_fname', output_fname,
                        '--mode', 'multi_in_single_out',
                        '--cam_type', 't6',])
