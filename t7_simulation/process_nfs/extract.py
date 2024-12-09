import os
import shutil

from tqdm import tqdm

nfs_coded_dir = '/Users/kelly/Documents/tcig-coded-sensor/NfS_coded_30'
nfs_files = sorted(os.listdir(nfs_coded_dir))
nfs_files.remove('.DS_Store')

nfs_extract_dir = '/Users/kelly/Documents/tcig-coded-sensor/NfS_controlnet_30'
os.makedirs(nfs_extract_dir, exist_ok=True)
os.makedirs(os.path.join(nfs_extract_dir, 'source'), exist_ok=True)
os.makedirs(os.path.join(nfs_extract_dir, 'target'), exist_ok=True)

for file in tqdm(nfs_files):
    groups = sorted(os.listdir(os.path.join(nfs_coded_dir, file)))
    if '.DS_Store' in groups:
        groups.remove('.DS_Store')
    for group in groups:
        group_dir = os.path.join(nfs_coded_dir, file, group)
        frames = sorted(os.listdir(group_dir))

        if '.DS_Store' in frames:
            frames.remove('.DS_Store')

        # Extract source frames
        source_frames = sorted([f for f in frames if os.path.basename(f).startswith(f"{file}_{group}_left_")])
        assert len(source_frames) == 4
        for frame in source_frames:
            shutil.copy(os.path.join(group_dir, frame), os.path.join(nfs_extract_dir, 'source', os.path.basename(frame)))
        
        # Extract target frames
        target_frames = sorted([f for f in frames if os.path.basename(f).startswith(f"{file}_{group}_frame_")])
        assert len(target_frames) == 4
        for frame in target_frames:
            shutil.copy(os.path.join(group_dir, frame), os.path.join(nfs_extract_dir, 'target', os.path.basename(frame)))