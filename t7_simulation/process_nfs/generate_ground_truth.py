import os
import cv2
import subprocess
from tqdm import tqdm
from pathlib import Path

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
        frames = sorted(os.listdir(group_dir))

        if '.DS_Store' in frames:
            frames.remove('.DS_Store')

        # Only keep files with .jpg extension
        frames = sorted([frame for frame in frames if frame.endswith('.jpg')])
        print(f"Processing {file}_{group} with {len(frames)} frames")

        for i, frame in enumerate(frames):
            # open the image
            img_path = os.path.join(group_dir, frame)
            img = cv2.imread(img_path, 0) # 0 for grayscale
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(group_dir, f"{file}_{group}_frame_{i:05d}.png"), img)
