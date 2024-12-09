import os
from tqdm import tqdm
import shutil

# Get list of files in the NfS directory
nfs_dir = '/Users/kelly/Documents/tcig-coded-sensor/NfS'
nfs_files = os.listdir(nfs_dir)
nfs_files.remove('.DS_Store')

# Define frame rate (either 30 or 240)
frame_rate = 30 # fps

nfs_coded_dir = '/Users/kelly/Documents/tcig-coded-sensor/NfS_coded_'+str(frame_rate)

for file in tqdm(nfs_files):
    frames = sorted(os.listdir(os.path.join(nfs_dir, file, str(frame_rate), file)))
    if '.DS_Store' in frames:
        frames.remove('.DS_Store')
    # os.makedirs(os.path.join(nfs_dir, file, str(frame_rate), 'grouped'), exist_ok=True)

    # split frames into groups of 4 (discard the last few frames if necessary)
    for i in range(0, len(frames), 4):
        if i + 4 > len(frames):
            break
        group = frames[i:i+4]
        group_dir = os.path.join(nfs_coded_dir, file, f'{i//4}')
        os.makedirs(group_dir, exist_ok=True)
        for frame in group:
            shutil.copy(os.path.join(nfs_dir, file, str(frame_rate), file, frame), os.path.join(group_dir, frame))
