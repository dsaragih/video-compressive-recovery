import os
import json

nfs_controlnet_dir = '/Users/kelly/Documents/tcig-coded-sensor/NfS_controlnet_30'
files = sorted(os.listdir(os.path.join(nfs_controlnet_dir, 'source')))

metadata = []
for file in files:
    split = file.split('.')[0].split('_')
    code = split[-1] # 00000, 00001, 00002, 00003
    group = split[-3] # group number (0, 1, 2, ...)
    scene_name = '_'.join(split[:-3]) # scene name, e.g. 'airboard_1'

    source_file = f"{scene_name}_{group}_left_{code}.png"
    target_file = f"{scene_name}_{group}_frame_{code}.png"

    metadata.append({
        'source': os.path.join('source', source_file),
        'target': os.path.join('target', target_file),
        'scene': scene_name,
        'group': group,
        'code': int(code)
    }) 

# Write the metadata to JSON file
with open('/Users/kelly/Documents/tcig-coded-sensor/NfS_controlnet_30/metadata.json', 'w') as f:
    for i, entry in enumerate(metadata):
        json.dump(entry, f)
        if i < len(metadata) - 1:  # If not the last entry
            f.write('\n')  # Write a newline only if it's not the last entry