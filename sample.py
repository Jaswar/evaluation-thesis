import json
import os
import random

def main(root_path, num_vids=5, take_length=10):
    with open(os.path.join(root_path, 'takes.json'), 'r') as f:
        takes = json.load(f)
    vids = random.sample(takes, num_vids)
    for vid in vids:
        take_name = vid['take_name']
        start_time = vid['task_start_sec']
        end_time = vid['task_end_sec']
        selected_start = random.uniform(start_time, end_time - take_length)
        selected_end = selected_start + take_length
        print(take_name, selected_start, selected_end)

if __name__ == '__main__':
    root_path = 'ego_exo_4d'
    main(root_path)