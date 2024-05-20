import os
from shutil import copy2
from time import sleep

# Run this code in the root directory: i.e. '/your_folders/.../DBF'

roots = ['mmdetection/mmdet/models/detectors', 'mmrotate/mmrotate/models/detectors']
stages = ['my_src/preliminaries/stages_for_hbb', 'my_src/preliminaries/stages_for_obb']

assert len(roots) == len(stages)

for i in range(len(roots)):
    candidates = [f'single_stage.py', f'two_stage.py']
    for candidate in candidates:
        if not os.path.exists(f'{roots[i]}/{candidate[:-3]} (original).py'):
            os.rename(f'{roots[i]}/{candidate}', f'{roots[i]}/{candidate[:-3]} (original).py')
            print(f'* Renamed: from "{roots[i]}/{candidate}" -> to "{roots[i]}/{candidate[:-3]} (original).py"')
            sleep(0.1)
            copy2(f'{stages[i]}/{candidate}', f'{roots[i]}/{candidate}')
            print(f'* Copied: from "stages/{candidate}" -> to "{roots[i]}/{candidate}"')
        else:
            print(f'* Already overwritten: "{candidate}" at "{roots[i]}"')
