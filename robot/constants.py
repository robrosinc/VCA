import pathlib
import os

### Task parameters
DATA_DIR = os.path.expanduser('/workspace/dataset')
TASK_CONFIGS = {
    'hanoi_mask':{
        'dataset_dir': DATA_DIR,
        'episode_len': 1600,
        'train_ratio': 0.99,
        'camera_names': ['rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_r'],
        # 'sample_weights': [3,1],
    },
    'hanoi2':{
        'dataset_dir': DATA_DIR,
        'episode_len': 3400,
        'train_ratio': 0.99,
        'camera_names': ['rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_r'],
        # 'sample_weights': [3,1],
    }
}

HZ = 20
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path
