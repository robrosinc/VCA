import pathlib
import os

### Task parameters
# DATA_DIR = os.path.expanduser('/home/robros-ai/dg/IL_data/new')
DATA_DIR = os.path.expanduser('/workspace/dataset2')
TASK_CONFIGS = {
    ### BLOCK SORT
    'blocksort_mask':{
        'dataset_dir': DATA_DIR,
        'episode_len': 5000,
        'train_ratio': 0.99,
        'camera_names': ['lhand_camera', 'rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_l', 'dsr_r'],
        # 'sample_weights': [3,1],
    },
    'blocksort_text':{
        'dataset_dir': DATA_DIR,
        'episode_len': 1700,
        'train_ratio': 0.99,
        'camera_names': ['lhand_camera', 'rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_l', 'dsr_r'],
        # 'sample_weights': [3,1],
    },
    'hanoi':{
        'dataset_dir': DATA_DIR,
        'episode_len': 1600,
        'train_ratio': 0.99,
        'camera_names': ['rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_r'],
        # 'sample_weights': [3,1],
    },
    'hanoi2':{
        'dataset_dir': DATA_DIR + '/hanoi',
        'episode_len': 3000,
        'train_ratio': 0.99,
        'camera_names': ['rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_r'],
        # 'sample_weights': [3,1],
    }
}

HZ = 20
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path
