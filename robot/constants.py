import pathlib
import os

### Task parameters
# DATA_DIR = os.path.expanduser('/workspace/dataset')
DATA_DIR = os.path.expanduser('/root/dataset')
TASK_CONFIGS = {
    'mask_demo':{
        'dataset_dir': DATA_DIR + '/dsr_block_sort_with_mask',
        'episode_len': 5000,
        'train_ratio': 0.99,
        'camera_names': ['lhand_camera', 'rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_l', 'dsr_r'],
        # 'sample_weights': [3,1],
    },
    'text_demo':{
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
