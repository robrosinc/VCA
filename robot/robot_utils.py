import numpy as np
import time
import cv2
class ImageRecorder:
    def __init__(self, camera_names, init_node=False, is_debug=False):
        import rospy
        from collections import deque
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = camera_names #['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_depth', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'lhand_camera':
                image_callback_func = self.image_cb_lhand_cam
                depth_callback_func = self.depth_cb_lhand_cam
            elif cam_name == 'rhand_camera':
                image_callback_func = self.image_cb_rhand_cam
                depth_callback_func = self.depth_cb_rhand_cam
            elif cam_name == 'head_camera':
                image_callback_func = self.image_cb_head_cam
                depth_callback_func = self.depth_cb_head_cam
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}/image_raw", Image, image_callback_func)
            # rospy.Subscriber(f"/{cam_name}/aligned_depth_to_color/image_raw", Image, depth_callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def image_cb_lhand_cam(self, data):
        cam_name = 'lhand_camera'
        return self.image_cb(cam_name, data)

    def image_cb_rhand_cam(self, data):
        cam_name = 'rhand_camera'
        return self.image_cb(cam_name, data)
    
    def image_cb_head_cam(self, data):
        cam_name = 'head_camera'
        return self.image_cb(cam_name, data)
    
    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def depth_cb_lhand_cam(self, data):
        cam_name = 'lhand_camera'
        return self.depth_cb(cam_name, data)

    def depth_cb_rhand_cam(self, data):
        cam_name = 'rhand_camera'
        return self.depth_cb(cam_name, data)
    
    def depth_cb_head_cam(self, data):
        cam_name = 'head_camera'
        return self.depth_cb(cam_name, data)
    
    def depth_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_depth', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))

    def get_depth_images(self):
        depth_dict = dict()
        for cam_name in self.camera_names:
            depth_dict[cam_name] = getattr(self, f'{cam_name}_depth')
        return depth_dict
    
    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()