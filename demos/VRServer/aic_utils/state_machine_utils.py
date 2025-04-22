import numpy as np
from aic_utils.interpolation_utils import linear_interpolation, quintic_interpolation, slerp
from aic_utils.data_loader_utils import load_motion_data
from aic_utils.data_loader_utils import load_face_data

class MotionStateMachine:
    def __init__(self, animation_paths, blend_duration=0.3):
        self.animation_paths = animation_paths
        self.states = {}
        self.current_state = "Idle"
        self.frame_index = 0
        self.animation_complete = False
        self.blend_duration = blend_duration
        self.blend_frames = []
        self.is_blending = False
        self.last_frame = None
        self.char_id = "chappie"

    def get_current_frame(self):
        if self.is_blending:
            frame = self.blend_frames[self.frame_index]
            self.frame_index += 1
            if self.frame_index >= len(self.blend_frames):
                self.is_blending = False
                self.frame_index = 0
            return frame
        
        if self.current_state not in self.states:
            self.load_state(self.current_state)

        frames = self.states[self.current_state]['data']
        frame = frames[self.frame_index]
        self.frame_index += 1
        if self.frame_index >= len(frames):
            self.frame_index = 0
            self.animation_complete = True
        self.last_frame = frame
        return frame

    def change_state(self, new_state):
        if new_state == "AI" or new_state not in self.states:
            self.load_state(new_state)
            
        if new_state != self.current_state:
            last_frame = self.last_frame if self.last_frame else self.get_current_frame()
            next_frame = self.states[new_state]['data'][0]
            self.blend_frames = self.generate_blend_frames(last_frame, next_frame, self.blend_duration)
            self.is_blending = True
            self.frame_index = 0
            self.current_state = new_state
            self.animation_complete = False
            # print(f"State changed to {new_state}")
            
    def load_state(self, state):
        if state in self.animation_paths:
            config = self.animation_paths[state]
            # print(f"loading state: {state} with config: {config}")
            print(f"load {self.char_id} state {state}")
            frames = load_motion_data(config['path'], self.char_id)
            fps = config['fps']
            self.states[state] = {'data':frames, 'fps': fps}
            
    def set_char_id(self, char_id):
        if self.char_id != char_id:
            self.char_id = char_id
            self.states.clear()

    def generate_blend_frames(self, last_frame, next_frame, duration):
        try:
            return quintic_interpolation(last_frame, next_frame, duration, slerp)
        except Exception as e:
            print(f"Error during quintic blending: {e}. Falling back to linear interpolation.")
            return linear_interpolation(last_frame, next_frame, duration, slerp)

    def set_animation_complete(self):
        self.animation_complete = True

    def get_current_fps(self):
        if self.is_blending:
            return 30
        return self.states[self.current_state]['fps']

    def play_animation(self):
        self.is_playing = True

    def stop_animation(self):
        self.is_playing = False
        
class FaceStateMachine:
    def __init__(self, face_animation_paths, blend_duration=0.3):
        self.face_animation_paths = face_animation_paths
        self.states = {}
        self.current_state = "Idle"
        self.frame_index = 0
        self.animation_complete = False
        self.last_frame = None
        self.char_id = "chappie"

    def get_current_frame(self):
        if self.current_state not in self.states:
            self.load_state(self.current_state)

        frames = self.states.get(self.current_state, {}).get('data', [])
        if frames and self.frame_index < len(frames):
            frame = frames[self.frame_index]
            self.frame_index += 1
            if self.frame_index >= len(frames):
                self.frame_index = 0
                self.animation_complete = True
            self.last_frame = frame
            return frame
        else:
            print(f"Warning: No valid face frames for state {self.current_state} at index {self.frame_index}")
            return {}  # 返回空字典，但不是 None

    def change_state(self, new_state):
        if new_state == "AI" or new_state != self.current_state:
            self.current_state = new_state
            self.frame_index = 0
            self.animation_complete = False
            self.load_state(new_state)  # 加载新状态的数据
            # print(f"State changed to {new_state}")
            
    def load_state(self, state):
        if state in self.face_animation_paths:
            config = self.face_animation_paths[state]
            print(f"Loading state: {state} with config: {config}")
            
            frames = load_face_data(config['path'])
            if not frames:
                print(f"Warning: No face frames loaded for state {state}")
            fps = config['fps']
            self.states[state] = {'data': frames, 'fps': fps}
        else:
            print(f"Error: State {state} not found in face_animation_paths")
            self.states[state] = {'data': [], 'fps': 30}

    def set_char_id(self, char_id):
        if self.char_id != char_id:
            self.char_id = char_id
            self.states.clear()  # 清空状态以加载新角色的动画数据

    def get_current_fps(self):
        return self.states[self.current_state]['fps'] if self.current_state in self.states else 30



