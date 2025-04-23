import numpy as np
from aic_utils.interpolation_utils import linear_interpolation, quintic_interpolation, slerp
from aic_utils.data_loader_utils import load_motion_data
from aic_utils.data_loader_utils import load_face_data

class MotionStateMachine:
    """
    State machine to manage character body animations and transitions between different states.
    
    This class handles loading, playing, and blending between different motion animations,
    providing smooth transitions between different animation states.
    """
    
    def __init__(self, animation_paths, blend_duration=0.3):
        """
        Initialize the motion state machine.
        
        Args:
            animation_paths (dict): Dictionary mapping state names to animation configuration
            blend_duration (float, optional): Duration in seconds for blending between states.
                                             Defaults to 0.3.
        """
        self.animation_paths = animation_paths
        self.states = {}  # Stores loaded animation data for each state
        self.current_state = "Idle"  # Default initial state
        self.frame_index = 0  # Current frame in the animation
        self.animation_complete = False  # Flag indicating if current animation has completed
        self.blend_duration = blend_duration  # Duration for transitions
        self.blend_frames = []  # Interpolated frames for transitions
        self.is_blending = False  # Whether currently in a blending transition
        self.last_frame = None  # Last frame shown, used for blending
        self.char_id = "chappie"  # Default character ID

    def get_current_frame(self):
        """
        Get the current animation frame based on the current state and frame index.
        
        If in a blending transition, returns the appropriate interpolated frame.
        Otherwise, returns the frame from the current animation state.
        
        Returns:
            dict: Animation frame data containing bone positions and rotations
        """
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
        """
        Change to a new animation state with smooth blending.
        
        This method handles the transition from the current animation state to a new one,
        generating interpolated frames for a smooth blend.
        
        Args:
            new_state (str): Name of the new animation state
        """
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
        """
        Load animation data for a specific state.
        
        Reads the animation data from the specified path for the given state
        and stores it in the state machine.
        
        Args:
            state (str): Name of the animation state to load
        """
        if state in self.animation_paths:
            config = self.animation_paths[state]
            # print(f"loading state: {state} with config: {config}")
            print(f"load {self.char_id} state {state}")
            frames = load_motion_data(config['path'], self.char_id)
            fps = config['fps']
            self.states[state] = {'data':frames, 'fps': fps}
            
    def set_char_id(self, char_id):
        """
        Set the character ID and clear loaded animations if it changes.
        
        This allows switching between different character models by reloading
        appropriate animations for the new character.
        
        Args:
            char_id (str): Identifier for the character model
        """
        if self.char_id != char_id:
            self.char_id = char_id
            self.states.clear()

    def generate_blend_frames(self, last_frame, next_frame, duration):
        """
        Generate interpolated frames for smooth transition between animations.
        
        Attempts to use quintic interpolation for smoother results, but falls back
        to linear interpolation if an error occurs.
        
        Args:
            last_frame (dict): Last frame of the current animation
            next_frame (dict): First frame of the new animation
            duration (float): Duration in seconds for the transition
            
        Returns:
            list: List of interpolated frames for the transition
        """
        try:
            return quintic_interpolation(last_frame, next_frame, duration, slerp)
        except Exception as e:
            print(f"Error during quintic blending: {e}. Falling back to linear interpolation.")
            return linear_interpolation(last_frame, next_frame, duration, slerp)

    def set_animation_complete(self):
        """
        Manually mark the current animation as complete.
        """
        self.animation_complete = True

    def get_current_fps(self):
        """
        Get the frames per second of the current animation state.
        
        Returns:
            int: Frames per second rate for the current animation or blending
        """
        if self.is_blending:
            return 30
        return self.states[self.current_state]['fps']

    def play_animation(self):
        """
        Start playing the animation.
        """
        self.is_playing = True

    def stop_animation(self):
        """
        Stop playing the animation.
        """
        self.is_playing = False
        
class FaceStateMachine:
    """
    State machine to manage character facial animations.
    
    Similar to MotionStateMachine but specialized for facial animations,
    handling the loading and playback of different facial expressions and states.
    """
    
    def __init__(self, face_animation_paths, blend_duration=0.3):
        """
        Initialize the face state machine.
        
        Args:
            face_animation_paths (dict): Dictionary mapping state names to facial animation configuration
            blend_duration (float, optional): Duration in seconds for blending between states.
                                             Defaults to 0.3.
        """
        self.face_animation_paths = face_animation_paths
        self.states = {}  # Stores loaded animation data for each state
        self.current_state = "Idle"  # Default initial state
        self.frame_index = 0  # Current frame in the animation
        self.animation_complete = False  # Flag indicating if current animation has completed
        self.last_frame = None  # Last frame shown
        self.char_id = "chappie"  # Default character ID

    def get_current_frame(self):
        """
        Get the current facial animation frame based on the current state and frame index.
        
        Returns:
            dict: Facial animation frame data containing blendshape values
                 Returns empty dict if no valid frames are available
        """
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
            return {}  # Return empty dict, not None

    def change_state(self, new_state):
        """
        Change to a new facial animation state.
        
        Args:
            new_state (str): Name of the new facial animation state
        """
        if new_state == "AI" or new_state != self.current_state:
            self.current_state = new_state
            self.frame_index = 0
            self.animation_complete = False
            self.load_state(new_state)  # Load data for the new state
            # print(f"State changed to {new_state}")
            
    def load_state(self, state):
        """
        Load facial animation data for a specific state.
        
        Reads the facial animation data from the specified path for the given state
        and stores it in the state machine.
        
        Args:
            state (str): Name of the facial animation state to load
        """
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
        """
        Set the character ID and clear loaded animations if it changes.
        
        Args:
            char_id (str): Identifier for the character model
        """
        if self.char_id != char_id:
            self.char_id = char_id
            self.states.clear()  # Clear states to load animations for the new character

    def get_current_fps(self):
        """
        Get the frames per second of the current facial animation state.
        
        Returns:
            int: Frames per second rate for the current animation
        """
        return self.states[self.current_state]['fps'] if self.current_state in self.states else 30



