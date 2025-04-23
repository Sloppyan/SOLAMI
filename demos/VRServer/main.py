import asyncio
from pynput.keyboard import Listener
from datetime import datetime
from aic_utils.state_machine_utils import MotionStateMachine, FaceStateMachine
from aic_utils.redis_utils import RedisConnection
from aic_utils.data_loader_utils import load_config
from aic_utils.audio_comm_utils import websocket_server
from aic_utils.file_utils import read_json_file
from aic_utils.diamond_utils import create_session, delete_session

# Load configuration files
face_config = "demos/VRServer/datasets/face_config.json"
motion_config = "demos/VRServer/datasets/motion_config.json"
main_config = read_json_file("demos/VRServer/config.json")
url_solami = ""  # URL pointing to the backend service, needs to be deployed by the user

class AnimationController:
    """
    Main controller for managing animation, character state, and communication with backend services.
    
    This class coordinates motion and facial animations, handles character switching,
    manages communication sessions with backend services, and processes user input.
    """
    
    def __init__(self, redis_host, redis_port, redis_db, redis_key):
        """
        Initialize the animation controller.
        
        Args:
            redis_host (str): Redis server hostname or IP address
            redis_port (int): Redis server port number
            redis_db (int): Redis database number
            redis_key (str): Redis stream key for animation data
        """
        self.send_audio_flag = False  # Flag for audio sending
        self.record_entry_id = False  # Flag for recording entry IDs
        self.motion_state_machine = None  # Will hold body animation state machine
        self.face_state_machine = None  # Will hold facial animation state machine
        self.current_char_id = main_config["start_up_char"]  # Initial character ID
        self.current_method = main_config["start_up_method"]  # Initial interaction method
        self.current_session_name = main_config["user_name"]  # User session name
        self.current_session_id = None  # Backend session ID
        self.redis_comm = RedisConnection(redis_host, redis_port, redis_db, redis_key)  # Redis connection
        self.init_state_machine()  # Initialize animation state machines
        self.target_prefix = "f1c1"  # Target identifier prefix

    def init_state_machine(self):
        """
        Initialize motion and facial animation state machines.
        
        Loads animation configuration and sets up state machines with 
        the current character ID.
        """
        motion_paths = load_config(motion_config)
        face_paths = load_config(face_config)
        self.motion_state_machine = MotionStateMachine(motion_paths, blend_duration=0.3)
        self.face_state_machine = FaceStateMachine(face_paths, blend_duration=0.1)
        self.motion_state_machine.set_char_id(self.current_char_id)
        self.face_state_machine.set_char_id(self.current_char_id)

    def on_press(self, key):
        """
        Handle keyboard input for controlling the animation system.
        
        This function responds to specific key presses:
        - 'z': Restart the backend session
        - 'p': Change to a "Wave" animation
        
        Args:
            key: The key that was pressed
        """
        try:
            if key.char == 'z':
                # Restart backend session
                delete_session(url_solami, self.current_session_id)
                self.current_method = "solami"
                now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.current_session_name = main_config["user_name"] + "_" + now_str
                self.current_session_id = create_session(url_solami, self.current_session_name, self.current_method, self.current_char_id)
            
            elif key.char == 'p':
                # Trigger wave animation
                self.motion_state_machine.change_state("Wave")
                self.record_entry_id = True
        except AttributeError:
            pass
        
    def check_char_id_change(self):
        """
        Check if the character ID has changed in Redis and update if needed.
        
        This allows external systems to trigger character changes by setting
        the 'char_id' key in Redis.
        """
        new_char_id = self.redis_comm.r.get("char_id")
        if new_char_id is not None:
            new_char_id = new_char_id.decode('utf-8')
        else:
            new_char_id = "Vrmgirl"  # Default fallback character
            
        if new_char_id != self.current_char_id:
            print(f"Character ID changed from {self.current_char_id} to {new_char_id}")
            self.current_char_id = new_char_id
            self.redis_comm.clear_ai_stream()  # Clear previous animation data
            self.init_state_machine()  # Reinitialize with new character
            now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.current_session_name = main_config["user_name"] + "_" + now_str
            
            # Create a new backend session with the new character
            delete_session(url_solami, self.current_session_id)
            self.current_session_id = create_session(url_solami, self.current_session_name, self.current_method, self.current_char_id)
            

    async def run(self):
        """
        Main loop for the animation controller.
        
        This asynchronous method continually updates the animation state,
        sends frames to Redis, and handles character changes and animation transitions.
        """
        # Set up keyboard listener
        listener = Listener(on_press=lambda key: self.on_press(key))
        listener.start()
        
        # Initialize the system
        self.redis_comm.clear_ai_stream()
        now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.current_session_name = main_config["user_name"] + "_" + now_str
        self.current_session_id = create_session(url_solami, self.current_session_name, self.current_method, self.current_char_id)
        
        # Main animation loop
        while True:
            self.check_char_id_change()  # Check for character changes
            
            # Get and combine motion and facial animation frames
            motion_frame = self.motion_state_machine.get_current_frame()
            face_frame = self.face_state_machine.get_current_frame()
            combined_frame = {**motion_frame, **face_frame}
            
            # Send frame to Redis
            entry_id = self.redis_comm.send_frame_data(combined_frame)
            if self.record_entry_id:
                self.redis_comm.send_change_id(entry_id)
                self.record_entry_id = False
                
            # Wait for next frame based on FPS
            await asyncio.sleep(1 / self.motion_state_machine.get_current_fps())

            # Handle animation state changes
            if self.motion_state_machine.current_state != "Idle" and self.motion_state_machine.animation_complete:
                self.motion_state_machine.change_state("Idle")
                
            if self.face_state_machine.current_state != "Idle" and self.face_state_machine.animation_complete:
                self.face_state_machine.change_state("Idle")


async def main():
    """
    Main application entry point.
    
    Sets up the animation controller and WebSocket server, and runs them
    concurrently using asyncio.
    """
    config = read_json_file("demos/VRServer/config.json")
    server_address = config.get("server_address", "127.0.0.1")
    controller = AnimationController(redis_host=server_address, redis_port=6379, redis_db=0, redis_key='aiStream')
    
    # Create a task for the WebSocket server
    websocket_task = asyncio.create_task(websocket_server(controller))
    
    # Run the main controller in parallel
    await controller.run()

if __name__ == "__main__":
    asyncio.run(main())