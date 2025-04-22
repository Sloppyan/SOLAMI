import asyncio
from pynput.keyboard import Listener
from datetime import datetime
from aic_utils.state_machine_utils import MotionStateMachine, FaceStateMachine
from aic_utils.redis_utils import RedisConnection
from aic_utils.data_loader_utils import load_config
from aic_utils.audio_comm_utils import websocket_server, send_audio
from aic_utils.file_utils import read_json_file
from aic_utils.diamond_utils import create_session, interact, delete_session
from aic_utils.a2f_utils import generate_face_data

face_config = "demos/VRServer/datasets/face_config.json"
motion_config = "demos/VRServer/datasets/motion_config.json"
main_config = read_json_file("demos/VRServer/config.json")
url_8081 = "http://kub-api.zoe.sensetime.com/embodied/forward0"
url_8082 = "http://kub-api.zoe.sensetime.com/embodied/forward1"

class AnimationController:
    def __init__(self, redis_host, redis_port, redis_db, redis_key):
        self.send_audio_flag = False
        self.record_entry_id = False
        self.motion_state_machine = None
        self.face_state_machine = None
        self.current_char_id = main_config["start_up_char"]
        self.current_method = main_config["start_up_method"]
        self.current_session_name = main_config["user_name"]
        self.current_session_id = None
        self.redis_comm = RedisConnection(redis_host, redis_port, redis_db, redis_key)
        self.init_state_machine()
        self.target_prefix = "f1c1"

    def init_state_machine(self):
        motion_paths = load_config(motion_config)
        face_paths = load_config(face_config)
        self.motion_state_machine = MotionStateMachine(motion_paths, blend_duration=0.3)
        self.face_state_machine = FaceStateMachine(face_paths, blend_duration=0.1)
        self.motion_state_machine.set_char_id(self.current_char_id)
        self.face_state_machine.set_char_id(self.current_char_id)

    def on_press(self, key):
        try:
            if key.char == 'z':
                if self.current_method == "solami":
                    delete_session(url_8081, self.current_session_id)
                else:
                    delete_session(url_8082, self.current_session_id)
                self.current_method = "solami"
                now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.current_session_name = main_config["user_name"] + "_" + now_str
                self.current_session_id = create_session(url_8081, self.current_session_name, self.current_method, self.current_char_id)
            elif key.char == 'x':
                if self.current_method == "solami":
                    delete_session(url_8081, self.current_session_id)
                else:
                    delete_session(url_8082, self.current_session_id)
                self.current_method = "llm+speech"
                now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.current_session_name = main_config["user_name"] + "_" + now_str
                self.current_session_id = create_session(url_8082, self.current_session_name, self.current_method, self.current_char_id)
            elif key.char == 'c':
                if self.current_method == "solami":
                    delete_session(url_8081, self.current_session_id)
                else:
                    delete_session(url_8082, self.current_session_id)
                self.current_method = "dlp+motiongpt"
                now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.current_session_name = main_config["user_name"] + "_" + now_str
                self.current_session_id = create_session(url_8082, self.current_session_name, self.current_method, self.current_char_id)
            elif key.char == 'v':
                if self.current_method == "solami":
                    delete_session(url_8081, self.current_session_id)
                else:
                    delete_session(url_8082, self.current_session_id)
                self.current_method = "dlp+motiongpt+retrieval"
                now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.current_session_name = main_config["user_name"] + "_" + now_str
                self.current_session_id = create_session(url_8082, self.current_session_name, self.current_method, self.current_char_id)
            elif key.char == 'b':
                if self.current_method == "solami":
                    delete_session(url_8081, self.current_session_id)
                else:
                    delete_session(url_8082, self.current_session_id)
                self.current_method = "echo_method_0"
                now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.current_session_name = main_config["user_name"] + "_" + now_str
                self.current_session_id = create_session(url_8082, self.current_session_name, self.current_method, self.current_char_id)
            elif key.char == 'n':
                if self.current_method == "solami":
                    delete_session(url_8081, self.current_session_id)
                else:
                    delete_session(url_8082, self.current_session_id)
            elif key.char == 'p':
                self.motion_state_machine.change_state("Wave")
                self.record_entry_id = True
        except AttributeError:
            pass
        
    def check_char_id_change(self):
        new_char_id = self.redis_comm.r.get("char_id")
        if new_char_id is not None:
            new_char_id = new_char_id.decode('utf-8')
        else:
            new_char_id = "Vrmgirl"
            
        if new_char_id != self.current_char_id:
            print(f"Character ID changed from {self.current_char_id} to {new_char_id}")
            self.current_char_id = new_char_id
            self.redis_comm.clear_ai_stream()
            self.init_state_machine()
            now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.current_session_name = main_config["user_name"] + "_" + now_str
            if self.current_method == "solami":
                delete_session(url_8081, self.current_session_id)
                self.current_session_id = create_session(url_8081, self.current_session_name, self.current_method, self.current_char_id)
            else:
                delete_session(url_8082, self.current_session_id)
                self.current_session_id = create_session(url_8082, self.current_session_name, self.current_method, self.current_char_id)

    async def run(self):
        listener = Listener(on_press=lambda key: self.on_press(key))
        listener.start()
        self.redis_comm.clear_ai_stream()
        now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.current_session_name = main_config["user_name"] + "_" + now_str
        if self.current_method == "solami":
            self.current_session_id = create_session(url_8081, self.current_session_name, self.current_method, self.current_char_id)
        else:
            self.current_session_id = create_session(url_8082, self.current_session_name, self.current_method, self.current_char_id)
        while True:
            self.check_char_id_change()
            
            motion_frame = self.motion_state_machine.get_current_frame()
            face_frame = self.face_state_machine.get_current_frame()
            combined_frame = {**motion_frame, **face_frame}
            entry_id = self.redis_comm.send_frame_data(combined_frame)
            if self.record_entry_id:
                self.redis_comm.send_change_id(entry_id)
                self.record_entry_id = False
            await asyncio.sleep(1 / self.motion_state_machine.get_current_fps())

            if self.motion_state_machine.current_state != "Idle" and self.motion_state_machine.animation_complete:
                self.motion_state_machine.change_state("Idle")
                
            if self.face_state_machine.current_state != "Idle" and self.face_state_machine.animation_complete:
                self.face_state_machine.change_state("Idle")


async def main():
    config = read_json_file("demos/VRServer/config.json")
    server_address = config.get("server_address", "127.0.0.1")
    controller = AnimationController(redis_host=server_address, redis_port=6379, redis_db=0, redis_key='aiStream')
    # 创建一个任务运行 websocket_server
    websocket_task = asyncio.create_task(websocket_server(controller))
    # 并行运行主任务
    await controller.run()

if __name__ == "__main__":
    asyncio.run(main())