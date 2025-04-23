import asyncio
import websockets
from aic_utils.file_utils import read_json_file, write_audio_file
from aic_utils.npz_saver_utils import process_redis_data
from aic_utils.diamond_utils import audio2audio, interact
from aic_utils.a2f_utils import generate_face_data

config = read_json_file("demos/VRRelay/config.json")
websocket_connection = None
connection_established = asyncio.Event()
url_solami = ""  # URL pointing to the backend service, needs to be deployed by the user


async def receive_audio(websocket, controller):
    """
    Receive audio data from WebSocket client, process it and trigger backend interactions.
    
    This function handles receiving audio from the client, saving it, and communicating
    with the backend service to generate responses. The backend service URL (url_solami)
    needs to be deployed by the user.
    
    Args:
        websocket: The WebSocket connection to receive data from
        controller: Controller object to manage application state
    """
    try:
        audio_file = await websocket.recv()
        audio_user_path = config["audio_user"]
        motion_user_path = config['motion_user']
        audio_ai_path = config["audio_ai"]
        motion_ai_path = config["motion_ai"]
        face_ai_path = config["face_ai"]
        
        task1 = write_audio_file(audio_user_path, audio_file)
        task2 = process_redis_data(motion_user_path)
        
        await asyncio.gather(task1, task2)
        
        await asyncio.to_thread(interact, url_solami, controller.current_session_id, audio_user_path, motion_user_path, audio_ai_path, motion_ai_path)
        await asyncio.to_thread(generate_face_data, wav_file=audio_ai_path, save_path=face_ai_path)
        if controller.current_method != "llm+speech":
            controller.motion_state_machine.change_state("AI")
        controller.face_state_machine.change_state("a2f")
        await send_audio(audio_ai_path)
        # controller.record_entry_id = True
        
    except websockets.ConnectionClosed as e:
        if e.code != 1000:
            print(f"Connection closed with error: {e.code} {e.reason}")
    except Exception as e:
        print(f"An error occurred while receiving audio: {e}")
        
    
async def send_audio(file_path):
    """
    Send audio data to the connected WebSocket client.
    
    Args:
        file_path (str): Path to the audio file to send
    """
    global websocket_connection
    await connection_established.wait()  # Wait for connection to be established
    try:
        with open(file_path, "rb") as f:
            audio_data = f.read()
        await websocket_connection.send(audio_data)
        # print("send audio success!!")
        
    except websockets.ConnectionClosed as e:
        if e.code != 1000:
            print(f"Connection closed with error: {e.code} {e.reason}")
    except Exception as e:
        print(f"An error occurred while sending audio: {e}")
  
        
async def interact_async(session_id, audio_user, motion_user, audio_ai, motion_ai):
    """
    Asynchronous wrapper for the interact function to wait for file downloads.
    
    This connects to the backend service which needs to be deployed by the user.
    
    Args:
        session_id (str): ID of the active session
        audio_user (str): Path to the user's audio file
        motion_user (str): Path to the user's motion file
        audio_ai (str): Path to save the AI's audio response
        motion_ai (str): Path to save the AI's motion response
    """
    loop = asyncio.get_event_loop()
    content, exception = await loop.run_in_executor(None, interact, session_id, audio_user, motion_user, audio_ai, motion_ai)
    
    if exception:
        print(f"An error occurred in interact: {exception}")
    else:
        print("interact completed successfully:", content)


async def websocket_handler(websocket, path, controller):
    """
    Handle WebSocket connections.
    
    This function manages the WebSocket connection lifecycle and processes
    received messages.
    
    Args:
        websocket: The WebSocket connection
        path: The connection path
        controller: Controller object to manage application state
    """
    global websocket_connection
    websocket_connection = websocket
    connection_established.set()  # Set the connection established event
    try:
        while True:
            await receive_audio(websocket, controller)
    except websockets.ConnectionClosed as e:
        if e.code != 1000:
            print(f"Connection closed with error: {e.code} {e.reason}")
    except Exception as e:
        print(f"An error occurred in websocket_handler: {e}")
    finally:
        connection_established.clear()
        websocket_connection = None


async def websocket_server(controller):
    """
    Start the WebSocket server to listen for incoming connections.
    
    Args:
        controller: Controller object to manage application state
    """
    try:
        # server_address = config.get("server_address", "127.0.0.1")  # Get server address, default to localhost
        server_address = "0.0.0.0"
        server = await websockets.serve(lambda ws, path: websocket_handler(ws, path, controller), server_address, 8099)
        print(f"WebSocket server started on ws://{server_address}:8099")
        await server.wait_closed()
    except Exception as e:
        print(f"Failed to start WebSocket server: {e}")
