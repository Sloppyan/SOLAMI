import pandas as pd
import numpy as np

## load all face data
def load_npz_face_anim(npz_file_path):
    face_data = np.load(npz_file_path)
    excluded_columns = ['Timecode', 'BlendShapeCount', 'n_frames']
    relevant_keys = [key for key in face_data.files if key not in excluded_columns]
    relevant_keys = [key[0].lower() + key[1:] for key in relevant_keys]
    
    face_frames = []
    frame_count = face_data["Timecode"].shape[0]
    
    for frame_index in range(frame_count):
        face_frame = {}
        for key in relevant_keys:
            face_frame[key] = face_data[key[0].upper()+key[1:]][frame_index]
        face_frames.append(face_frame)
    return face_frames

## load only eyeBlink and jawOpen
def load_face_anim(csv_file_path):
    face_data = pd.read_csv(csv_file_path)
    
    # Drop the first 2 column (timecode, blendshape count)
    face_data = face_data.iloc[:, 2:]
    face_data.columns = [col[0].lower() + col[1:] for col in face_data.columns]
    
    desired_keys = ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen']
    face_data = face_data[desired_keys]
    
    # Convert each row to a dictionary and append to the list
    face_frames = []
    for _, row in face_data.iterrows():
        face_frame = row.to_dict()
        face_frames.append(face_frame)
        

    return face_frames

# load_face_data("datasets/face_data/xiaoning_lzq.csv", None)
