import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
from tqdm import tqdm
import argparse

def process_audio(audio_path, output_dir, gpu_id="-1"):
    print(f"Processing {audio_path}...")
    print(f"Output dir: {output_dir}")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="$YOUT_HF_TOKENS TODO")
    if gpu_id != "-1":
        pipeline.to(torch.device("cuda:" + gpu_id))
    # apply pretrained pipeline
    diarization = pipeline(audio_path)
    # load audio file
    audio = AudioSegment.from_file(audio_path)
    
    ext = audio_path.split(".")[-1]
    
    previous_speaker = None
    previous_end_time = None
    combined_segment = None
    segment_index = 0
    previous_start_time = None
    
    for i, (turn, _, speaker) in tqdm(enumerate(diarization.itertracks(yield_label=True))):
        start_time = turn.start * 1000
        end_time = turn.end * 1000
        
        if speaker == previous_speaker and previous_end_time is not None and start_time - previous_end_time <= 1500:
            combined_segment += audio[previous_end_time:end_time]
            previous_end_time = end_time
        else:
            if combined_segment is not None and len(combined_segment) > 2000:
                speaker_folder = os.path.join(output_dir, f"{previous_speaker}")
                os.makedirs(speaker_folder, exist_ok=True)
                combined_segment.export(f"{speaker_folder}/segment_{segment_index}_{int(previous_start_time)/1000}_to_{int(previous_end_time/1000)}.{ext}", format=ext)
                segment_index += 1

            combined_segment = audio[start_time:end_time]
            previous_speaker = speaker
            previous_end_time = end_time
            previous_start_time = start_time
        
    if combined_segment is not None and len(combined_segment) > 2000:
        speaker_folder = os.path.join(output_dir, f"{previous_speaker}")
        os.makedirs(speaker_folder, exist_ok=True)
        combined_segment.export(f"{speaker_folder}/segment_{segment_index}_{int(previous_start_time)/1000}_to_{int(previous_end_time/1000)}.{ext}", format=ext)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="SOLAMI_data/audio/character/splited_pyannote_audio")
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()
    
    # check audio path exist
    audio_path = args.audio_path
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not exist!")
        exit()
    audio_name = os.path.basename(audio_path)
    output_dir = os.path.join(args.output_dir, audio_name.split(".")[0])
    if os.path.exists(output_dir):
        print(f"Output dir {output_dir} already exist!")
        exit()
    process_audio(audio_path, output_dir, gpu_id=args.gpu_id)