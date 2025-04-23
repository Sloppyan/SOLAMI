import torchaudio
import random
import torch

def load_audio_segments(audio_path, sample_rate, segment_duration=5, segment_num=5, one_channel=True):
    metadata = torchaudio.info(audio_path)
    or_num_frames = metadata.num_frames
    orig_sample_rate = metadata.sample_rate
    or_segment_length = segment_duration * orig_sample_rate

    segment_num = max(min(segment_num, or_num_frames // or_segment_length), 1)
    waveform, or_sample_rate = torchaudio.load(
        audio_path,
    )

    if or_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=or_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)
    if one_channel and waveform.shape[0] >= 2:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    segments_list = []
    start_frame=0
    segment_length = segment_duration * sample_rate
    for i in range(segment_num):
        segment = waveform[:, start_frame:start_frame + segment_length]
        segments_list.append(segment)
        start_frame += segment_length
    
    return segments_list


def load_audio_sample(audio_path, sample_rate, min_duration=5, max_duration=5, one_channel=False, start_from_begin=True):
    metadata = torchaudio.info(audio_path)
    num_frames = metadata.num_frames
    orig_sample_rate = metadata.sample_rate
    segment_length = random.randint(min_duration, max_duration) * orig_sample_rate

    start_frame = 0 if start_from_begin else random.randint(0, max(0, num_frames - segment_length))
    waveform, or_sample_rate = torchaudio.load(
        audio_path,
        frame_offset=start_frame,
        num_frames=segment_length,
    )

    if or_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=or_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    if one_channel and waveform.shape[0] >= 2:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    
    return waveform
    
    
def load_audio(audio_path, sample_rate, segment_duration=5, one_channel=True, start_from_begin=False):
    metadata = torchaudio.info(audio_path)
    num_frames = metadata.num_frames
    orig_sample_rate = metadata.sample_rate
    segment_length = segment_duration * orig_sample_rate if segment_duration != -1 else num_frames

    start_frame = 0 if start_from_begin else random.randint(0, max(0, num_frames - segment_length))
    waveform, or_sample_rate = torchaudio.load(
        audio_path,
        frame_offset=start_frame,
        num_frames=segment_length,
    )

    if or_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=or_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    if one_channel and waveform.shape[0] >= 2:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    
    return waveform


def encode_music_by_path(audio_path, sample_rate, model, processor, device, segment_duration=-1, one_channel=True, start_from_begin=True):
    # load the audio as a PyTorch tensor
    if isinstance(audio_path, (list, tuple)):
        waveform = [ load_audio(p, sample_rate, segment_duration, 
                                one_channel, start_from_begin).numpy() for p in audio_path]
    else:
        waveform = load_audio(audio_path, sample_rate, segment_duration=segment_duration, one_channel=one_channel).squeeze(0)
    inputs = processor(raw_audio=waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = model.encode(inputs["input_values"].to(device) , inputs["padding_mask"].to(device) )
    return encoder_outputs.audio_codes


def encode_music(waveform_list, sample_rate, model, processor, device):
    # pre-process the inputs
    inputs = processor(raw_audio=waveform_list, sampling_rate=sample_rate, return_tensors="pt")
    # EncodecFeatureExtractor
    # explicitly encode then decode the audio inputs
    with torch.no_grad():
        encoder_outputs = model.encode(inputs["input_values"].to(device) , inputs["padding_mask"].to(device) )
    
    # print('inputs["input_values"].shape', inputs["input_values"].shape)
    # print('encoder_outputs.audio_codes.shape', encoder_outputs.audio_codes.shape)
    return encoder_outputs.audio_codes