import sys
import os
sys.path.append("SOLAMI/models/vla/")
sys.path.append("SOLAMI/models/vla/anygpt/src")
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from m_utils.prompter import *
from m_utils.anything2token import *
from m_utils.anything2token import modality_tokens_to_string
from einops import rearrange
import re
import json
from m_utils.loggings import get_logger
import fire
import torch.distributed as dist
import debugpy
from tqdm import tqdm


def process_speech_data(part=0, period=4, gpu_id=0, max_items=None):
    DEVICE = 'cuda:%d'%gpu_id 
    speech_tokenizer_config = "SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/config.json"
    speech_tokenizer_path = "SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/ckpt.dev"

    speech_meta_path = "SOLAMI/extra/AnyInstruct/speech_conv/metadata.jsonl"
    speech_dir = "SOLAMI/extra/AnyInstruct/speech_conv/speech"
    output_dir = "SOLAMI_data/audio/anyinstruct"
    output_path = os.path.join(output_dir, "anyinstruct_{}_{}.jsonl".format(part, period))

    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)        
    speech_tokenizer.eval()
    speech_tokenizer.to(device=DEVICE)

    if max_items is not None:
        if max_items <= 0:
            raise ValueError("max_items must be a positive integer when provided.")


    def encode_speech(
            audio_path,
            logger=None,
        ):
        wav, sr = torchaudio.load(audio_path)
        num_frames = wav.size(1)
        duration = num_frames / sr
        if logger is not None:
            audio_path_log = os.path.join(*audio_path.split("/")[-2:])
            # logger.info(f"Path: {audio_path_log} duration: {duration:.2f} seconds")
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(DEVICE)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]


    logger = get_logger(local_rank=0, save_path=os.path.join(output_dir, "speech_process.log"), log_level='info')

    line_counter = 0
    data_buffer = []
    processed_total = 0
    should_stop = False

    def flush_buffer(buffer, remaining_limit=None):
        """Write buffered items and return (written_count, remaining_buffer)."""
        if not buffer or remaining_limit == 0:
            return 0, buffer
        if remaining_limit is None:
            to_write = buffer
            rest = []
        else:
            to_write = buffer[:remaining_limit]
            rest = buffer[remaining_limit:]
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in to_write:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return len(to_write), rest
    
    with open(speech_meta_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(speech_meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing", unit="line"):
            line_counter += 1
            if line_counter % period != part:
                continue
            chat_json = json.loads(line.strip())
            
            
            processed_data = {}
            processed_data['id'] = f"anyinstruct_{line_counter}"
            processed_data['chat'] = []
            chat = chat_json['chat']
            for idx, turn in enumerate(chat):
                if "message" in turn and "speech" in turn:
                    role = 'user' if turn["role"] == "USER" else "assistant"
                    text = turn["message"]
                    speech_path = turn["speech"]
                    speech_path = os.path.join(speech_dir, speech_path)
                    try:
                        speech_code = encode_speech(speech_path, logger)
                    # except:
                    #     logger.error(f"Error processing {speech_path}")
                    #     break
                    except Exception as exc:
                        logger.error("Error processing %s: %s", speech_path, exc)
                        break
                    speech_str = modality_tokens_to_string(speech_code, modality="speech")
                    data_item = {
                        'role': role,
                        "speech_path": speech_path,
                        "text": text,
                        "speech": speech_str,
                    }
                    processed_data['chat'].append(data_item)
            data_buffer.append(processed_data)
            
            if len(data_buffer) >= 50:
                remaining_limit = None if max_items is None else max_items - processed_total
                written, data_buffer = flush_buffer(data_buffer, remaining_limit)
                processed_total += written
                if written:
                    logger.info('Processed {} lines in the {}-th part (total items written: {})'.format(line_counter, part, processed_total))
                if max_items is not None and processed_total >= max_items:
                    should_stop = True
                    break
    if not should_stop and data_buffer and (max_items is None or processed_total < max_items):
        remaining_limit = None if max_items is None else max_items - processed_total
        written, _ = flush_buffer(data_buffer, remaining_limit)
        processed_total += written
        if written:
            logger.info('Processed {} lines in the {}-th part (total items written: {})'.format(line_counter, part, processed_total))


if __name__ == "__main__":
    fire.Fire(process_speech_data)
