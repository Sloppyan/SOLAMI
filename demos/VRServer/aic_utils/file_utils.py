import json
import aiofiles
from bidict import bidict

async def write_audio_file(file_path, audio_data):
    """
    Asynchronously write audio data to a file.
    
    Args:
        file_path (str): Path where the audio file will be saved
        audio_data (bytes): Binary audio data to write
    """
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(audio_data)
        print("write audio file finished")
        
def get_key_from_value(dictionary, value):
    """
    Find the key in a dictionary that corresponds to a specific value.
    
    Args:
        dictionary (dict): The dictionary to search in
        value: The value to search for
        
    Returns:
        object: The key corresponding to the value, or None if not found
    """
    return next((k for k, v in dictionary.items() if v == value), None)

def read_json_file(file_path):
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file to read
        
    Returns:
        dict: The parsed JSON data
    """
    with open(file_path, 'r') as file:
        return json.load(file)
    
def parse_bmap_line(line):
    """
    Parse a single line from a bone mapping (.bmap) file.
    
    The format is expected to be bone_name%flag1%type%values1%values2%value3%flag2%flag3%flag4
    
    Args:
        line (str): A line from the .bmap file
        
    Returns:
        tuple: (bone_name, rest_of_data_as_dict)
    """
    parts = line.split('%')
    bone_name = parts[0] if parts[0] != "None" else None
    rest = {
        "flag1": parts[1],
        "type": parts[2],
        "values1": parts[3],
        "values2": parts[4],
        "value3": parts[5],
        "flag2": parts[6],
        "flag3": parts[7],
        "flag4": parts[8]
    }
    return bone_name, rest

def read_bmap_file(file_path):
    """
    Read and parse a bone mapping (.bmap) file that defines relationships between bones
    in different skeletal hierarchies.
    
    This function processes the specialized format of .bmap files and creates a
    bidirectional mapping between source and target bone names.
    
    Args:
        file_path (str): Path to the .bmap file
        
    Returns:
        bidict: A bidirectional dictionary mapping source bone names to target bone names
    """
    mapping = {}
    try:
        with open(file_path, 'r') as file:
            current_bone = None
            for line in file:
                line = line.strip()
                if not line:
                    continue
                elif '%' in line:
                    bone_name, rest = parse_bmap_line(line)
                    if bone_name:
                        current_bone = bone_name
                        mapping[current_bone] = rest
                elif current_bone:
                    if "additional" not in mapping[current_bone]:
                        mapping[current_bone]["additional"] = []
                    mapping[current_bone]["additional"].append(line)
    except Exception as e:
        print(f"Error reading file: {e}")
    # Extract the actual bone mapping
    bone_mapping = bidict()
    for target_bone, details in mapping.items():
        if "additional" in details:
            source_bone = details["additional"][0]
            bone_mapping[source_bone] = target_bone
        else:
            bone_mapping[target_bone] = target_bone
    return bone_mapping