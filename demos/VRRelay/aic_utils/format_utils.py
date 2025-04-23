import json

def format_data(value):
    """
    Convert various data types to string format suitable for transmission or display.
    
    This utility function handles different data types and converts them to
    appropriate string representations:
    - Lists/tuples become comma-separated strings
    - Dictionaries become JSON strings
    - Numbers are converted to string representations
    - Other types are converted using str()
    
    Args:
        value: The data to format, can be of various types
        
    Returns:
        str: A string representation of the input data
    """
    if isinstance(value, (list, tuple)):
        # Convert lists or tuples to a comma-separated string
        return ','.join(map(str, value))
    elif isinstance(value, dict):
        # Convert dictionaries to a JSON string
        return json.dumps(value)
    elif isinstance(value, (int, float)):
        # Convert numbers to a string
        return str(value)
    else:
        # Assume it's already a string or something easily convertible
        return str(value)
