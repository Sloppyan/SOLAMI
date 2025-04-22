import json

def format_data(value):
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
