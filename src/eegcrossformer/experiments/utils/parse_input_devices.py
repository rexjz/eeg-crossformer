import json

def parse_input_devices_arg(input_str):
  try:
    # Attempt to parse the string as a JSON object
    parsed = json.loads(input_str)

    # Check if the parsed object is an integer or a list
    if isinstance(parsed, int):
      return parsed
    elif isinstance(parsed, list):
      return parsed
    else:
      raise ValueError("Input is neither a valid integer nor a list.")
  except json.JSONDecodeError:
    # Handle the case where the string is not valid JSON
    try:
      # Attempt to parse the string as an integer
      return int(input_str)
    except ValueError:
      raise ValueError("Input is neither a valid integer nor a list.")
