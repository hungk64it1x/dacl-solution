
import os
import json
from tqdm import tqdm

def jsons2jsonl(jsons_dir, jsonl_path):
    if not os.path.exists(jsons_dir) or not os.path.isdir(jsons_dir):
        raise ValueError(f"The directory {jsons_dir} does not exist.")

    with open(jsonl_path, 'w') as jsonl_file:

        for json_file_name in os.listdir(jsons_dir):
            if json_file_name.endswith('.json'):
                json_file_path = os.path.join(jsons_dir, json_file_name)
                with open(json_file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    jsonl_file.write(json.dumps(json_data) + '\n')
                
if __name__ == '__main__':
    jsons_dir = 'output/dev/json_output'
    jsonl_path = 'output/dev/jsonl_output/dev.jsonl'
    jsons2jsonl(jsons_dir, jsonl_path)