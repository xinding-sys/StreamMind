import os
import json
import re

def find_specific_files(root_path, target_filenames):
    paths = []
    # Traverse the directory structure
    for dirpath, _, filenames in os.walk(root_path):
        # Check if either of the target files is in the current directory
        for target_filename in target_filenames:
            if target_filename in filenames:
                # Append the full path of the found file
                paths.append(os.path.join(dirpath, target_filename))
    return paths

def trans_video_2_json(file_paths):
    import pdb
    pdb.set_trace()
        # Replace 'features_video' with 'dataset/MatchTime/train'
    new_path = file_paths.replace("features_video", "dataset/MatchTime/train")
        
    # Replace '1_224p.mkv' or '2_224p.mkv' with 'Labels-caption.json'
    if "1_224p.mkv" in new_path:
        new_path = new_path.replace("1_224p.mkv", "Labels-caption.json")
    elif "2_224p.mkv" in new_path:
        new_path = new_path.replace("2_224p.mkv", "Labels-caption.json")
        
        # Append the modified path to the new list
    new_path
    
    return new_path

# Set the root directory and list of target filenames
root_path = "/home/v-dingxin/blob/MatchTime/features_video"
target_filenames = ["1_224p.mkv", "2_224p.mkv"]
target_filenames = ["1_224p.mkv"]



# # Find all matching files and store paths in a list
# file_paths = find_specific_files(root_path, target_filenames)

json_file = trans_video_2_json("/home/v-dingxin/blob/MatchTime/features_video/england_epl_2015-2016/2016-01-13_-_22-45_Chelsea_2_-_2_West_Brom/2_224p.mkv")

# # Print the list of paths
print(json_file)
# /home/v-dingxin/blob/MatchTime/dataset/MatchTime/train/spain_laliga_2016-2017/2017-01-29_-_22-45_Real_Madrid_3_-_0_Real_Sociedad/1_224p.mkv
# /home/v-dingxin/blob/MatchTime/features_video/spain_laliga_2016-2017/2017-04-15_-_17-15_Gijon_2_-_3_Real_Madrid/1_224p.mkv


def parse_labels_caption(caption_data_path,video_data_path):
    """
    Parses a Labels-caption.json file and extracts the required data.
    Parameters:
        file_path (str): The path to the Labels-caption.json file.
        league (str): The league name.
        game (str): The game name.
    Returns:
        list: A list of tuples containing (half, timestamp, type, anonymized, league, game).
    """
    with open(caption_data_path, 'r') as file:
        data = json.load(file)

    result = []
    for annotation in data.get('annotations', []):
        try:
            gameTime, _ = annotation.get(timestamp_key, ' - ').split(' - ')
            half = int(gameTime.split(' ')[0])
            if half not in [1, 2]:
                continue
            minutes, seconds = map(int, _.split(':'))
            timestamp = minutes * 60 + seconds
            label = annotation.get('label', '')
            anonymized = annotation.get('anonymized', '')
            result.append((half, timestamp, label, anonymized,))
        except ValueError:
            continue
    # print(len(result))
    return result

def extract_video_half(video_data_path):
    # Extract the filename from the path
    filename = os.path.basename(video_data_path)
    
    # Use regex to find the number before the underscore
    match = re.match(r"(\d+)_\d+p\.mkv", filename)
    if match:
        return int(match.group(1))
    return None
def parse_labels_caption(caption_data_path,video_data_path=None):
    """
    Parses a Labels-caption.json file and extracts the required data.
    Parameters:
        file_path (str): The path to the Labels-caption.json file.
        league (str): The league name.
        game (str): The game name.
    Returns:
        list: A list of tuples containing (half, timestamp, type, anonymized, league, game).
    """
    with open(caption_data_path, 'r') as file:
        data = json.load(file)

    label_result = []
    anonymized_result = []
    timestamp_result = []
    half_result = []
    half_base = extract_video_half(video_data_path)
    for annotation in data.get('annotations', []):
        gameTime, _ = annotation.get("gameTime",'').split(' - ')
        half = int(gameTime.split(' ')[0])
        if half != half_base:
            continue
        minutes, seconds = map(int, _.split(':'))
        timestamp = minutes * 60 + seconds
        label_result.append(annotation.get('label', ''))
        anonymized_result.append(annotation.get('anonymized', ''))
        timestamp_result.append(timestamp)
        half_result.append(half)
    return label_result,anonymized_result,timestamp_result,half_result
# label,anony,time,half = parse_labels_caption("/home/v-dingxin/blob/MatchTime/dataset/MatchTime/train/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/Labels-caption.json",
# "/home/v-dingxin/blob/MatchTime/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_224p.mkv")
# print(label)
# print(len(anony))
# print(time)
# print(half)

# half = extract_video_half()
# print(half)