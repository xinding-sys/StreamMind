# import torch
# import os

# from tqdm import tqdm


# path = '/home/v-dingxin/blob/MatchTime/features_video_encode_ddp'
# # 存储所有文件路径的列表





# def video_timestamp_to_video(path):
#     all_files = []
#     # 递归遍历目录和子目录
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             # 获取每个文件的完整路径
#             full_path = os.path.join(root, file)
#             all_files.append(full_path)


#     # if input_device is None:
#     input_device = "cpu"
#     video_fps = 25
#     fps = 2
#     segment = video_fps//fps
#     import pdb
#     pdb.set_trace()
#     progress_bar = tqdm(total=len(all_files), ncols=80)
#     for video_encoder_path in all_files:
#         save_feature_path = video_encoder_path.replace("features_video_encode_ddp","features_video_encode_ddp_fps")
#         dir_path = os.path.dirname(save_feature_path)
#         os.makedirs(dir_path, exist_ok=True)
#         data = torch.load(video_encoder_path)[:,::segment]
#         torch.save(data,save_feature_path)
#         print( "{} file is save to {}".format(video_encoder_path,save_feature_path))
#         progress_bar.update(1)
# video_timestamp_to_video(path)




import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

path = '/home/v-dingxin/blob/MatchTime/features_video_encode_ddp'

# 设置参数
input_device = "cpu"
video_fps = 25
fps = 2
segment = video_fps // fps

# 存储所有文件路径的列表
def get_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files

# 单个文件的处理函数
def process_file(video_encoder_path, progress_bar, lock):
    try:
        save_feature_path = video_encoder_path.replace("features_video_encode_ddp", "features_video_encode_ddp_fps")
        dir_path = os.path.dirname(save_feature_path)
        os.makedirs(dir_path, exist_ok=True)

        data = torch.load(video_encoder_path)[:, ::segment]
        torch.save(data, save_feature_path)
        
        # 更新进度条（加锁以确保线程安全）
        with lock:
            progress_bar.update(1)
            
        return f"{video_encoder_path} file is saved to {save_feature_path}"
    except Exception as e:
        return f"Failed to process {video_encoder_path}: {e}"

# 多线程处理所有文件
def video_timestamp_to_video(path):
    all_files = get_all_files(path)
    progress_bar = tqdm(total=len(all_files), ncols=80)
    lock = Lock()  # 创建一个锁对象

    # 使用 ThreadPoolExecutor 实现多线程
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_file, file, progress_bar, lock): file for file in all_files}
        
        for future in as_completed(futures):
            result = future.result()
            print(result)

    progress_bar.close()  # 关闭进度条

video_timestamp_to_video(path)
