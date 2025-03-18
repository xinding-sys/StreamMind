# # from videollama2 import conversation as conversation_lib




# # sources = [[{'from': 'human', 'value': '<video>\nPlease describe the video content in detail based on the provided information.'}, {'from': 'gpt', 'value': 'Sure! In the video, we can see a child skiing on the snow while holding two poles and wearing skis. The child is followed by a group of adults skiing behind him. At one point, the child comes to a stop and starts to walk until he begins to slide down the hill again. This video seems to be a fun activity for the child to enjoy the snow and learn how to ski. It also shows the child’s determination to continue skiing even after stopping and starting again. The video provides a scenic view of the snow-covered hills and highlights the importance of learning and practicing the sport of skiing. Overall, the video captures the joy of wintertime activities and the beauty of nature.'}]]
# # sources = [[{'from': 'human', 'value': '<video>\nPlease describe the video content in detail based on the provided information.'}, {'from': 'gpt', 'value': 'Sure! In the video, we can see a child skiing on the snow while holding two poles and wearing skis. The child is followed by a group of adults skiing behind him. At one point, the child comes to a stop and starts to walk until he begins to slide down the hill again. This video seems to be a fun activity for the child to enjoy the snow and learn how to ski. It also shows the child’s determination to continue skiing even after stopping and starting again. The video provides a scenic view of the snow-covered hills and highlights the importance of learning and practicing the sport of skiing. Overall, the video captures the joy of wintertime activities and the beauty of nature.'}]]
# # conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]
# # conv = conversation_lib.default_conversation.copy()
# # # sources = [[{'from': 'human', 'value': '<video>\nCan you describe the process of making the pasta dish in the video?'}]] 
# # import pdb
# # pdb.set_trace()
# # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
# # # Apply prompt templates
# # conversations = []
# # for i, source in enumerate(sources):
# #     if roles[source[0]["from"]] != conv.roles[0]:
# #         # Skip the first one if it is not from human
# #         source = source[1:]
# #     conv.messages = []
# #     for j, sentence in enumerate(source):
# #         role = roles[sentence["from"]]
# #         assert role == conv.roles[j % 2], f"{i}"
# #         conv.append_message(role, sentence["value"])
# #     conversations.append(conv.get_prompt())

# # print(conversations)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline
# from matplotlib import rcParams

# # 假设你有一个数据列表
# data_50 = [1.73,1.72,1.65,1.60,1.57,1.53,1.49,1.43,1.37,1.35,
#            1.37,1.39,1.42,1.44,1.46,1.46,1.48,1.52,1.53,1.55,
#            1.57,1.59,1.61,1.62,1.62,1.65,1.67,1.69,1.73,1.75,
#            1.76,1.81,1.82,1.84,1.84,1.87,1.92,1.95,1.97,2.05,
#            2.08,2.11,2.12,2.15,2.18,2.21,2.25,2.29,2.31,2.35]

# quant_data_50 = [1.73,1.71,1.63,1.62,1.58,1.54,1.48,1.42,1.36,1.36,
#            1.37,1.39,1.41,1.45,1.46,1.46,1.48,1.52,1.52,1.56,
#            1.57,1.60,1.61,1.62,1.62,1.65,1.67,1.70,1.73,1.75,
#            1.76,1.83,1.81,1.84,1.83,1.85,1.92,1.95,1.98,2.04,
#            2.08,2.13,2.11,2.15,2.17,2.21,2.26,2.28,2.34,2.39]

# cache_data_50 = [1.73,1.73,1.65,1.60,1.56,1.53,1.48,1.44,1.37,1.35,
#            1.36,1.39,1.42,1.44,1.46,1.46,1.48,1.52,1.54,1.57,
#            1.57,1.58,1.61,1.63,1.63,1.66,1.67,1.70,1.73,1.75,
#            1.75,1.81,1.82,1.85,1.86,1.87,1.93,1.95,1.96,2.07,
#            2.08,2.12,2.11,2.16,2.18,2.23,2.27,2.29,2.33,2.38]

# quant_cache_data_50 = [1.73,1.72,1.67,1.61,1.59,1.54,1.49,1.43,1.37,1.35,
#            1.37,1.39,1.43,1.45,1.45,1.46,1.49,1.52,1.54,1.57,
#            1.57,1.60,1.63,1.64,1.64,1.66,1.69,1.70,1.75,1.78,
#            1.80,1.83,1.84,1.86,1.88,1.92,1.94,1.96,2.01,2.10,
#            2.15,2.18,2.22,2.23,2.26,2.30,2.34,2.36,2.40,2.46]

# quant_cache_data_50_correction = [1.73,1.73,1.65,1.60,1.56,1.53,1.48,1.44,1.37,1.35,
#            1.36,1.39,1.42,1.44,1.46,1.47,1.48,1.51,1.54,1.57,
#            1.58,1.58,1.62,1.63,1.63,1.66,1.67,1.72,1.72,1.74,
#            1.75,1.81,1.83,1.85,1.86,1.87,1.92,1.96,1.97,2.05,
#            2.09,2.09,2.12,2.18,2.19,2.23,2.27,2.29,2.33,2.38]

# font_size = 18
# # rcParams.update({'font.size': font_size, 'font.family': font_name})
# # rcParams.update({'font.size': font_size})
# # rcParams.update({
# #     'font.size': font_size,
# #     'font.family': 'serif',
# #     'font.serif': ['DejaVu Serif']
# # })
# plt.rc('font',family='Times New Roman')
# # 生成x轴上的点，根据数据长度
# x1 = np.arange(len(data_50))
# x2 = np.arange(len(quant_data_50))
# x3 = np.arange(len(cache_data_50))
# x4 = np.arange(len(quant_cache_data_50))

# # 设置颜色的深浅，从浅蓝到深蓝
# colors = ['#ADD8E6', '#6495ED', '#4682B4', '#000080']  # 蓝色的渐变

# # 设置线型
# linestyles = ['-', '--', '-.', ':']

# # 创建图形
# plt.figure(figsize=(8, 6))

# # 绘制每一个list数据，不进行平滑处理，直接使用原始数据
# plt.plot(x1, data_50, color="red", linestyle=linestyles[0], label='Original model')
# plt.plot(x2, quant_data_50, color=colors[1], linestyle=linestyles[1], label='Quantized model')
# plt.plot(x3, cache_data_50, color=colors[2], linestyle=linestyles[2], label='Cache model')
# plt.plot(x4, quant_cache_data_50, color="lightgreen", linestyle=linestyles[3], label='Quant-Cache model')
# # plt.plot(x4, quant_cache_data_50, color="darkgreen", linestyle=linestyles[3], label='Quant-Cache model')
# # plt.plot(x4, quant_cache_data_50_correction, color="darkgreen", linestyle=linestyles[1], label='Ours')

# # 设置图例
# plt.legend()

# # 显示网格
# plt.grid(True)

# # 添加标题和标签
# # plt.title('Comparison of Four Lists')
# plt.xticks([0,10,20,30,40,50], labels=['50','40', '30', "20",'10',"0"])

# plt.xlabel('Timestep')

# plt.ylabel('MSE Loss')

# plt.savefig('./exposure_bias_new.png')



import os 

def video_timestamp_to_video_test(video_path,timestamp,half,fps=2,input_device=None):
    # if input_device is None:
    input_device = "cpu"
    video_fps = 25
    segment = video_fps//fps
    frame_stamp = 25 * timestamp
    import pdb
    pdb.set_trace()
    video_encode_list = []
    video_dir = os.path.dirname(video_path)
    video_encode_feature_dir = video_dir.replace("features_video", "features_video_encode_ddp")

    if (frame_stamp//500) == 0:
        video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,0,500))
        if frame_stamp + 100 < 500:
            video_encode_list.append(torch.tensor([1]))
        else:
            video_encode_list.append(torch.tensor([1]))
    else:
        for video_encode_id in range(frame_stamp//500):
            video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,video_encode_id*500,(video_encode_id+1)*500))
            if os.path.exists(video_encode_path):
                frame_id = frame_stamp % 500
                if video_encode_id == (frame_stamp//500) - 1:
                    if frame_stamp % 500 + 100 < 500:
                        video_encode_list.append(torch.tensor([1]))
                    else:
                        video_encode_list.append(torch.tensor([1]))
                else:
                    video_encode_list.append(torch.tensor([1]))
            else:
                continue
    torch.cuda.empty_cache()
    return video_encode_list


# video_timestamp_to_video_test(video_path="/home/v-dingxin/blob/MatchTime/features_video/europe_uefa-champions-league_2014-2015/2015-04-14_-_21-45_Atl._Madrid_0_-_0_Real_Madrid/2_224p.mkv",timestamp = 336, half 

import os

def count_files_in_subdirectories(directory_path):
    file_counts = {}
    for root, dirs, files in os.walk(directory_path):
        file_counts[root] = len(files)
    return file_counts

# 使用示例
directory_path = "/home/v-dingxin/blob/MatchTime/features_video_encode_ddp/italy_serie-a_2016-2017"  # 替换为要统计的目录路径
file_counts = count_files_in_subdirectories(directory_path)

for dir_path, count in file_counts.items():
    print(f"Directory '{dir_path}' contains {count} files.")
    # print(f"{count} ")


