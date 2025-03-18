# # from transformers import AutoModel, AutoConfig

# # # 加载配置文件
# # config = AutoConfig.from_pretrained("/home/v-dingxin/blob/video_llm_save/checkpoint-500/config.json")

# # # 加载分片的权重
# # model = AutoModel.from_pretrained(
# #     "/home/v-dingxin/blob/video_llm_save/checkpoint-500",
# #     config=config,
# #     trust_remote_code=True  # 如果使用自定义模型需要
# # )
# # import pdb
# # pdb.set_trace()
# # print(666)
# # # 注意："path/to/directory/containing/model-*.safetensors" 是包含分片文件和 .index.json 文件的路径。


# import torch
# def logit(p: torch.Tensor) -> torch.Tensor:
#     return torch.log(p / (1 - p))
    
# def test_focal_loss_equals_ce_loss() -> None:
#     inputs = logit(
#             torch.tensor(
#                 [[0.05, 0.9], [0.52, 0.45], [0.89, 0.8], [0.39, 0.5]],
#                 dtype=torch.float32,
#             )
#         )
#     print(inputs)


# test_focal_loss_equals_ce_loss()



import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Add general-purpose spatial feature extraction block
ax.add_patch(patches.Rectangle((0.5, 4.5), 2, 1, edgecolor='black', facecolor='#FFD700', lw=2))
ax.text(1.5, 5, 'Spatial Feature Extractor', ha='center', va='center', fontsize=10, color='black')

# Add temporal redundancy reduction block
ax.add_patch(patches.Rectangle((3, 4.5), 2, 1, edgecolor='black', facecolor='#87CEEB', lw=2))
ax.text(4, 5, 'Projection Layer', ha='center', va='center', fontsize=10, color='black')

# Add state update block
ax.add_patch(patches.Rectangle((5.5, 4.5), 2, 1, edgecolor='black', facecolor='#32CD32', lw=2))
ax.text(6.5, 5, 'Hidden State Update', ha='center', va='center', fontsize=10, color='black')

# Add input and output arrows
ax.arrow(0.2, 5, 0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.text(0, 5, 'Input Frame', ha='right', va='center', fontsize=10)
ax.arrow(7.5, 5, 0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.text(8, 5, 'Processed Output', ha='left', va='center', fontsize=10)

# Add temporal dependency arrow
ax.arrow(4, 4.3, 0, -0.8, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.text(4, 3.2, 'Temporal Relations', ha='center', va='center', fontsize=10, color='black')

# Add a dashed line indicating the loop for recurrent updates
ax.add_patch(patches.FancyArrowPatch((6.5, 4.5), (6.5, 3.5),
                                     connectionstyle="arc3,rad=0.5",
                                     arrowstyle="->", color='black', lw=1.5, linestyle='dashed'))
ax.text(6.5, 3.2, 'Recurrent Update', ha='center', va='center', fontsize=10, color='black')

# Set axis limits and remove axes
ax.set_xlim(0, 9)
ax.set_ylim(3, 6)
ax.axis('off')

# Display the diagram
plt.title('Temporal-Aware Feature Extractor (TAFE)', fontsize=14, fontweight='bold')
plt.savefig("./test.png")
