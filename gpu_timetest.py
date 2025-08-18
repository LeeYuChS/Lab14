# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.profiler import profile, record_function, ProfilerActivity
# import utils

# # 检查是否有可用的 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # 定义一个简单的神经网络
# # class SimpleModel(nn.Module):
# #     def __init__(self):
# #         super(SimpleModel, self).__init__()
# #         self.fc1 = nn.Linear(100, 50)
# #         self.fc2 = nn.Linear(50, 10)
    
# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# vit_base_patch16 = "vit_base_patch16_224"
# model = utils.create_model(vit_base_patch16, 2).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # 实例化模型、损失函数和优化器
# # model = SimpleModel().to(device)
# # criterion = nn.MSELoss().to(device)
# # optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 创建输入数据
# inputs = torch.randn(16, 3, 224, 224).to(device)
# targets = torch.randint(low=0, high=2, size=(16,)).to(device)

# # 使用 PyTorch Profiler 分析 GPU 上的算子
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         # 前向传播
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
        
#         # 后向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# # 打印性能分析结果
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# # 也可以保存成文件以供进一步分析
# prof.export_chrome_trace("trace.json")



import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    print(f"Compute Capability: {cc[0]}.{cc[1]}")  # 輸出格式如 8.9
else:
    print("No CUDA device found.")