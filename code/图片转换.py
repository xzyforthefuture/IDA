# from PIL import Image
# image_path = "D:/case_study.png"
# image = Image.open(image_path)
# image = image.convert("RGB")
# output_path = "D:/case_study.eps"
# image.save(output_path, format="EPS")


import torch

# 创建一个示例张量
i_i_graph = torch.tensor([[1, 0, 3], [0, 5, 0], [7, 0, 9]])
positem = 1

# 执行代码
zero_indices = torch.nonzero(i_i_graph[positem] == 0).squeeze()

print("i_i_graph:", i_i_graph)
print("positem:", positem)
print("zero_indices:", zero_indices)