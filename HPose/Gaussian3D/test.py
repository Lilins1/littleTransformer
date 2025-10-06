import math
import torch
import numpy as np
from PIL import Image

# 尝试导入 gsplat 库
try:
    from gsplat import rasterization
except ImportError:
    print("错误：无法导入 gsplat 库。")
    print("请确保您已经通过 'pip install gsplat' 安装了GPU版本（需要CUDA支持）。")
    exit()

# 检查CUDA是否可用
if not torch.cuda.is_available():
    print("错误：未检测到可用的CUDA设备。")
    print("请确保您的系统已安装NVIDIA GPU和对应的CUDA驱动。")
    exit()

# ------------------------------
# 1. 配置渲染参数
# ------------------------------
device = torch.device("cuda")
H, W = 512, 512
num_points = 10000
output_path = "gsplat_gpu_output.png"
print(f"配置完成：将在 {torch.cuda.get_device_name(device)} 上渲染一个 {H}x{W} 的图像，包含 {num_points} 个高斯点。")

# ------------------------------
# 2. 随机生成高斯参数
# ------------------------------
means = (torch.rand(num_points, 3, device=device) - 0.5) * 2.0
scales = torch.rand(num_points, 3, device=device) * 0.1 + 0.02
rgbs = torch.rand(num_points, 3, device=device)

# 修正：opacities 改为一维张量 (num_points,)，去掉多余的维度
opacities = torch.ones(num_points, device=device)  # 原来的代码是 (num_points, 1)

# 生成四元数（旋转参数）
u = torch.rand(num_points, 1, device=device)
v = torch.rand(num_points, 1, device=device)
w = torch.rand(num_points, 1, device=device)
quats = torch.cat(
    [
        torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
    ],
    dim=-1,
)
quats = quats / torch.norm(quats, dim=-1, keepdim=True)
print("已成功生成随机高斯参数。")

# ------------------------------
# 3. 设置相机参数
# ------------------------------
fov_x = math.pi / 2.0
focal = 0.5 * float(W) / math.tan(0.5 * fov_x)

viewmat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
    device=device,
)

K = torch.tensor(
    [
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ],
    dtype=torch.float32,
    device=device,
)
print("相机参数已设置。")

# ------------------------------
# 4. 调用 gsplat 进行渲染
# ------------------------------
print("开始GPU渲染...")
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

renders, _, _ = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,  # 现在形状为 (10000,)，符合要求
    colors=rgbs,
    viewmats=viewmat.unsqueeze(0),
    Ks=K.unsqueeze(0),
    width=W,
    height=H,
)

end_time.record()
torch.cuda.synchronize()
render_time = start_time.elapsed_time(end_time) / 1000.0
print(f"GPU渲染完成，耗时: {render_time:.4f} 秒")

# ------------------------------
# 5. 保存渲染结果
# ------------------------------
img_tensor = renders[0, ..., :3].cpu()
img_np = img_tensor.clamp(0, 1).numpy()
img_uint8 = (img_np * 255).astype(np.uint8)
Image.fromarray(img_uint8).save(output_path)
print(f"渲染图像已成功保存到: {output_path}")
    