import cv2
import torch
import numpy as np
import os
import gsplat
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# --- 1. 环境与模型初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载MiDaS深度模型
try:
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
    print("MiDaS模型加载成功")
except Exception as e:
    print(f"MiDaS加载失败: {e}")
    exit()

# --- 2. 图像捕获与深度估计 ---
def capture_and_estimate_depth():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取画面")
            break
        cv2.imshow('按 "c" 捕捉，按 "q" 退出', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            frame = None
            break
        elif key == ord('c'):
            print("已捕捉图像，正在估计深度...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if frame is None:
        return None, None

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    depth_map = 1.0 - depth_map  # 反转深度（近大远小）
    return img_rgb, depth_map

# --- 3. 点云生成 ---
def create_point_cloud(rgb, depth, fov=60, sample_step=5):
    H, W, _ = rgb.shape
    focal_length = H / (2 * np.tan(np.deg2rad(fov / 2)))
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    
    # 下采样以减少点数量（避免计算量过大）
    jj = jj[::sample_step, ::sample_step].flatten()
    ii = ii[::sample_step, ::sample_step].flatten()
    depth = depth[::sample_step, ::sample_step].flatten()
    rgb_sampled = rgb[::sample_step, ::sample_step].reshape(-1, 3)

    cx, cy = W / 2, H / 2
    Z = depth
    X = (jj - cx) * Z / focal_length
    Y = (ii - cy) * Z / focal_length

    colors = rgb_sampled / 255.0
    points = np.stack([X, Y, -Z], axis=-1)  # 负号调整坐标系
    
    # 过滤无效点
    mask = Z > 0.1  # 去除过近的噪声点
    points = points[mask]
    colors = colors[mask]
    print(f"生成点云: {len(points)} 个点")
    return points, colors

# --- 4. 高斯参数初始化（关键：确保形状正确） ---
def init_gaussian_parameters(points, colors):
    N = len(points)
    if N == 0:
        raise ValueError("点云为空，无法初始化高斯参数")
    
    # 转换为torch张量并移动到设备
    means = torch.tensor(points, dtype=torch.float32, device=device)  # [N, 3]
    
    # 缩放因子：[N, 3]（每个维度单独缩放）
    scales = torch.ones_like(means) * 0.01
    depths = means[:, 2:3]  # 取z轴作为深度
    scales = scales / (1 + depths * 0.1)  # 远处的高斯更小
    
    # 旋转：[N, 4]（四元数，初始为单位旋转）
    rotations = torch.zeros(N, 4, device=device)
    rotations[:, 0] = 1.0  # 单位四元数 (w=1, x=0, y=0, z=0)
    
    # 颜色：[N, 3]
    colors = torch.tensor(colors, dtype=torch.float32, device=device)
    
    # 不透明度：[N, 1]
    opacities = torch.sigmoid(torch.ones(N, 1, device=device) * 1.5)  # 初始不透明度较高
    
    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities
    }

# --- 5. 使用gsplat渲染（核心：参数形状验证） ---
def render_gaussians(gaussian_params, image_size=(640, 480), yaw=0, pitch=0):
    # 创建相机
    camera = gsplat.PerspectiveCamera(
        fov=60.0,
        width=image_size[0],
        height=image_size[1],
        device=device
    )
    
    # 相机位姿变换（旋转）
    yaw_rad = torch.tensor(yaw * np.pi / 180, device=device)
    pitch_rad = torch.tensor(pitch * np.pi / 180, device=device)
    
    # 旋转矩阵（yaw绕y轴，pitch绕x轴）
    cos_y, sin_y = torch.cos(yaw_rad), torch.sin(yaw_rad)
    cos_p, sin_p = torch.cos(pitch_rad), torch.sin(pitch_rad)
    
    rot_y = torch.tensor([[cos_y, 0, sin_y, 0],
                         [0, 1, 0, 0],
                         [-sin_y, 0, cos_y, 0],
                         [0, 0, 0, 1]], device=device, dtype=torch.float32)
    
    rot_x = torch.tensor([[1, 0, 0, 0],
                         [0, cos_p, -sin_p, 0],
                         [0, sin_p, cos_p, 0],
                         [0, 0, 0, 1]], device=device, dtype=torch.float32)
    
    c2w = rot_y @ rot_x @ torch.eye(4, device=device)
    c2w[2, 3] = -3.0  # 相机沿z轴后移，远离物体
    camera.c2w = c2w
    
    # 渲染（关键：确保所有参数形状正确）
    image, alpha = gsplat.render(
        means=gaussian_params["means"],       # [N, 3]
        colors=gaussian_params["colors"],     # [N, 3]
        scales=gaussian_params["scales"],     # [N, 3]
        rotations=gaussian_params["rotations"], # [N, 4]
        opacities=gaussian_params["opacities"].squeeze(),  # [N]
        camera=camera
    )
    return image.cpu().numpy()

# --- 6. 保存为PLY文件 ---
def save_to_ply(path, gaussian_params):
    params = {k: v.cpu().numpy() for k, v in gaussian_params.items()}
    N = len(params["means"])
    
    # 构造PLY格式的结构化数组
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # 法线（占位）
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')] + \
        [(f'f_rest_{i}', 'f4') for i in range(15*3)] + \
        [('opacity', 'f4'),
         ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
         ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    elements = np.empty(N, dtype=dtype)
    # 填充数据（按PLY格式要求的顺序）
    elements['x'], elements['y'], elements['z'] = params["means"].T
    elements['nx'], elements['ny'], elements['nz'] = np.zeros(3)  # 法线占位
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = (params["colors"] - 0.5) / 0.28209479177  # SH系数转换
    for i in range(15*3):
        elements[f'f_rest_{i}'] = 0  # 高阶SH系数为0
    elements['opacity'] = params["opacities"].flatten()
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = np.log(params["scales"]).T  # 尺度保存在log空间
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = params["rotations"].T
    
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    print(f"已保存到 {path}")

# --- 7. 主函数 ---
if __name__ == "__main__":
    # 1. 捕获图像和深度
    rgb, depth = capture_and_estimate_depth()
    if rgb is None or depth is None:
        print("程序退出")
        exit()
    
    # 2. 创建点云（采样步长5，平衡速度和质量）
    points, colors = create_point_cloud(rgb, depth, sample_step=5)
    if len(points) == 0:
        print("点云为空，退出")
        exit()
    
    # 3. 初始化高斯参数（确保形状正确）
    gaussian_params = init_gaussian_parameters(points, colors)
    
    # 4. 渲染并显示
    rendered = render_gaussians(gaussian_params, image_size=rgb.shape[:2])
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(rgb), plt.title("原始图像")
    plt.subplot(122), plt.imshow(rendered), plt.title("高斯喷溅效果")
    plt.tight_layout(), plt.show()
    
    # 5. 保存为PLY文件
    save_to_ply("output.ply", gaussian_params)
    print("推荐使用在线查看器：https://antimatter15.com/splat/")
    