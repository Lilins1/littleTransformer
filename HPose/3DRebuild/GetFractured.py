import torch, cv2, numpy as np, open3d as o3d
from typing import Tuple, Optional

# --- 导入 LaMa Cleaner 库 ---
# 注意：你需要先安装 lama-cleaner: pip install lama-cleaner
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config

# ---------------------------
# 全局 LaMa 模型初始化
# ---------------------------
LAMA_MANAGER = None
try:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化 ModelManager，指定使用 lama 模型
    # LAMA_MANAGER 将自动下载并加载预训练权重
    LAMA_MANAGER = ModelManager(name='lama', device=DEVICE)
    print(f"LaMa model (via lama-cleaner) initialized on {DEVICE}.")
    
except Exception as e:
    print(f"ERROR: Could not initialize LaMa model. Please install 'lama-cleaner' and check internet connection for weights. Error: {e}")

def lama_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    使用全局的 lama-cleaner ModelManager 修复图像。
    """
    if LAMA_MANAGER is None:
        return None
    
    try:
        # 1. 转换颜色空间 (BGR -> RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. 转换蒙版格式: (1/0) -> (255/0) uint8
        # lama-cleaner 期望蒙版中缺失区域为 255
        cleaner_mask = (mask * 255).astype(np.uint8) 

        inpaint_config = Config(
            # 必填项 (保持或微调)
            ldm_steps=10, 
            hd_strategy="Resize", 
            hd_strategy_crop_margin=128,
            hd_strategy_crop_trigger_size=768, # 提高裁剪阈值
            hd_strategy_resize_limit=768,     # 提高最大处理尺寸

            # 提高 SD 参数以获取高质量输出
            prompt="A detailed and realistic scene", # 增加提示以指导修复
            sd_steps=50,                             # 提高到 50 甚至 75
            sd_guidance_scale=7.5,                   # 保持或微调上下文相关性
            sd_sampler="uni_pc"                      # 使用快速采样器
        )
        
        # 3. ✨ 核心调用：直接调用 LAMA_MANAGER 实例 (调用 __call__)
        # 传入 image, mask 和 config 三个参数
        # 注意：ModelManager 内部的 __call__ 会处理 NumPy 到 Tensor 的转换
        inpainted_rgb = LAMA_MANAGER(
            image_rgb,               # image (NumPy RGB)
            cleaner_mask,            # mask (NumPy uint8 255/0)
            inpaint_config           # config (Config object)
        )
        inpainted_rgb_uint8 = np.clip(inpainted_rgb, 0, 255).astype(np.uint8)
        # 4. 转换颜色空间 (RGB -> BGR)
        # inpainted_bgr = cv2.cvtColor(inpainted_rgb_uint8, cv2.COLOR_RGB2BGR)
        return inpainted_rgb_uint8
        
    except Exception as e:
        # 捕捉修复过程中的错误，例如尺寸不兼容等
        print(f"Error during LaMa inpainting: {e}")
        return None


# ---------------------------
# 点云处理类 (保持不变)
# ---------------------------
class PointCloudProcessor:
    """
    负责从图像生成深度点云、应用偏移、重投影回图像，并生成缺失蒙版。
    """
    def __init__(self, model_type: str = "DPT_Large"):
        """
        初始化 MiDaS 深度估计模型。
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MiDaS model: {model_type} on {self.device}")
        
        # 加载 MiDaS 模型和变换
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = self.transforms.dpt_transform
            self.midas.to(self.device).eval()
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            raise

        self.max_depth = 3
        self.step = 1 # 点云采样步长

    def _estimate_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            depth = self.midas(input_batch).squeeze().cpu().numpy()
        # print("深度原始最小值:", depth.min())
        # print("深度原始最大值:", depth.max())

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.model_type == "DPT_Large":
            depth = depth / 42
        if self.model_type == "DPT_Hybrid":
            depth = depth / 3200
        
        depth_normalized = 1 - depth
        depth_scaled = np.clip(depth_normalized * self.max_depth, 0, self.max_depth)
        
        return depth_scaled

    def _generate_pointcloud(self, frame_bgr: np.ndarray, depth_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        h, w = frame_bgr.shape[:2]
        fx = fy = 0.8 * w
        cx, cy = w / 2, h / 2

        i, j = np.meshgrid(np.arange(0, w, self.step), np.arange(0, h, self.step))
        z = depth_scaled[::self.step, ::self.step]
        
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        y = -y # 修正 Y 轴

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = frame_bgr[::self.step, ::self.step, ::-1].reshape(-1, 3) / 255.0
        
        return points, colors, fx, fy, cx, cy

    @staticmethod
    def _pointcloud_to_image(points: np.ndarray, colors: np.ndarray, w: int, h: int, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.round(points[:, 0] * fx / points[:, 2] + cx).astype(np.int32)
        v = np.round(points[:, 1] * fy / points[:, 2] + cy).astype(np.int32)

        v_flipped = (h - 1) - v 
        
        mask_valid = (u >= 0) & (u < w) & (v_flipped >= 0) & (v_flipped < h)
        u_valid, v_valid = u[mask_valid], v_flipped[mask_valid]
        points_valid = points[mask_valid]
        colors_valid = colors[mask_valid]

        depth_img = np.full((h, w), np.inf, dtype=np.float32)
        color_img = np.zeros((h, w, 3), dtype=np.float32)
        mask_img = np.ones((h, w), dtype=np.uint8) 

        for i in range(len(u_valid)):
            x_pix, y_pix, z = u_valid[i], v_valid[i], points_valid[i, 2]
            
            if z < depth_img[y_pix, x_pix]:
                depth_img[y_pix, x_pix] = z
                color_img[y_pix, x_pix] = colors_valid[i]
                mask_img[y_pix, x_pix] = 0 

        depth_img[depth_img == np.inf] = 0
        
        return depth_img, color_img, mask_img

    def process_and_reproject(self, frame_bgr: np.ndarray, translation_x: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]

        # 1. 估计深度
        depth_scaled = self._estimate_depth(frame_bgr)
        
        # 2. 生成原始点云和内参
        points, colors, fx, fy, cx, cy = self._generate_pointcloud(frame_bgr, depth_scaled)

        # 3. 应用 X 轴偏移
        points_translated = points.copy()
        points_translated[:, 0] += translation_x

        # 4. 重投影回图像并生成蒙版
        depth_img_reprojected, color_img_rgb_reprojected, mask_img_missing = self._pointcloud_to_image(
            points_translated, colors, w, h, fx, fy, cx, cy
        )

        # 5. RGB -> BGR (OpenCV 格式)
        color_img_bgr_reprojected = (color_img_rgb_reprojected * 255).astype(np.uint8)
        color_img_bgr_reprojected = cv2.cvtColor(color_img_bgr_reprojected, cv2.COLOR_RGB2BGR)

        return color_img_bgr_reprojected, mask_img_missing, depth_img_reprojected

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    
    if LAMA_MANAGER is None:
        print("\n*** LaMa 修复功能未启用。请检查安装和网络连接。***")

    processor = PointCloudProcessor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
        
    print("\n实时点云重投影和 LaMa 修复处理中，按 'q' 退出...")
    
    TRANSLATION_X = 0.15

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]

        # 1. 运行点云处理流程：获取重投影图像和缺失蒙版
        reprojected_bgr, missing_mask, reprojected_depth = processor.process_and_reproject(
            frame, TRANSLATION_X
        )

        # 2. 执行 LaMa 修复
        inpainted_image = lama_inpaint(reprojected_bgr, missing_mask)

        # --------------------
        # 可视化
        # --------------------
        
        # 原始图像 (BGR)
        original_vis = frame
        
        # 修复结果或原始重投影
        if inpainted_image is not None:
            # 修复成功，展示修复结果
            result_vis = inpainted_image
            result_label = "Inpainted Result"
        else:
            # 修复失败，展示带空洞的重投影图像
            result_vis = reprojected_bgr
            result_label = "Reprojected (Holes)"

        # 蒙版可视化 (白色区域为缺失/空洞)
        mask_vis_color = cv2.cvtColor(missing_mask * 255, cv2.COLOR_GRAY2BGR)

        # 拼接显示
        display_img = np.hstack((
            original_vis,
            
            result_vis,      # 展示修复结果
            reprojected_bgr, # 保持展示带空洞的重投影结果
            mask_vis_color
        ))
        
        # 确保 display_img 尺寸足够大以便添加文字
        if display_img.shape[0] < h + 30:
            display_img = cv2.copyMakeBorder(display_img, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        cv2.putText(display_img, f"Original |{result_label} | Reprojected (Holes) | Missing Mask", (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Real-time View Synthesis and Inpainting", display_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()