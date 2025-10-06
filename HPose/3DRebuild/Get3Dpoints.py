import torch
import cv2
import numpy as np
import open3d as o3d

def main(image_path=r"C:\Users\Ruizhe\Desktop\Study\Code\AI\HPose\3DRebuild\pic\captured\20251006_200218.jpg"):
    try:
        # 1. 加载MiDaS模型
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device).eval()
        print(f"使用设备: {device}")

        # 2. 加载并预处理图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法加载图片: {image_path}")
        
        # 保存原始图像的尺寸和副本（关键：用于后续对齐）
        original_h, original_w = img.shape[:2]
        img_original = img.copy()  # 原始BGR图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB用于模型输入

        # 获取MiDaS的预处理函数
        transform = midas_transforms.dpt_transform
        input_batch = transform(img_rgb).to(device)  # 预处理后的图像（可能被缩放）

        # 3. 推理生成深度图
        with torch.no_grad():
            prediction = midas(input_batch)
            depth = prediction.squeeze().cpu().numpy()  # 模型输出的深度图（尺寸与预处理后图像一致）

        # 关键修复：将深度图缩放回原始图像尺寸（确保像素坐标对齐）
        depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # 归一化深度图到0-1范围
        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        # 反转深度值（近处小、远处大）
        depth = 1.3 - depth

        # 显示深度图和原始图片（此时已对齐）
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        # 为了更直观对比，将深度图与原图叠加显示
        # overlay = cv2.addWeighted(img_original, 0.5, depth_color, 0.5, 0)
        # cv2.imshow("Original Image", img_original)
        # cv2.imshow("Depth Color Map (Aligned)", depth_color)
        # cv2.imshow("Overlay (Aligned)", overlay)  # 叠加图用于验证对齐效果
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # 保存结果
        # cv2.imwrite("depth_color_aligned.png", depth_color)
        # cv2.imwrite("original_image.png", img_original)
        # cv2.imwrite("overlay_aligned.png", overlay)

        # 4. 转换为点云（坐标已对齐）
        h, w = depth.shape  # 此时h=original_h, w=original_w，与原图一致
        # 相机内参：根据原始图像尺寸调整（更合理的焦距设置）
        fx = fy = 0.8 * w  # 焦距与图像宽度成正比（经验值，可根据实际相机调整）
        cx, cy = w / 2, h / 2  # 主点在图像中心

        points = []
        colors = []
        depth_scaled = depth * 5.0  # 最大深度5米

        # 遍历每个像素（此时深度图与原图像素一一对应）
        for v in range(h):
            for u in range(w):
                z = depth_scaled[v, u]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                color = img[v, u, ::-1] / 255.0  # BGR→RGB
                points.append([x, y, z])
                colors.append(color)

        points = np.array(points)
        colors = np.array(colors)
        print(f"点云大小: {len(points)}个点")

        # 5. 可视化点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name="对齐后的点云",
            width=1024,
            height=768
        )

    except Exception as e:
        print(f"运行过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main(r"C:\Users\Ruizhe\Desktop\Study\Code\AI\HPose\3DRebuild\pic\captured\20251006_200218.jpg")
    