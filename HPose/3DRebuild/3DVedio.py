import torch, cv2, numpy as np, open3d as o3d, time
from threading import Thread, Lock

# ---------------------------
# 全局变量
# ---------------------------
current_pcd = o3d.geometry.PointCloud()
exit_flag = False
pcd_lock = Lock()
update_event = False

# ---------------------------
# 可视化线程
# ---------------------------
def pointcloud_visualizer():
    global current_pcd, exit_flag, pcd_lock, update_event
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="实时点云", width=960, height=720)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="平行点云", width=960, height=720)

    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coord_frame.translate([-5, -5, 0])
    vis.add_geometry(coord_frame)
    vis2.add_geometry(coord_frame)

    # 初始化点云
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    pcd2 = o3d.geometry.PointCloud()
    vis2.add_geometry(pcd2)

    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(1)

    ctr2 = vis2.get_view_control()
    ctr2.set_front([0, 0, -1])
    ctr2.set_up([0, 1, 0])
    ctr2.set_lookat([0, 0, 0])
    ctr2.set_zoom(1)

    time.sleep(0.5)

    while not exit_flag:
        with pcd_lock:
            if update_event:
                pcd.points = o3d.utility.Vector3dVector(np.asarray(current_pcd.points))
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(current_pcd.colors))
                vis.update_geometry(pcd)

                # 平移点云
                points_translated = np.asarray(current_pcd.points).copy()
                points_translated[:, 0] += 0.15
                pcd2.points = o3d.utility.Vector3dVector(points_translated)
                pcd2.colors = o3d.utility.Vector3dVector(np.asarray(current_pcd.colors))
                vis2.update_geometry(pcd2)

                update_event = False

        vis.poll_events()
        vis.update_renderer()
        vis2.poll_events()
        vis2.update_renderer()
        time.sleep(0.03)

    vis.destroy_window()
    vis2.destroy_window()

# ---------------------------
# 点云投影回图像
# ---------------------------
def pointcloud_to_image(points, colors, w, h, fx, fy, cx, cy):
    u = np.round(points[:, 0] * fx / points[:, 2] + cx).astype(np.int32)
    v = np.round(points[:, 1] * fy / points[:, 2] + cy).astype(np.int32)

    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[mask], v[mask]
    v = (h - 1) - v # <--- 在这里添加反转
    points_valid = points[mask]
    colors_valid = colors[mask]

    depth_img = np.full((h, w), np.inf, dtype=np.float32)
    color_img = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(len(u)):
        x, y, z = u[i], v[i], points_valid[i, 2]
        if z < depth_img[y, x]:
            depth_img[y, x] = z
            color_img[y, x] = colors_valid[i]

    depth_img[depth_img == np.inf] = 0
    return depth_img, color_img

# ---------------------------
# 相机帧处理线程
# ---------------------------
def process_frames(camera_id=0):
    global current_pcd, exit_flag, pcd_lock, update_event

    # 加载 MiDaS 模型
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    vis_thread = Thread(target=pointcloud_visualizer, daemon=True)
    vis_thread.start()

    step = 1
    max_depth = 1.5

    print("实时深度点云处理中，按 'q' 退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            depth = midas(input_batch).squeeze().cpu().numpy()

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = depth / 3200
        depth_normalized = 1 - depth
        depth_scaled = np.clip(depth_normalized * max_depth, 0, max_depth)

        fx = fy = 0.8 * w
        cx, cy = w / 2, h / 2

        i, j = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        z = depth_scaled[::step, ::step]
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        y = -y

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = frame[::step, ::step, ::-1].reshape(-1, 3) / 255.0

        # 平移点云
        translation_x = 0.15
        points_translated = points.copy()
        points_translated[:, 0] += translation_x
        colors_translated = colors.copy()

        # 投影回图像
        depth_img, color_img = pointcloud_to_image(points, colors, w, h, fx, fy, cx, cy)
        depth_img_trans, color_img_trans = pointcloud_to_image(points_translated, colors_translated, w, h, fx, fy, cx, cy)

        # 可视化 OpenCV
        depth_vis = (depth_img / (depth_img.max() + 1e-6) * 255).astype(np.uint8)
        depth_vis_trans = (depth_img_trans / (depth_img_trans.max() + 1e-6) * 255).astype(np.uint8)
        color_vis_rgb = (color_img * 255).astype(np.uint8)
        color_vis_trans_rgb = (color_img_trans * 255).astype(np.uint8)

        # ✨ 颜色通道修正：将 RGB 转换为 BGR 以便 cv2.imshow 正确显示
        color_vis = cv2.cvtColor(color_vis_rgb, cv2.COLOR_RGB2BGR) 
        color_vis_trans = cv2.cvtColor(color_vis_trans_rgb, cv2.COLOR_RGB2BGR)

        # 转换灰度深度图为 3 通道
        depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        depth_vis_trans_color = cv2.cvtColor(depth_vis_trans, cv2.COLOR_GRAY2BGR)

        # 拼接显示
        cv2.imshow(
            "全部显示: RGB | 深度 | 原始点云 | 平移点云",
            np.hstack((frame, depth_vis_color, color_vis, color_vis_trans))
        )

        # 更新全局点云
        with pcd_lock:
            current_pcd.points = o3d.utility.Vector3dVector(points)
            current_pcd.colors = o3d.utility.Vector3dVector(colors)
            update_event = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
    vis_thread.join()

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    process_frames()
