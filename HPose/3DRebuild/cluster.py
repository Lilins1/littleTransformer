import torch, cv2, numpy as np, open3d as o3d, time
from threading import Thread, Lock
from sklearn.cluster import DBSCAN,KMeans
import numpy as np

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
    vis.create_window(window_name="实时点云", width=960, height=720)\
    # 平移点云窗口
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="平行点云", width=960, height=720)

    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coord_frame.translate([-5, -5, 0])  # 平移到左下角或远离点云的位置
    vis.add_geometry(coord_frame)
    vis2.add_geometry(coord_frame)
        

    # 初始化点云
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    pcd2 = o3d.geometry.PointCloud()
    vis2.add_geometry(pcd2)

    # 设置相机视角（相机原点 + Z轴前方 + Y轴向上）
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])   # 相机朝向Z轴（蓝色）
    ctr.set_up([0, 1, 0])      # Y轴向上（绿色）
    ctr.set_lookat([0, 0, 0])  # 视点中心，通常放在点云前方
    ctr.set_zoom(1)          # 缩放

    ctr2 = vis2.get_view_control()
    ctr2.set_front([0, 0, -1])
    ctr2.set_up([0, 1, 0])
    ctr2.set_lookat([0, 0, 0])
    ctr2.set_zoom(1)

    time.sleep(0.5)  # 等待窗口初始化

    while not exit_flag:
        with pcd_lock:
            if update_event:
                pcd.points = o3d.utility.Vector3dVector(np.asarray(current_pcd.points))
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(current_pcd.colors))
                vis.update_geometry(pcd)
                

                # 平移点云
                points_translated = np.asarray(current_pcd.points).copy()
                points_translated[:, 0] += 0.15  # X轴平移
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
def cluster_kmeans(depth_scaled, n_clusters=10, n_levels=10):
    H, W = depth_scaled.shape
    fx, fy = 525.0, 525.0  # 相机焦距
    cx, cy = W / 2, H / 2   # 光心

    # 生成网格坐标
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # 转为三维点云
    x = (uu - cx) * depth_scaled / fx
    y = (vv - cy) * depth_scaled / fy
    z = depth_scaled
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # N x 3

    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(points)

    clustered_depth = np.zeros_like(z.flatten())

    for l in range(n_clusters):
        mask = labels == l
        avg_depth = points[mask, 2].mean()  # 每类平均深度
        clustered_depth[mask] = avg_depth

    # 对聚类后的深度归一化到 [0,1] 并分层
    clustered_depth = clustered_depth.reshape(H, W)
    clustered_depth -= clustered_depth.min()
    if clustered_depth.max() > 0:
        clustered_depth /= clustered_depth.max()

    # 分成 n_levels 层
    clustered_depth = np.floor(clustered_depth * n_levels) / (n_levels - 1)

    return clustered_depth
def cluster(depth_scaled, eps=0.05, min_samples=10, n_levels=4):
    H, W = depth_scaled.shape
    fx, fy = 525.0, 525.0  # 相机焦距
    cx, cy = W / 2, H / 2   # 光心

    # 生成网格坐标
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # 转为三维点云
    x = (uu - cx) * depth_scaled / fx
    y = (vv - cy) * depth_scaled / fy
    z = depth_scaled
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # N x 3

    # DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    
    clustered_depth = np.zeros_like(z.flatten())
    
    unique_labels = [l for l in np.unique(labels) if l != -1]  # 排除噪声
    for l in unique_labels:
        mask = labels == l
        avg_depth = points[mask, 2].mean()  # 取每类的平均深度
        clustered_depth[mask] = avg_depth

    # 对聚类后的深度归一化到 [0,1] 并分层
    clustered_depth = clustered_depth.reshape(H, W)
    clustered_depth -= clustered_depth.min()
    if clustered_depth.max() > 0:
        clustered_depth /= clustered_depth.max()
    
    # 分成 n_levels 层
    clustered_depth = np.floor(clustered_depth * n_levels) / (n_levels - 1)

    return clustered_depth

# ---------------------------
# 相机帧处理线程
# ---------------------------
def process_frames(camera_id=0):
    global current_pcd, exit_flag, pcd_lock, update_event

    # 加载 MiDaS 模型（轻量 DPT_Hybrid 适合实时）
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)

    # 启动可视化线程
    vis_thread = Thread(target=pointcloud_visualizer, daemon=True)
    vis_thread.start()

    step = 1  # 点云采样步长
    max_depth = 1.5  # 最大深度（米）

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
        # print("深度原始最小值:", depth.min())
        # print("深度原始最大值:", depth.max())

        # 缩放回原图尺寸
        depth = cv2.resize(depth, (w, h))
        # depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        depth = depth/3200
        depth_normalized = 1 - depth  # 反转深度（近大远小）
        # depth = 1.5 - depth  # 反转深度，使近处大
        # depth_scaled = depth * max_depth  # 近似真实深度（米）
        depth_min, depth_max = depth.min(), depth.max()
        # depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_scaled = depth_normalized * max_depth

        depth_scaled = np.clip(depth_scaled, 0, max_depth)
        depth_scaled = cluster_kmeans(depth_scaled)


        # ---------------------------
        # 点云计算（相机原点，Z轴前方，Y轴向上）
        # ---------------------------
        fx = fy = 0.8 * w
        cx, cy = w / 2, h / 2

        i, j = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        z = depth_scaled[::step, ::step]
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        y = -y  # Y轴向上

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = frame[::step, ::step, ::-1].reshape(-1, 3) / 255.0
        # 平移量，沿相机 X 轴
        translation_x = 0.15  # 单位：米
        # 平行点云，沿 X 轴平移 15cm
        points_translated = points.copy()
        points_translated[:, 0] += translation_x  # X轴平移

        # 颜色可以保持不变
        colors_translated = colors.copy()

        # 更新全局点云
        with pcd_lock:
            current_pcd.points = o3d.utility.Vector3dVector(points)
            current_pcd.colors = o3d.utility.Vector3dVector(colors)

            update_event = True

        # 显示 RGB + 深度伪彩图
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_scaled * 255), cv2.COLORMAP_JET)
        cv2.imshow("RGB | 深度", np.hstack((frame, depth_color)))

        time.sleep(1)  # 控制处理速度

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
