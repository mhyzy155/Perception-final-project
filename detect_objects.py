#%%
import cv2
import numpy as np
from pathlib import Path,PurePath
import matplotlib.pyplot as plt
import open3d as o3d
# %%
base_path = Path("./without_occlusions")
left = base_path / 'left' 
right = base_path / 'right'
# %%
test_images = ["1585434279_805531979_Left.png", "1585434282_858812094_Left.png", "1585434284_783691883_Left.png"]


# %%
map1_l = np.load("map1_l_cropped.npy")
map1_r = np.load("map1_r_cropped.npy")
map2_l = np.load("map2_l_cropped.npy")
map2_r = np.load("map2_r_cropped.npy")
new_cam_l = np.load("newcameramtx_l.npy")
new_cam_r = np.load("newcameramtx_r.npy")

# %%
def calibrateImages(img_l, img_r):
   dst_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_NEAREST)
   dst_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_NEAREST)
   return dst_l, dst_r

# %% Create pointcloud
def createPointCloud(rgb_img, depth):
   #print(depth.astype('float32'))
   depth = o3d.geometry.Image(depth.astype('float32'))
   rgb = o3d.geometry.Image(rgb_img.astype('int8'))
   rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
      rgb, depth,
      depth_scale=1000
   )
   pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd,
    o3d.camera.PinholeCameraIntrinsic(
        rgb_img.shape[1],
        rgb_img.shape[0],
        new_cam_l[0][0],
        new_cam_l[1][1],
        new_cam_l[0][2],
        new_cam_l[1][2]
        ))
    #o3d.camera.PinholeCameraIntrinsic(
    #   o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    #))
   # Flip it, otherwise the pointcloud will be upside down
   pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
   return pcd
# %%
def draw_labels_on_model(pcl,labels):
    cmap = plt.get_cmap("tab20")
    pcl_temp = pcl
    max_label = labels.max()
    print("scan has %d clusters" % (max_label + 1))
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcl_temp.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcl_temp])
# %% Depth camera setup
min_disp = 70
num_disp = 10 * 16
block_size = 57
stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(200)
stereo.setUniquenessRatio(10)
stereo.setSpeckleRange(3)
stereo.setSpeckleWindowSize(3)
# %%
#images_l = sorted(left.glob('*_Left.png'))

static_cloud = None
max_dist = 5e-8
cluster_density = 2e-7
cluster_minpoints = 1000


for filename in test_images:
   filename_left = left / filename
   #print(PurePath(filename).stem)
   filename_stem = PurePath(filename_left).stem
   filename_parts = filename_stem.split('_')
   file_number = filename_parts[0] + '_' + filename_parts[1]
   filename_right = right / (file_number + '_Right.png')
   #print(file_number)
   #print(filename_left)
   #print(filename_right)
   img_l = cv2.imread(str(filename_left))
   img_r = cv2.imread(str(filename_right))

   dst_l, dst_r = calibrateImages(img_l, img_r)
   dst_l_blur = cv2.GaussianBlur(dst_l, (7,7), 0)
   dst_r_blur = cv2.GaussianBlur(dst_r, (7,7), 0)
   gray_left = cv2.cvtColor(dst_l_blur, cv2.COLOR_BGR2GRAY)
   gray_right = cv2.cvtColor(dst_r_blur, cv2.COLOR_BGR2GRAY)

   disp = stereo.compute(gray_left, gray_right).astype('float')
   #disp_median = (cv2.medianBlur((disp*5).astype('uint8'), 7).astype('float32'))/5
   depth = 1/(disp)

   depth[depth == np.max(depth)] = np.nan

   pcd = createPointCloud(dst_l, depth)
   if static_cloud is None:
      static_cloud = pcd
      
   else:
      distances = np.asarray(pcd.compute_point_cloud_distance(static_cloud))
      pcd = pcd.select_by_index(np.where(distances > max_dist)[0])
      labels = np.asarray(pcd.cluster_dbscan(eps=cluster_density, min_points=cluster_minpoints))
      print(labels, np.max(labels)+1)
      pcd = pcd.select_by_index(np.where(labels == 0)[0])
      #draw_labels_on_model(pcd, labels)

   cv2.imshow('left' , dst_l)
   cv2.imshow('right', dst_r)
   cv2.imshow('disp' , disp/np.max(disp))
   #cv2.imshow('depth' , depth/np.max(depth))
   key = cv2.waitKey(16)
   if key == 27:
      break
   o3d.visualization.draw_geometries([pcd])
cv2.destroyAllWindows()
# %%

# %%
