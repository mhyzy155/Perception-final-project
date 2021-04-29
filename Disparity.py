# %%
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
test_image = "1585434279_805531979"

# %%
map1_l = np.load("map1_l.npy")
map1_r = np.load("map1_r.npy")
map2_l = np.load("map2_l.npy")
map2_r = np.load("map2_r.npy")
new_cam_l = np.load("newcameramtx_l.npy")
new_cam_r = np.load("newcameramtx_r.npy")

# %%
def calibrateImages(img_l, img_r):
    dst_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_NEAREST)
    dst_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_NEAREST)
    return dst_l, dst_r

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
   #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
   return pcd

def draw_labels_on_model(pcl,labels):
    cmap = plt.get_cmap("tab20")
    pcl_temp = pcl
    max_label = labels.max()
    print("scan has %d clusters" % (max_label + 1))
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcl_temp.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcl_temp])

def getObjectCenter(pcd):
   points = np.asarray(pcd.points)
   center = np.mean(points, axis=0)
   return center

def projectPointToImage(point):
   fx = new_cam_l[0][0]
   fy = new_cam_l[1][1]
   cx = new_cam_l[0][2]
   cy = new_cam_l[1][2]
   u = cx + fx*point[0]/point[2]
   v = cy + fy*point[1]/point[2]
   return np.asarray([u,v])

# %%
def biFilter(img_l, img_r, k_size, sigma):
    return cv2.bilateralFilter(img_l,k_size,sigma,sigma), cv2.bilateralFilter(img_r,k_size,sigma,sigma)

def gaussFilter(img_l, img_r, k_size):
    return cv2.GaussianBlur(img_l,(k_size,k_size),0), cv2.GaussianBlur(img_r,(k_size,k_size),0)

# %%
min_disp = 70
num_disp = 10 * 16
block_size = 31
stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(200)
stereo.setUniquenessRatio(10)
stereo.setSpeckleRange(3)
stereo.setSpeckleWindowSize(3)

images_l = sorted(left.glob('*_Left.png'))
for filename_left in images_l:
    filename_stem = PurePath(filename_left).stem
    filename_parts = filename_stem.split('_')
    file_number = filename_parts[0] + '_' + filename_parts[1]
    filename_right = right / (file_number + '_Right.png')
    img_l = cv2.imread(str(filename_left))
    img_r = cv2.imread(str(filename_right))

    dst_l, dst_r = calibrateImages(img_l, img_r)
    
    #dst_l, dst_r = biFilter(dst_l, dst_r, 17, 35)
    #dst_l, dst_r = gaussFilter(dst_l, dst_r, 17)

    gray_l = cv2.cvtColor(dst_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(dst_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gray_l, gray_r).astype('float')
    
    #disp1 = stereo.compute(dst_l[:,:,0], dst_r[:,:,0]).astype('float')
    #disp2 = stereo.compute(dst_l[:,:,1], dst_r[:,:,1]).astype('float')
    #disp3 = stereo.compute(dst_l[:,:,2], dst_r[:,:,2]).astype('float')
    #disp_m = cv2.merge((disp1/np.max(disp1), disp2/np.max(disp2), disp3/np.max(disp3)))
    #disp_m = np.median([disp1, disp2, disp3], axis=0)
        
    #ratio = 20
    #disp = (cv2.medianBlur((disp/ratio).astype('uint8'), 11).astype('float32'))*ratio

    #depth = 1/(disp)*100+50
    #pcd = createPointCloud(dst_l, depth)
    #o3d.visualization.draw_geometries([pcd])

    cv2.imshow('left' , dst_l)
    #cv2.imshow('right', dst_r)
    cv2.imshow('disp' , disp/4096.0)
    
    #cv2.imshow('disp_m' , disp_m/np.max(disp_m))
    #cv2.imshow('disp1' , disp1/np.max(disp1))
    #cv2.imshow('disp2' , disp2/np.max(disp2))
    #cv2.imshow('disp3' , disp3/np.max(disp3))
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()

# %%
max_dist = 5e-8
#cluster_density = 2*2e-7
#cluster_minpoints = 500

min_disp = 70
num_disp = 20 * 16
block_size = 15
stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(10)
stereo.setUniquenessRatio(25)
stereo.setSpeckleRange(3)
stereo.setSpeckleWindowSize(3)

margin_g = 40
margin_d = 10
contour_thresh = 500
images_l = sorted(left.glob('*_Left.png'))
filename_stem = PurePath(images_l[0]).stem
filename_parts = filename_stem.split('_')
file_number = filename_parts[0] + '_' + filename_parts[1]
filename_right = right / (file_number + '_Right.png')
image_prev_l, image_prev_r = calibrateImages(cv2.imread(str(images_l[0])), cv2.imread(str(filename_right)))

gray_l = cv2.cvtColor(image_prev_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(image_prev_r, cv2.COLOR_BGR2GRAY)
disp = stereo.compute(gray_l, gray_r).astype('float')
depth = 1/(disp)
depth[depth == np.max(depth)] = np.nan
static_cloud = createPointCloud(image_prev_l, depth)
#static_cloud = static_cloud.uniform_down_sample(2)

for filename_left in images_l:
    filename_stem = PurePath(filename_left).stem
    filename_parts = filename_stem.split('_')
    file_number = filename_parts[0] + '_' + filename_parts[1]
    filename_right = right / (file_number + '_Right.png')
    img_l = cv2.imread(str(filename_left))
    img_r = cv2.imread(str(filename_right))

    dst_l, dst_r = calibrateImages(img_l, img_r)

    diff_l = cv2.absdiff(image_prev_l, dst_l)
    diff_r = cv2.absdiff(image_prev_r, dst_r)

    diff_gray_l = cv2.cvtColor(diff_l, cv2.COLOR_BGR2GRAY)
    diff_gray_r = cv2.cvtColor(diff_r, cv2.COLOR_BGR2GRAY)

    diff_blur_l, diff_blur_r = gaussFilter(diff_gray_l, diff_gray_r, 21)

    dst_l_copy = dst_l.copy()

    _, thresh_l = cv2.threshold(diff_gray_l, 15, 255, cv2.THRESH_BINARY)
    _, thresh_r = cv2.threshold(diff_gray_r, 15, 255, cv2.THRESH_BINARY)
    dilate_l = cv2.dilate(thresh_l, None, iterations=3)
    dilate_r = cv2.dilate(thresh_r, None, iterations=3)
    _, contours_l, _ = cv2.findContours(dilate_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_r, _ = cv2.findContours(dilate_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_l = np.zeros(diff_gray_l.shape, dtype=np.uint8)
    mask_r = np.zeros(diff_gray_r.shape, dtype=np.uint8)

    xl2 = mask_l.shape[1]-1
    yl2 = mask_l.shape[0]-1
    xl1 = 0
    yl1 = 0

    if len(contours_l) > 0:
        contours_area = np.array([cv2.contourArea(contour) for contour in contours_l])
        contour_max = contours_l[np.argmax(contours_area)]
        if cv2.contourArea(contour_max) > contour_thresh:
            (x, y, w, h) = cv2.boundingRect(contour_max)
            cv2.rectangle(dst_l_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            xg2 = min(mask_l.shape[1]-1, x+w+margin_g)
            yg2 = min(mask_l.shape[0]-1, y+h+margin_g)
            xg1 = max(0, x-margin_g)
            yg1 = max(0, y-margin_g)
            mask_l[yg1:yg2, xg1:xg2] = 1

            xl2 = min(mask_l.shape[1]-1, x+w+margin_d)
            yl2 = min(mask_l.shape[0]-1, y+h+margin_d)
            xl1 = max(0, x-margin_d)
            yl1 = max(0, y-margin_d)
    dst_l_copy = cv2.bitwise_and(dst_l_copy, dst_l_copy, mask=mask_l)
    

    if len(contours_r) > 0:
        contours_area = np.array([cv2.contourArea(contour) for contour in contours_r])
        contour_max = contours_r[np.argmax(contours_area)]
        if cv2.contourArea(contour_max) > contour_thresh:
            (x, y, w, h) = cv2.boundingRect(contour_max)
            
            xg2 = min(mask_r.shape[1]-1, x+w+margin_g)
            yg2 = min(mask_r.shape[0]-1, y+h+margin_g)
            xg1 = max(0, x-margin_g)
            yg1 = max(0, y-margin_g)
            mask_r[yg1:yg2, xg1:xg2] = 1
    
    gray_l = cv2.cvtColor(dst_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(dst_r, cv2.COLOR_BGR2GRAY)

    gray_l = cv2.bitwise_and(gray_l, gray_l, mask=mask_l)
    gray_r = cv2.bitwise_and(gray_r, gray_r, mask=mask_r)

    disp = stereo.compute(gray_l, gray_r).astype('float')
    disp_min = np.min(disp)
    disp[:yl1,:] = disp_min
    disp[yl2:,:] = disp_min
    disp[yl1:yl2,:xl1] = disp_min
    disp[yl1:yl2,xl2:] = disp_min

    depth = 1/(disp)
    depth[depth == np.max(depth)] = np.nan


    pcd = createPointCloud(dst_l, depth)
    #pcd = pcd.uniform_down_sample(2)
    distances = np.asarray(pcd.compute_point_cloud_distance(static_cloud))
    pcd = pcd.select_by_index(np.where(distances > max_dist)[0])
    #labels = np.asarray(pcd.cluster_dbscan(eps=cluster_density, min_points=cluster_minpoints))
    #if len(labels) > 0:
    #    print(labels, np.max(labels)+1)
    #    if np.max(labels) > -1:
    #        pcd = pcd.select_by_index(np.where(labels == 0)[0])
    #        center = getObjectCenter(pcd)
    #    
    #        image_center_point = projectPointToImage(center)
    #        print(center, image_center_point)
    #        cv2.circle(dst_l_copy, (int(image_center_point[0]), int(image_center_point[1])), 5, (255, 0, 0))
        #draw_labels_on_model(pcd, labels)
    if len(pcd.points)>500:
        center = getObjectCenter(pcd)
        image_center_point = projectPointToImage(center)
        print(center, image_center_point)
        cv2.circle(dst_l_copy, (int(image_center_point[0]), int(image_center_point[1])), 5, (255, 0, 0))


    cv2.imshow('left' , dst_l_copy)
    #cv2.imshow('left_diff' , thresh_l)
    #cv2.imshow('right', dst_r)
    cv2.imshow('disp' , disp/8192.0)

    key = cv2.waitKey(1)
    if key == 27:
        break
    image_prev_l = dst_l
    image_prev_r = dst_r
cv2.destroyAllWindows()
# %%
