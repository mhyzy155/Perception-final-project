# %%
import cv2
import numpy as np
from pathlib import Path,PurePath
import matplotlib.pyplot as plt
import open3d as o3d

# %%
base_path = Path("./with_occlusions")
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
   #bounding = pcd.get_axis_aligned_bounding_box()
   #center = bounding.get_center()
   #center = pcd.get_center()
   return center

def projectPointToImage(point):
   fx = new_cam_l[0][0]
   fy = new_cam_l[1][1]
   cx = new_cam_l[0][2]
   cy = new_cam_l[1][2]
   u = cx + fx*point[0]/point[2]
   v = cy + fy*point[1]/point[2]
   return np.asarray([u,v])

def projectImageToPoint(point):
    fx = new_cam_l[0][0]
    fy = new_cam_l[1][1]
    cx = new_cam_l[0][2]
    cy = new_cam_l[1][2]
    u = (point[0]-cx)/fx
    v = (point[1]-cy)/fy
    return np.asarray([u,v,1])

def cleanFgMask(fg_mask):
    _, fg_mask_thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
    fg_mask_fill = cv2.dilate(fg_mask_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)), iterations=3)
    fg_mask_erode = cv2.erode(fg_mask_fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=10)
    return cv2.dilate(fg_mask_erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)), iterations=5)

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
cluster_density = 2e-7
cluster_minpoints = 3

min_disp = 70
num_disp = 20 * 16
block_size = 31
stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(10)
stereo.setUniquenessRatio(25)
stereo.setSpeckleRange(3)
stereo.setSpeckleWindowSize(3)

# create point cloud of background
static_cloud = o3d.geometry.PointCloud()
images_l = sorted(left.glob('*_Left.png'))
for filename_left in images_l[:30]:
    filename_stem = PurePath(filename_left).stem
    filename_parts = filename_stem.split('_')
    file_number = filename_parts[0] + '_' + filename_parts[1]
    filename_right = right / (file_number + '_Right.png')
    img_l = cv2.imread(str(filename_left))
    img_r = cv2.imread(str(filename_right))
    dst_l, dst_r = calibrateImages(img_l, img_r)
    gray_l = cv2.cvtColor(dst_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(dst_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gray_l, gray_r).astype('float')
    depth = 1/(disp)
    depth[depth == np.max(depth)] = np.nan
    static_cloud += createPointCloud(dst_l, depth)
    print(len(static_cloud.points))
o3d.visualization.draw_geometries([static_cloud])
cl, ind = static_cloud.remove_radius_outlier(nb_points=10, radius=1.0e-9)
print(len(ind))
static_cloud = static_cloud.select_by_index(ind)
o3d.visualization.draw_geometries([static_cloud])
static_cloud = static_cloud.voxel_down_sample(1.0e-8)
print(len(static_cloud.points))
o3d.visualization.draw_geometries([static_cloud])

#%% prediction part
import torch, torchvision

import numpy as np
import cv2

from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

FILE_PATH='Models/Dense169.bin'
classes=['book', 'box', 'cup']

# %%
def create_model(n_classes):
  model = models.densenet169()
  #Uncomment if you use ResNet model. It specify how many featurers and classes we use
  #For densenet it does not work, check gow to do it?
#   n_features = model.fc.in_features
#   model.fc = nn.Linear(n_features, n_classes)

  return model.to(device)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = create_model(len(classes))
trained_model.load_state_dict(torch.load(FILE_PATH, map_location=torch.device('cpu')))
trained_model.eval()

#%%
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transforms = {'test': T.Compose([
  T.Resize((256, 256)),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
])
}
from PIL import Image
from matplotlib import cm
def prediction(model, image):
  img = Image.fromarray(image.astype('uint8'), 'RGB')
#   img = img.convert('RGB')
  img = transforms['test'](img).unsqueeze(0)

  pred = model(img.to(device))
  pred = F.softmax(pred, dim=1)
  return pred.detach().cpu().numpy().flatten()

# %%
dt = 1/120
initCovariance = 10000
updateNoise = 0.001
measurementNoise = 10
vel_fac = 0.1

kalman = cv2.KalmanFilter(6, 3, 0)
kalman.transitionMatrix = np.array([
   [1, 0, 0, dt, 0, 0],
   [0, 1, 0, 0, dt, 0],
   [0, 0, 1, 0, 0, dt],
   [0, 0, 0, 1, 0, 0 ],
   [0, 0, 0, 0, 1, 0 ],
   [0, 0, 0, 0, 0, 1 ]
]).astype('float32')
kalman.measurementMatrix = np.array([
   [1, 0, 0, 0, 0, 0],
   [0, 1, 0, 0, 0, 0],
   [0, 0, 1, 0, 0, 0]
]).astype('float32')
kalman.processNoiseCov = updateNoise*np.diag([1, 1, 1, vel_fac, vel_fac, vel_fac]).astype('float32')
kalman.measurementNoiseCov = measurementNoise*np.eye(3).astype('float32')
kalman.statePost = np.zeros((6,1)).astype('float32')
kalman.statePost[2] = 1e-9
kalman.errorCovPost = initCovariance * np.eye(6).astype('float32')
# %%
backSub_l = cv2.createBackgroundSubtractorMOG2(varThreshold=32)
backSub_r = cv2.createBackgroundSubtractorMOG2(varThreshold=32)
margin_g = 40
margin_d = 10
h, w = 0, 0
roi_counter = 0
roi_area_max = 1
roi_x = 0
roi_y = 0
contour_thresh = 500
threshold = 1.0e-10
trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
point_to_point =  o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
point_to_plane =  o3d.pipelines.registration.TransformationEstimationPointToPlane()
pcd_old = o3d.geometry.PointCloud()

images_l = sorted(left.glob('*_Left.png'))
filename_stem = PurePath(images_l[0]).stem
filename_parts = filename_stem.split('_')
file_number = filename_parts[0] + '_' + filename_parts[1]
filename_right = right / (file_number + '_Right.png')
image_prev_l, image_prev_r = calibrateImages(cv2.imread(str(images_l[0])), cv2.imread(str(filename_right)))

for filename_left in images_l[2:]:
    filename_stem = PurePath(filename_left).stem
    filename_parts = filename_stem.split('_')
    file_number = filename_parts[0] + '_' + filename_parts[1]
    filename_right = right / (file_number + '_Right.png')
    img_l = cv2.imread(str(filename_left))
    img_r = cv2.imread(str(filename_right))

    dst_l, dst_r = calibrateImages(img_l, img_r)
    fgMask_l = backSub_l.apply(dst_l)
    fgMask_margin_l = cleanFgMask(fgMask_l)
    fgMask_r = backSub_r.apply(dst_r)
    fgMask_margin_r = cleanFgMask(fgMask_r)

    _, contours_l, _ = cv2.findContours(fgMask_margin_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_r, _ = cv2.findContours(fgMask_margin_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_l = np.zeros(fgMask_margin_l.shape, dtype=np.uint8)
    mask_r = np.zeros(fgMask_margin_r.shape, dtype=np.uint8)

    dst_l_copy = dst_l.copy()
    dst_r_copy = dst_r.copy()

    xl2 = mask_l.shape[1]-1
    yl2 = mask_l.shape[0]-1
    xl1 = 0
    yl1 = 0
    predict_flag=False
    if len(contours_l) > 0:
        contours_area = np.array([cv2.contourArea(contour) for contour in contours_l])
        contour_max = contours_l[np.argmax(contours_area)]
        if cv2.contourArea(contour_max) > contour_thresh:
            (x, y, w, h) = cv2.boundingRect(contour_max)
            roi_x = int(x+w/2)
            roi_y = int(y+h/2)            
            if w>0 and h>0:
                to_predict=dst_l_copy[y:(y+h), x:(x+w)]
                predict_flag=True
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
    #dst_l_copy = cv2.bitwise_and(dst_l_copy, dst_l_copy, mask=mask_l)
    

    if len(contours_r) > 0:
        contours_area = np.array([cv2.contourArea(contour) for contour in contours_r])
        contour_max = contours_r[np.argmax(contours_area)]
        if cv2.contourArea(contour_max) > contour_thresh:
            (x, y, w, h) = cv2.boundingRect(contour_max)
            cv2.rectangle(dst_r_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)
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

    # Kalman update
    kalman.predict()

    pcd = createPointCloud(dst_l, depth)
    distances = np.asarray(pcd.compute_point_cloud_distance(static_cloud))
    pcd = pcd.select_by_index(np.where(distances > max_dist)[0])
    #print('before:', len(pcd.points))
    #cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=5.0e-9)
    #print('after:', len(pcd.points))
    #pcd = pcd.select_by_index(ind)
    
    # voxel_size = 1.0e-9
    # if len(pcd.points) > 1500:
    #     cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=5.0e-9)
    #     pcd = pcd.select_by_index(ind)
    #     if len(pcd_old.points) > 0:
    #         pcd_old.paint_uniform_color([0, 0.651, 0.929])
    #         pcd.paint_uniform_color([1, 0.706, 0])
    #         o3d.visualization.draw_geometries([pcd_old, pcd])

    #         #pcd_old.estimate_normals()
    #         #pcd.estimate_normals()
    #         pcd_old.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100), fast_normal_computation=True)
    #         pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100), fast_normal_computation=True)
    #         source_features = o3d.pipelines.registration.compute_fpfh_feature(pcd_old, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    #         target_features = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
            
    #         ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #             pcd_old, pcd, 
    #             source_features, target_features, True, 
    #             voxel_size * 10,
    #             point_to_point, criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500))
    #         pcd_old.transform(ransac_result.transformation)
    #         o3d.visualization.draw_geometries([pcd_old, pcd])

    #         print("Initial alignment")
    #         evaluation = o3d.pipelines.registration.evaluate_registration(pcd_old, pcd, threshold, trans_init)
    #         print(evaluation)

    #         pcd_old.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=30), fast_normal_computation=True)
    #         pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=30), fast_normal_computation=True)
    #         icp_result = o3d.pipelines.registration.registration_icp(
    #             pcd_old, pcd, threshold*10, trans_init,
    #             point_to_plane, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    #         #pcd_old.transform(icp_result.transformation)
    #         #o3d.visualization.draw_geometries([pcd_old, pcd])
    #     pcd_old += pcd
    #     pcd_old.paint_uniform_color([0, 0.651, 0.929])
    #     print('before:', len(pcd_old.points))
    #     cl, ind = pcd_old.remove_radius_outlier(nb_points=15, radius=1.0e-8)
    #     print('left:', len(ind))
    #     pcd_old = pcd_old.select_by_index(ind)
    #     #o3d.visualization.draw_geometries([pcd_old])
    #     pcd_old = pcd_old.voxel_down_sample(2.0e-9)
    #     print('after downsampling:', len(pcd_old.points))
    #     #o3d.visualization.draw_geometries([pcd_old])

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
    # print("Number of points in pointcloud: ", len(pcd_old.points))

    if len(pcd.points)>200 and roi_x < 1230 and roi_y > 300:
        roi_area = w*h
        print('Roi area:',roi_area)
        roi_area_max = max(roi_area_max, roi_area)
        roi_ratio = roi_area/roi_area_max
        print('RoiRatio:',roi_ratio)

    if roi_x < 420 and roi_y > 560:
        roi_area_max = 1
        kalman.statePost = np.zeros((6,1)).astype('float32')
        kalman.statePost[2] = 1e-9
        kalman.errorCovPost = initCovariance * np.eye(6).astype('float32')


    if len(pcd.points)>200 and roi_x < 1230 and roi_y > 300 and roi_ratio > 0.6:
        #print(len(pcd.points))
        #o3d.visualization.draw_geometries([pcd])
        pcd = pcd.voxel_down_sample(1.0e-8)
        #print(len(pcd.points))
        #o3d.visualization.draw_geometries([pcd])
        center = getObjectCenter(pcd)
        M = projectImageToPoint([roi_x, roi_y])
        center = np.dot(M, center)/np.dot(M,M) * M
        #Kalman update
        kalman.correct(np.reshape(center, (3,1)).astype('float32'))
        image_center_point = projectPointToImage(center)
        # print(center, image_center_point)
        cv2.circle(dst_l_copy, (int(image_center_point[0]), int(image_center_point[1])), 5, (255, 0, 0))
    out=""
    if predict_flag:        
        pred = prediction(trained_model, to_predict)
        out = classes[np.argmax(pred[0:3])] + str(np.max(pred[0:3]))
        # cv2.imshow('left' , to_predict)
    kalman_center_point = projectPointToImage(kalman.statePost)
    cv2.circle(dst_l_copy, (int(kalman_center_point[0]), int(kalman_center_point[1])), 5, (0,0,255))
    cv2.putText(dst_l_copy, str(kalman.statePost[0]) ,(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, str(kalman.statePost[1]) ,(10,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, str(kalman.statePost[2]) ,(10,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, str(kalman.statePost[3]) ,(10,250), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, str(kalman.statePost[4]) ,(10,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, str(kalman.statePost[5]) ,(10,350), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    cv2.putText(dst_l_copy, out ,(10,500), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.imshow('left' , dst_l_copy)
    #cv2.imshow('fgmask', fgMask_thresh_l)
    cv2.imshow('fgmask', fgMask_margin_l)
    #cv2.imshow('left_diff' , diff_blur_l)
    #cv2.imshow('right', dst_r)
    # cv2.imshow('disp' , disp/8192.0)

    key = cv2.waitKey(1)
    if key == 27:
        break
    image_prev_l = dst_l
    image_prev_r = dst_r
cv2.destroyAllWindows()
# %%
