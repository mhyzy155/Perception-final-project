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
Q_mat = np.load("Q_mat.npy")

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
      depth_scale=1, depth_trunc=1040.0
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

def getJacobian(u,v,d,camMatrix,scale):
    fx = camMatrix[0][0]
    fy = camMatrix[1][1]
    cx = camMatrix[0][2]
    cy = camMatrix[1][2]
    return np.array([
        [d/fx/scale, 0, (u-cx)/fx/scale],
        [0, d/fy/scale, (v-cy)/fy/scale],
        [0, 0, 1/scale]]).astype('float32')
# %%
def biFilter(img_l, img_r, k_size, sigma):
    return cv2.bilateralFilter(img_l,k_size,sigma,sigma), cv2.bilateralFilter(img_r,k_size,sigma,sigma)

def gaussFilter(img_l, img_r, k_size):
    return cv2.GaussianBlur(img_l,(k_size,k_size),0), cv2.GaussianBlur(img_r,(k_size,k_size),0)

# %%
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

min_disp = 80
num_disp = 15 * 16
block_size = 5
stereo = cv2.StereoSGBM_create(numDisparities = num_disp, blockSize = block_size, P1 = 8*block_size**2, P2 = 32*block_size**2, preFilterCap = 2)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(230)
stereo.setUniquenessRatio(15)
stereo.setSpeckleRange(1)
stereo.setSpeckleWindowSize(50)

# create a point cloud of the background
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
    disp = stereo.compute(gray_l, gray_r).astype('float32') / 16.0
    depth = cv2.reprojectImageTo3D(disp, Q_mat)
    static_cloud += createPointCloud(dst_l, depth[:,:,2])
    print("static_cloud size:", len(static_cloud.points))

#o3d.visualization.draw_geometries([static_cloud, mesh_frame])
cl, ind = static_cloud.remove_radius_outlier(nb_points=3, radius=0.3)
print("static_cloud size after removing noise:", len(ind))
static_cloud = static_cloud.select_by_index(ind)
#o3d.visualization.draw_geometries([static_cloud, mesh_frame])
static_cloud = static_cloud.voxel_down_sample(1.0)
print("static_cloud size after downsampling:", len(static_cloud.points))
#o3d.visualization.draw_geometries([static_cloud, mesh_frame])

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
dt = 1/30
initCovariance = 10000
updateNoise = 0.001
measurementNoise = 0.5
vel_fac = 0.1
depth_fac = 0.1

kalmanImageCov = measurementNoise*np.diag([1, 1, depth_fac]).astype('float32')

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
#kalman.processNoiseCov = updateNoise*np.eye(6).astype('float32')
#kalman.measurementNoiseCov = measurementNoise*np.eye(3).astype('float32')
kalman.statePost = np.zeros((6,1)).astype('float32')
kalman.statePost[2] = 1e-9
kalman.errorCovPost = initCovariance * np.eye(6).astype('float32')
# %%
backSub_l = cv2.createBackgroundSubtractorMOG2(varThreshold=32)

static_cloud.paint_uniform_color([1, 0.706, 0])
h, w = 0, 0
roi_counter = 0
pcd_len_max = 1
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('result.avi',fourcc, 30.0, image_prev_l.shape[1::-1])

for iteration, filename_left in enumerate(images_l[1:]):
    filename_stem = PurePath(filename_left).stem
    filename_parts = filename_stem.split('_')
    file_number = filename_parts[0] + '_' + filename_parts[1]
    filename_right = right / (file_number + '_Right.png')
    img_l = cv2.imread(str(filename_left))
    img_r = cv2.imread(str(filename_right))

    dst_l, dst_r = calibrateImages(img_l, img_r)

    fgMask_l = backSub_l.apply(dst_l)
    fgMask_clean_l = cleanFgMask(fgMask_l)
    _, contours_l, _ = cv2.findContours(fgMask_clean_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_l = np.zeros(fgMask_clean_l.shape, dtype=np.uint8)

    dst_l_copy = dst_l.copy()

    # Kalman update
    kalman.predict()

    predict_flag=False
    if len(contours_l) > 0 and iteration > 30:
        contours_area = np.array([cv2.contourArea(contour) for contour in contours_l])
        contour_max = contours_l[np.argmax(contours_area)]
        if cv2.contourArea(contour_max) > contour_thresh:
            (x, y, w, h) = cv2.boundingRect(contour_max)
            cv2.rectangle(dst_l_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)

            roi_x = int(x+w/2)
            roi_y = int(y+h/2)

            if w>0 and h>0:
                to_predict=dst_l_copy[y:(y+h), x:(x+w)]
                predict_flag=True
            
            x2 = min(mask_l.shape[1]-1, x+w)
            y2 = min(mask_l.shape[0]-1, y+h)
            x1 = max(0, x)
            y1 = max(0, y)
            mask_l[y1:y2, x1:x2] = 1
    
        gray_l = cv2.cvtColor(dst_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(dst_r, cv2.COLOR_BGR2GRAY)

        disp = stereo.compute(gray_l, gray_r).astype('float32') / 16.0
        disp = cv2.bitwise_and(disp, disp, mask=mask_l)
        cv2.imshow('disp' , disp/256.0)

        depth = cv2.reprojectImageTo3D(disp, Q_mat)
        pcd = createPointCloud(dst_l, depth[:,:,2])
        distances = np.asarray(pcd.compute_point_cloud_distance(static_cloud))
        pcd = pcd.select_by_index(np.where(distances > 30.0)[0])
        #if(len(pcd.points)) > 0:
        #    o3d.visualization.draw_geometries([pcd, static_cloud, mesh_frame])
        
        #print('pcd w/  noise:', len(pcd.points))
        cl, ind = pcd.remove_radius_outlier(nb_points=200, radius=30.0)
        pcd = pcd.select_by_index(ind)
        #print('pcd w/o noise:', len(pcd.points))
        #bb = o3d.geometry.AxisAlignedBoundingBox().create_from_points(pcd.points)
        #if(len(pcd.points)) > 0:
        #    o3d.visualization.draw_geometries([pcd, static_cloud, bb, mesh_frame])
        pcd_len = len(pcd.points)

        if pcd_len > 200 and roi_x < 1230 and roi_y > 300:
            pcd_len_max = max(pcd_len_max, pcd_len)
            pcd_ratio = pcd_len/pcd_len_max
            print('pcd len: ', pcd_len)
            print('pcdRatio:', pcd_ratio)

        if pcd_len > 200 and roi_x < 1230 and roi_y > 300 and pcd_ratio > 0.6:
            pcd = pcd.voxel_down_sample(1.0e-1)
            center = getObjectCenter(pcd)
            M = projectImageToPoint([roi_x, roi_y])
            center = np.dot(M, center)/np.dot(M,M) * M

            #Kalman update
            measure_image_center = projectPointToImage(center)
            jacobian = getJacobian(measure_image_center[0], measure_image_center[1], center[2], new_cam_l, 1)
            print(center)
            print(jacobian)
            kalman.measurementNoiseCov = jacobian@kalmanImageCov@np.transpose(jacobian)
            print(kalman.measurementNoiseCov)
            kalman.correct(np.reshape(center, (3,1)).astype('float32'))
            image_center_point = projectPointToImage(center)
            cv2.circle(dst_l_copy, (int(image_center_point[0]), int(image_center_point[1])), 5, (0, 255, 0), 2)

        if roi_x < 430:
            pcd_len_max = 1
            kalman.statePost = np.zeros((6,1)).astype('float32')
            kalman.statePost[2] = 1e-9
            kalman.errorCovPost = initCovariance * np.eye(6).astype('float32')

    out=""
    if predict_flag:        
        pred = prediction(trained_model, to_predict)
        out = "{0}, confidence: {1:4.2f}".format(classes[np.argmax(pred[0:3])], np.max(pred[0:3]))
        # cv2.imshow('left' , to_predict)
    
    kalman_center_point = projectPointToImage(kalman.statePost)
    cv2.circle(dst_l_copy, (int(kalman_center_point[0]), int(kalman_center_point[1])), 9, (0,0,255), 2)

    abs_vel = np.sqrt(kalman.statePost[3][0]**2+kalman.statePost[4][0]**2+kalman.statePost[5][0]**2)
    cv2.putText(dst_l_copy, 'frame: ' + str(iteration+1) ,(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "x:       {0:4.2f} m".format(kalman.statePost[0][0]/1000), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "y:       {0:4.2f} m".format(kalman.statePost[1][0]/1000), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "z:       {0:4.2f} m".format(kalman.statePost[2][0]/1000), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "x vel:   {0:4.2f} m/s".format(kalman.statePost[3][0]/1000), (10,250), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "y vel:   {0:4.2f} m/s".format(kalman.statePost[4][0]/1000), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "z vel:   {0:4.2f} m/s".format(kalman.statePost[5][0]/1000), (10,350), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, "abs vel: {0:4.2f} m/s".format(abs_vel/1000), (10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(dst_l_copy, out ,(10,500), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.imshow('left' , dst_l_copy)
    video_out.write(dst_l_copy)

    key = cv2.waitKey(1)
    if key == 27:
        break
    image_prev_l = dst_l
    image_prev_r = dst_r
video_out.release()
cv2.destroyAllWindows()
# %%
