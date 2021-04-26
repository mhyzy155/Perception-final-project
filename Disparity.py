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
    print(depth.astype('float32').dtype)
    depth = o3d.geometry.Image(depth.astype('float32'))
    rgb = o3d.geometry.Image(rgb_img.astype('float32'))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
      rgb, depth,
      depth_trunc=1000.0
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
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

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