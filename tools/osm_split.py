
import cv2
import os
import glob
import shutil
import tqdm
import numpy as np

# 不删除全背景，裁剪
# 删除全背景，裁剪

# #  图像宽不足裁剪宽度,填充至裁剪宽度
# def fill_right(img, size_w):
#     size = img.shape
#     img_fill_right = cv2.copyMakeBorder(img, 0, 0, 0, size_w - size[1], 
#                                         cv2.BORDER_CONSTANT, value = (0, 0, 0))
#     return img_fill_right

# #  图像高不足裁剪高度,填充至裁剪高度
# def fill_bottom(img, size_h):
#     size = img.shape
#     img_fill_bottom = cv2.copyMakeBorder(img, 0, size_h - size[0], 0, 0, 
#                                          cv2.BORDER_CONSTANT, value = (0, 0, 0))
#     return img_fill_bottom

#  图像宽高不足裁剪宽高度,填充至裁剪宽高度
def fill_right_bottom(img, size_w, size_h, value=(0,0,0)):
    size = img.shape
    img_fill_right_bottom = cv2.copyMakeBorder(img, 0, max(0,size_h - size[0]), 0, max(0,size_w - size[1]), 
                                               cv2.BORDER_CONSTANT, value = value)
    return img_fill_right_bottom


def data_crop(img_paths, label_folder, osm_folder, out_img_floder, 
              out_label_floder, out_osm_folder, out_img_floder_test, out_label_floder_test, out_osm_folder_test,
              size_w = 512, size_h = 512, step = 512):
    
    count = 0
    for img_path in tqdm.tqdm(img_paths):
        
        number = 0
        
        img_name = os.path.basename(img_path)[:-4]  #获取文件名（不包括.tif）
        label_path = os.path.join(label_folder,img_name.replace("RGB","label")+".tif")  #获取完整相对路径
        osm_path = os.path.join(osm_folder, img_name, img_name+".tif")

        print(img_name)
        print(label_path)
        print(osm_path)
        #exit(0)
        img = cv2.imread(img_path,1)
        label = cv2.imread(label_path,0)
        osm = cv2.imread(osm_path,0)
        
        size = img.shape
        
        if size[0] < size_h or size[1] < size_w:
            img = fill_right_bottom(img,  size_w, size_h)
            label = fill_right_bottom(label,  size_w, size_h, value=(0))
            # print(f'图片{img_name}需要补齐')
        
        size = img.shape
        
        count = count + 1
        for h in range(0, size[0] - 1, step):
            start_h = h
            for w in range(0, size[1] - 1, step):
                
                start_w = w
                end_h = start_h + size_h
                if end_h > size[0]:
                   start_h = size[0] - size_h 
                   end_h = start_h + size_h
                end_w = start_w + size_w
                if end_w > size[1]:
                   start_w = size[1] - size_w
                end_w = start_w + size_w
                
                img_cropped = img[start_h : end_h, start_w : end_w]
                label_cropped = label[start_h : end_h, start_w : end_w]
                osm_cropped = osm[start_h : end_h, start_w : end_w]
                
                #  用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
                img_name_cropped = img_name + '_'+ str(start_h) +'_' + str(start_w)
                
                if img_name not in ['top_potsdam_5_13_RGB']:
                    cv2.imwrite(os.path.join(out_img_floder,img_name_cropped+".tif"), img_cropped)
                    cv2.imwrite(os.path.join(out_label_floder,img_name_cropped+".png"), label_cropped)
                    cv2.imwrite(os.path.join(out_osm_folder,img_name_cropped+".png"), osm_cropped/255.0)
                else:
                    cv2.imwrite(os.path.join(out_img_floder_test,img_name_cropped+".tif"), img_cropped)
                    cv2.imwrite(os.path.join(out_label_floder_test,img_name_cropped+".png"), label_cropped)
                    cv2.imwrite(os.path.join(out_osm_folder_test,img_name_cropped+".png"), osm_cropped/255.0)

                number = number + 1
                    
        print('{}.png切割成{}张.'.format(img_name,number))
    print('共完成{}张图片'.format(count))
    

#################################################################
#  标签图像文件夹
img_floder = "./dataset/osm_seg/2_Ortho_RGB/*.tif"    #需要修改
#  预测图像文件夹
label_folder = "./dataset/osm_seg/5_Labels_all_idx/"    #需要修改
# OSM dir
osm_folder = './dataset/osm_seg/5_Labels_osm/'

img_paths = glob.glob(img_floder)
label_paths = glob.glob(label_folder)

# 裁剪的图像、标签
out_img_floder = './dataset/osm_split_img/'
out_label_floder = './dataset/osm_split_label/'
out_osm_folder = './dataset/osm_split_osm_label/'

out_img_floder_test = './dataset/osm_split_img_test/'
out_label_floder_test = './dataset/osm_split_label_test/'
out_osm_folder_test = './dataset/osm_split_osm_label_test/'

if os.path.exists(out_img_floder):
    shutil.rmtree(out_img_floder)
os.makedirs(out_img_floder)
if os.path.exists(out_label_floder):
    shutil.rmtree(out_label_floder)
os.makedirs(out_label_floder)
if os.path.exists(out_osm_folder):
    shutil.rmtree(out_osm_folder)
os.makedirs(out_osm_folder)


if os.path.exists(out_img_floder_test):
    shutil.rmtree(out_img_floder_test)
os.makedirs(out_img_floder_test)
if os.path.exists(out_label_floder_test):
    shutil.rmtree(out_label_floder_test)
os.makedirs(out_label_floder_test)
if os.path.exists(out_osm_folder_test):
    shutil.rmtree(out_osm_folder_test)
os.makedirs(out_osm_folder_test)


#  切割图像宽å
size_w = 512
#  切割图像高
size_h = size_w
#  切割步长,重叠度为size_w - step
step = int(size_h*1.0)

data_crop(img_paths, label_folder, osm_folder, out_img_floder, 
          out_label_floder, out_osm_folder, out_img_floder_test, out_label_floder_test, out_osm_folder_test,
          size_w = size_w, size_h = size_h, step = step)