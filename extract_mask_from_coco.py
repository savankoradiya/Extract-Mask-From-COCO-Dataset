from pycocotools.coco import COCO
import numpy as np
import cv2
import os
#Add your coco dataset images folder path
data_dir = '/home/savan/dataset/coco/val2017'
ann_file='/home/savan/dataset/coco/instances_val2017.json'
#Add your output mask folder path
seg_output_path = '/savan/dataset/coco/seg'
#Store original images into another folder
original_img_path = '/savan/dataset/coco/img'
coco = COCO(ann_file)
catIds = coco.getCatIds(catNms=['person']) #Add more categories ['person','dog']
imgIds = coco.getImgIds(catIds=catIds )

for i in range(len(imgIds)):
	img = coco.loadImgs(imgIds[i])[0]
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
	anns = coco.loadAnns(annIds)
	mask = coco.annToMask(anns[0])
	for i in range(len(anns)):
		mask += coco.annToMask(anns[i])
	file_name = os.path.join(data_dir,img['file_name'])
	original_img = cv2.imread(file_name)
	cv2.imwrite(os.path.join(original_img_path,img['file_name']),original_img)
	cv2.imwrite(os.path.join(seg_output_path,img['file_name']),mask)
	print("processing...")

print("Done")
