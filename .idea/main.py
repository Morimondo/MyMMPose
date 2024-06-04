from mmpose.apis import MMPoseInferencer
from PIL import Image
import cv2
from imageHelper import ImageHelper
import os

ih = ImageHelper()
img_dir_source = './image/source/'
img_dir_result = './image/result/'
ih.save(img_dir_source + "img_tmp.png", ih.getCaptureImage())

image_path_list = os.listdir(img_dir_source)

for img_name in image_path_list:
    img_path = img_dir_source + img_name   # replace this with your own image path
    print(img_path)

    #resize
    # img = ih.getPilImage(img_path)    # img_resize = ih.trimming(img, left=700, right=700, top=400, bottom=400)
    # img_resize_path = "./image/img_resize.jpg"
    # img_resize.save(img_resize_path)

    img_resize_path = img_path
    # instantiate the inferencer using the model alias
    inferencer = MMPoseInferencer('human')

    # The MMPoseInferencer API employs a lazy inference approach,
    # creating a prediction generator when given input
    # result_generator = inferencer(img_path, show=True)
    result_generator = inferencer(img_resize_path, show=False)
    result = next(result_generator)
    #ポイント
    keypoints = result["predictions"][0][0]['keypoints']
    keypoints_int = []
    for pos in keypoints:
        p = (int(pos[0]), int(pos[1]))
        keypoints_int.append(p)
    eye_left = keypoints_int[2]
    eye_right = keypoints_int[3]
    ear_left = keypoints_int[4]
    ear_right = keypoints_int[5]

    #cv2
    img = cv2.imread(img_resize_path)

    print(eye_left)
    color = (0, 0, 255)
    for pos in keypoints_int[1:5]:
        img[pos[1] : pos[1] + 5, pos[0] : pos[0] + 5] = color
        color = (0, 255, 0) if color == (0, 0, 255) else (0, 0, 255)
    #反射板
    ref_pos = ih.getReflectorPos(img)
    for p in ref_pos:
        img[p[0], p[1]] = (255, 0, 0)
    # cv2.imshow("result", img)
    cv2.imwrite(img_dir_result + img_name, img)