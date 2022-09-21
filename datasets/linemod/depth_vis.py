import numpy as np
from cv2 import imshow, waitKey
from PIL import Image
def depth2show(depth, norm_type='max'):
    show_depth = ((depth / (depth.max()/2)) * 255).astype("uint8")
    return show_depth
path = "/data1/zzh/DenseFusion-Pytorch-1.0/datasets/linemod/Linemod_preprocessed/data/01/depth/0000.png"
depth = np.array(Image.open(path))
new_depth = depth2show(depth)
imshow("new_depth", new_depth)
waitKey(0)