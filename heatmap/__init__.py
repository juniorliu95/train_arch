#import cf
import heatmap
import os
import cv2
if __name__ == '__main__':
    path = '/home/eric/Documents/SHENZHEN&MC/NLM_jpg/'
    out_path = '/home/eric/Documents/SHENZHEN&MC/NLM_crop/'
    for name in os.listdir(path):
        img = cv2.imread(path+name,0)
        img_crop = heatmap.crop(img)
#        cv2.imshow('',img_crop)
#        cv2.waitKey()
#        cv2.destroyAllWindows()
        cv2.imwrite(out_path+name,img_crop)
        