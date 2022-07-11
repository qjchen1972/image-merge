import cv2
import numpy as np
import os    
import matplotlib.pyplot as plt
import sys
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils

def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image-mean))
    image = (image - mean)/np.sqrt(var)
    return image
    
def minMax(img):    
    return (img - img.min()) / np.maximum(img.max() - img.min(), np.finfo(np.float32).eps)
   

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def delPadding(img):
    temp = np.where(img!=0)
    return img[:temp[0].max(), :temp[1].max()]


def mask2point(mask, num=1):
    img = (mask > 0)
    img = cv2.copyMakeBorder(img.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    
    polygons = cv2.findContours(img, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE, offset=(-1,-1))
    
    #polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    polygons = imutils.grab_contours(polygons)
    
    area = np.array([cv2.contourArea(i) for i in polygons]) 
    if area.shape[0] == 0: return None    
    temp = [polygons[id] for id in area.argsort()[-num:]]    
    return temp[::-1]    
    #return polygons[area.argmax()] if area.shape[0]>0  else None
    
class ImgMerge:

    def __init__(self):
        super().__init__()
        self.src_img = None
        self.dst_img = None
        
    def setSrc(self, img):
        self.src_img = img
        self.src_kps, self.src_features = self.detectAndCompute(img)
    
    def setDst(self, img):
        self.dst_img = img
        self.dst_kps, self.dst_features = self.detectAndCompute(img)    
    
    
    def detectAndCompute(self, img):
    
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        '''
        nfeatures: 需要保留的特征点的个数，特征按分数排序（分数取决于局部对比度）
	    nOctaveLayers：每一组高斯差分金字塔的层数，sift论文中用的3。
        高斯金字塔的组数通过图片分辨率计算得到
	    contrastThreshold： 对比度阈值，用于过滤低对比度区域中的特征点。阈值越大，检测器产生的特征越少。 
        (sift论文用的0.03，nOctaveLayers若为3， 设置参数为0.09，实际值为：contrastThreshold/nOctaveLayers)
	    edgeThreshold：用于过滤掉类似图片边界处特征的阈值(边缘效应产生的特征)，
        注意其含义与contrastThreshold不同，即edgeThreshold越大，检测器产生的特征越多(过滤掉的特征越少)；sift论文中用的10；
	    sigma：第一组高斯金字塔高斯核的sigma值，sift论文中用的1.6。 
        (图片较模糊，或者光线较暗的图片，降低这个参数)
        '''
        sift = cv2.SIFT_create(nfeatures=0, 
                               nOctaveLayers=3, 
                               contrastThreshold=0.04, 
                               edgeThreshold=10, 
                               sigma=1.6)
                               
        '''
        image：需要检测关键点的图片
	    mask：掩膜，为0的区域表示不需要检测关键点，大于0的区域检测
        返回：
        kps:
        ngle: 特征点的方向，值在0-360
        class_id: 用于聚类id,没有进行聚类时为-1
        octave: 特征点所在的高斯金差分字塔组
        pt: 特征点坐标
        response: 特征点响应强度，代表了该点时特征点的程度（特征点分数排序时，会根据特征点强度）
        size:特征点领域直径
        
        features:
        检测点对应的descriptor，是一个128维的向量
        '''        
        (kps, features) = sift.detectAndCompute(image=image_gray, mask=None)
        
        '''
        image：检测关键点的原始图像
	    keypoints：检测到的关键点
	    outImage：绘制关键点后的图像，其内容取决于falgs的设置
	    color：绘制关键点采用的颜色
	    flags：
		cv2.DRAW_MATCHES_FLAGS_DEFAULT:默认值，匹配了的关键点和单独的关键点都会被绘制
		cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 绘制关键点，且每个关键点都绘制圆圈和方向
		cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：
		cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：只绘制匹配的关键点，单独的关键点不绘制
        '''
        #img = cv2.drawKeypoints(image=image_gray, outImage=img, keypoints=kps,
        #                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        #                    color=(51, 163, 236))
        
        #cv_show("myImage", img) 
        
        kps = np.float32([kp.pt for kp in kps])         
        return kps, features


    # ratio是最近邻匹配的推荐阈值
    # reprojThresh是随机取样一致性的推荐阈值     
    def matchKeyPoints(self, ratio=0.75, reprojThresh=4.0):
    
        assert self.src_img is not None and self.dst_img is not None,\
             "src or dst is None"
        
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(self.src_features, self.dst_features, 2)
    
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
             matches.append((m[0].queryIdx, m[0].trainIdx))
             
        self.src_match_kps = np.float32([np.round(self.src_kps[m[0]]) for m in matches]) 
        self.dst_match_kps = np.float32([np.round(self.dst_kps[m[1]]) for m in matches])
    
        '''
        kpsA: 输入源平面的坐标矩阵，我这里就是像素坐标
        kpsB: 输入目标平面的坐标矩阵，我输入世界坐标
        method:
            0 - 利用所有点的常规方法
            RANSAC - RANSAC-基于RANSAC的鲁棒算法
            LMEDS - 最小中值鲁棒算法
            RHO - PROSAC-基于PROSAC的鲁棒算法
       
        ransacReprojThreshold:
            将点对视为内点的最大允许重投影错误阈值（仅用于RANSAC和RHO方法）
            
        并不是所有的点都有匹配解，它们的状态存在status中    
        '''
        self.M, self.status = cv2.findHomography(self.dst_match_kps, 
                                                  self.src_match_kps, 
                                                  cv2.RANSAC, 
                                                  reprojThresh)
                                                  
        dst_kps = self.dst_match_kps[self.status[:,0] == 1].astype(int)
        self.dst_zone = (dst_kps[:, 1].min(),
                         dst_kps[:, 0].min(),
                         dst_kps[:, 1].max(),
                         dst_kps[:, 0].max())
                                 
        src_kps = self.src_match_kps[self.status[:,0] == 1].astype(int)
        self.src_zone = (src_kps[:, 1].min(),
                         src_kps[:, 0].min(),                         
                         src_kps[:, 1].max(),
                         src_kps[:, 0].max())
        
        
    def drawMatches(self):
    
        (srcH, srcW) = self.src_img.shape[0:2]
        (dstH, dstW) = self.dst_img.shape[0:2]
        
        drawImg = np.zeros((srcH + dstH, srcW + dstW, 3), 'uint8')
        
        src_kps = self.src_match_kps[self.status[:,0] == 1]
        dst_kps = self.dst_match_kps[self.status[:,0] == 1]

        drawImg[0:srcH, 0:srcW] = self.src_img        
        drawImg[srcH:, srcW:] = self.dst_img
        
        for src, dst in zip(src_kps, dst_kps):
            pt1 = (round(src[0]), round(src[1]))
            pt2 = (round(dst[0]) + srcW, round(dst[1]) + srcH)
            cv2.line(drawImg, pt1, pt2, (0, 0, 255))    
    
        cv_show("drawImg", drawImg)
    
    
    def stich(self):
    
        (srcH, srcW) = self.src_img.shape[0:2]
        (dstH, dstW) = self.dst_img.shape[0:2]        
    
        result = cv2.warpPerspective(self.dst_img, self.M, (srcW + dstW, srcH + dstH),
                                     flags=cv2.INTER_NEAREST, 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=None)
        #cv_show('result', result)        
        
        for i in range(srcH):
            for j in range(srcW):
                if self.src_img[i,j].any():
                    result[i,j] = self.src_img[i,j]
        
        return  delPadding(result)     
  
    
    def getSrcZoneFromSrc(self):
        return self.src_img[self.src_zone[0]:self.src_zone[2] + 1,
                            self.src_zone[1]:self.src_zone[3] + 1]
        
    def getDstZoneFromSrc(self):
    
                                               
        (srcH, srcW) = self.src_img.shape[0:2]
        (dstH, dstW) = self.dst_img.shape[0:2]        
         
        #使用M矩阵来求出dst映射到src的相应区域 
        result = cv2.warpPerspective(self.dst_img, self.M, (srcW + dstW, srcH + dstH),
                                     flags=cv2.INTER_NEAREST)
        
        return result[self.src_zone[0]:self.src_zone[2] + 1,
                      self.src_zone[1]:self.src_zone[3] + 1]
                            
    def getDstZoneFromDst(self):
        return self.dst_img[self.dst_zone[0]:self.dst_zone[2] + 1,
                            self.dst_zone[1]:self.dst_zone[3] + 1]
        
    def getSrcZoneFromDst(self):
    
                                               
        (srcH, srcW) = self.src_img.shape[0:2]
        (dstH, dstW) = self.dst_img.shape[0:2]        
         
        #使用M逆矩阵来求出src映射到dst的相应区域 
        result = cv2.warpPerspective(self.src_img, self.M, (srcW + dstW, srcH + dstH),
                                     flags=cv2.WARP_INVERSE_MAP)
        
        return result[self.dst_zone[0]:self.dst_zone[2] + 1,
                      self.dst_zone[1]:self.dst_zone[3] + 1]
                      
    
     
    def sim(self):
        
        imageA = self.getSrcZoneFromSrc()
        imageB = self.getDstZoneFromSrc()
        
        #cv_show("img", imageA)
        #cv_show("img", imageB)
        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        #min_max        
        grayA = minMax(grayA)
        grayB = minMax(grayB)
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        
        #np.set_printoptions(threshold=sys.maxsize)        
        diff = (diff * 255).astype("uint8")        
        txt = "SSIM: {:.3f}".format(score)
        
        image = cv2.rectangle(self.src_img,
                              (self.src_zone[1], self.src_zone[0]), 
                              (self.src_zone[3], self.src_zone[2]), 
                              (255, 0, 0), 2)
        
        image = cv2.putText(image, txt, 
                            (self.src_zone[1], self.src_zone[0]), 
                            cv2.FONT_HERSHEY_COMPLEX, 1., (0, 0, 0), 2)
        
                 
        
        mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY_INV)[1]
        points = mask2point(mask, 5)    
        if points is None: return 
        
        for id, point in enumerate(points):
        
            point[:,:,0] = point[:,:,0] + self.src_zone[1]
            point[:,:,1] = point[:,:,1] + self.src_zone[0]        
            cv2.polylines(image, [point], isClosed=True, 
                          color=(id*30, id*30, id*30), thickness=2)
            (x, y, w, h) = cv2.boundingRect(point)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image
        
 
def img_check(src, dst):
    obj = ImgMerge()
    obj.setSrc(src)
    obj.setDst(dst)
    obj.matchKeyPoints()
    img = obj.sim()
    cv_show("img", img)      

def img_merge(src, dst):

    obj = ImgMerge()
    obj.setSrc(src)    
    obj.setDst(dst)
    
    obj.matchKeyPoints()
    obj.drawMatches()
    res = obj.stich()
    cv_show("result", res)
    
def dir_merge(path):

    filelist = os.listdir(path)
    filelist.sort()
    
    obj = ImgMerge()
    
    img = cv2.imread(os.path.join(path, filelist[0]))
    print(img.shape)
    obj.setSrc(img)
    
    for id, x in enumerate(filelist[1:]):
        print(id, x)
        img = cv2.imread(os.path.join(path, x))
        print(img.shape)        
        obj.setDst(img)
        obj.matchKeyPoints(ratio=0.75, reprojThresh=4.0)
        #obj.drawMatches()
        res = obj.stich()
        obj.setSrc(res)        
    cv2.imwrite("img/save9.jpg", res)         
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 0)')    
    args = parser.parse_args()
    
    if args.m == 0:
        src = cv2.imread('src/001_001.png')
        print(src.shape)
        src = cv2.resize(src, (src.shape[1]//2, src.shape[0]//2))   
        cv_show("src", src)    
        
        dst = cv2.imread('src/001_002.png')
        print(dst.shape)
        dst = cv2.resize(dst, (dst.shape[1]//2, dst.shape[0]//2))   
        cv_show("src", dst)                
        img_merge(src, dst)
        
    elif args.m == 1:
        src = cv2.imread('img/save8.jpg')
        print(src.shape)
        src = cv2.resize(src, (src.shape[1]//4, src.shape[0]//4))   
        cv_show("src", src)    
        
        dst = cv2.imread('img/test1.png')
        print(dst.shape)
        dst = cv2.resize(dst, (dst.shape[1]//8, dst.shape[0]//8))   
        cv_show("src", dst)
        img_check(src, dst)
    elif args.m == 2:
        dir_merge('./src')
    
    