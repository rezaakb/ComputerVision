import cv2
import numpy as np

from math import floor, log
import cv2
import numpy as np
import matplotlib.pyplot as plt

scale_percent = 30

img1 = cv2.imread('Building1.JPG')
img1 = cv2.resize(img1, ((img1.shape[1] * scale_percent) // 100,(img1.shape[0] * scale_percent) // 100), interpolation = cv2.INTER_AREA)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('Building2.JPG')
img2 = cv2.resize(img2, ((img2.shape[1] * scale_percent) // 100,(img2.shape[0] * scale_percent) // 100), interpolation = cv2.INTER_AREA)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

n=img1.shape[0]
m=img1.shape[1]+img2.shape[1]+20

sift = cv2.xfeatures2d.SIFT_create()

kp1, d1 = sift.detectAndCompute(img1_gray, None)
kp2, d2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(d1,d2, k=2)

good = []
points1 = []
points2 = []
for a,b in matches:
    if a.distance/b.distance < 0.75:
        good.append([a.distance/b.distance,a.queryIdx,a.trainIdx])
        points1.append(kp1[a.queryIdx].pt)
        points2.append(kp2[a.trainIdx].pt)

points1 = np.int32(points1)
points2 = np.int32(points2)

F, mask = cv2.findFundamentalMat(points1,points2,method=cv2.FM_RANSAC)

points1 = points1[mask.ravel()==1]
points2 = points2[mask.ravel()==1]
'''
res1 = np.copy(img1)
res2 = np.copy(img2)

r2_1 = np.ones((n,m,3))*255

for i in range(len(points1)):
    res1=cv2.circle(res1,(points1[i,0],points1[i,1]), 3, (0, 255, 0), thickness=2)
    res2=cv2.circle(res2,(points2[i,0],points2[i,1]), 3, (0, 255, 0), thickness=2)


r2_1[:,0:img1.shape[1],:] = res1[:,:,:]
r2_1[floor(n/2-img2.shape[0]/2):floor(n/2+img2.shape[0]/2),img1.shape[1]+20:img1.shape[1]+img2.shape[1]+20,:] = res2[:,:,:]


cv2.imwrite('outputs/r2_1.jpg',r2_1)
'''
def interSectionTwoLine(x1,x2):
    a = x1[1]*x2[2] - x1[2]*x2[1]
    b = x1[2]*x2[0] - x1[0]*x2[2]
    c = x1[0]*x2[1] - x1[1]*x2[0]
    return [a/c, b/c, 1]


I1 = np.matmul(F,[points1[5][0],points1[5][1],1])
I1 = I1/I1[2]
I2 = np.matmul(F,[points1[1][0],points1[1][1],1])
I2 = I2/I2[2]
e2 = interSectionTwoLine(I1,I2)

I1 = np.matmul(F.T,[points2[5][0],points2[5][1],1])
I1 = I1/I1[2]

I2 = np.matmul(F.T,[points2[1][0],points2[1][1],1])
I2 = I2/I2[2]

e1 = interSectionTwoLine(I1,I2)

print('e1 = ', e1[0:2])
print('e2 = ', e2[0:2])

res1 = np.copy(img1)
res2 = np.copy(img2)

for i in range(10):
    res1=cv2.circle(res1,(points1[i,0],points1[i,1]), 3, (0, 255, 0), thickness=2)
    res2=cv2.circle(res2,(points2[i,0],points2[i,1]), 3, (0, 255, 0), thickness=2)
    cv2.line(res1,(points1[i,0],points1[i,1]),tuple(map(int, e1[0:2])),(0, 255, 0), thickness=2)
    cv2.line(res2,(points2[i,0],points2[i,1]),tuple(map(int, e2[0:2])),(0, 255, 0), thickness=2)

cv2.imwrite('r2_2.jpg', res1)
cv2.imwrite('r2_3.jpg', res2)

'''
res1 = np.copy(img1)
res2 = np.copy(img2)

r2_2 = np.ones((n,m,3))*255

for i in range(10):
    res1=cv2.circle(res1,(points1[i,0],points1[i,1]), 3, (0, 255, 0), thickness=2)
    res2=cv2.circle(res2,(points2[i,0],points2[i,1]), 3, (0, 255, 0), thickness=2)
r2_2[:,0:img1.shape[1],:] = res1[:,:,:]
r2_2[floor(n/2-img2.shape[0]/2):floor(n/2+img2.shape[0]/2),img1.shape[1]+20:img1.shape[1]+img2.shape[1]+20,:] = res2[:,:,:]

for i in range(10):
    r2_2 = cv2.line(r2_2, (points1[i,0],points1[i,1]),
                   (img1.shape[1]+20+points2[i,0],points2[i,1]+floor(n/2-img2.shape[0]/2)),
                   (0, 255, 0), thickness=2)
cv2.imwrite('outputs/r2_2.jpg',r2_2)
'''