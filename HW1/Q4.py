from math import floor, log
import cv2
import numpy as np

scale_percent = 30

img1 = cv2.imread('input/11.JPG')
img1 = cv2.resize(img1, ((img1.shape[1] * scale_percent) // 100,(img1.shape[0] * scale_percent) // 100), interpolation = cv2.INTER_AREA)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('input/12.JPG')
img2 = cv2.resize(img2, ((img2.shape[1] * scale_percent) // 100,(img2.shape[0] * scale_percent) // 100), interpolation = cv2.INTER_AREA)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, d1 = sift.detectAndCompute(img1_gray, None)
kp2, d2 = sift.detectAndCompute(img2_gray, None)

n=img1.shape[0]
m=img1.shape[1]+img2.shape[1]+20
r14_corners = np.ones((n,m,3))*255
tmp1 = np.zeros_like(img1)
tmp1[:,:,:] = img1[:,:,:]
cv2.drawKeypoints(img1,kp1,tmp1,color=(0,255,0), flags=0)
r14_corners[:,0:img1.shape[1],:] = tmp1[:,:,:]
tmp2 = np.ones_like(img2)*255
tmp2[:,:,:] = img2[:,:,:]
cv2.drawKeypoints(img2,kp2,tmp2,color=(0,255,0), flags=0)
r14_corners[floor(n/2-img2.shape[0]/2):floor(n/2+img2.shape[0]/2),img1.shape[1]+20:img1.shape[1]+img2.shape[1]+20,:] = tmp2[:,:,:]

cv2.imwrite('output/r14_corners.jpg',r14_corners)

bf = cv2.BFMatcher()
matches = bf.knnMatch(d1,d2, k=2)

good = []
for a,b in matches:
    if a.distance/b.distance < 0.75:
        good.append([a.distance/b.distance,a.queryIdx,a.trainIdx])

tmp3 = np.zeros_like(tmp1)
tmp3[:,:,:] = tmp1[:,:,:]
tmp4 = np.zeros_like(img2)
tmp4[:,:,:] = tmp2[:,:,:]

r15_corres = np.ones((n,m,3))*255

for d,i,j in good:
    tmp3=cv2.circle(tmp3,(int(kp1[i].pt[0]),int(kp1[i].pt[1])), 3, (255, 0, 0), thickness=2)
    tmp4=cv2.circle(tmp4, (int(kp2[j].pt[0]),int(kp2[j].pt[1])), 3, (255, 0, 0), thickness=2)

r15_corres[:,0:img1.shape[1],:] = tmp3[:,:,:]
r15_corres[floor(n/2-img2.shape[0]/2):floor(n/2+img2.shape[0]/2),img1.shape[1]+20:img1.shape[1]+img2.shape[1]+20,:] = tmp4[:,:,:]

cv2.imwrite('output/r15_correspondencess.jpg',r15_corres)

r16_SIFT = r15_corres

for d,i,j in good:
    r16_SIFT = cv2.line(r16_SIFT, (int(kp1[i].pt[0]), int(kp1[i].pt[1])),
                        (img1.shape[1] + 20 + int(kp2[j].pt[0]), floor(n / 2 - img2.shape[0] / 2) +  int(kp2[j].pt[1])),
                        (255, 0, 0))

cv2.imwrite('output/r16_SIFT.jpg',r16_SIFT)


tmp1 = np.zeros_like(img1)
tmp1[:,:,:] = img1[:,:,:]
tmp2 = np.zeros_like(img2)
tmp2[:,:,:] = img2[:,:,:]

counter =0
for d,i,j in good:
    if counter==20:
        break
    counter=counter+1
    tmp1 = cv2.circle(tmp1, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), 3, (255, 0, 0), thickness=2)
    tmp2 = cv2.circle(tmp2, (int(kp2[j].pt[0]), int(kp2[j].pt[1])), 3, (255, 0, 0), thickness=2)


r17 = np.ones((n,m,3))*255
r17[:,0:img1.shape[1],:] = tmp1[:,:,:]
r17[floor(n/2-img2.shape[0]/2):floor(n/2+img2.shape[0]/2),img1.shape[1]+20:img1.shape[1]+img2.shape[1]+20,:] = tmp2[:,:,:]

counter =0
for d,i,j in good:
    if counter==20:
        break
    counter=counter+1
    r17 = cv2.line(r17, (int(kp1[i].pt[0]), int(kp1[i].pt[1])),
                        (img1.shape[1] + 20 + int(kp2[j].pt[0]), floor(n / 2 - img2.shape[0] / 2) +  int(kp2[j].pt[1])),
                        (255, 0, 0),thickness=2)


cv2.imwrite('output/r17.jpg',r17)

w=0
counter=0
N=10000000
w_min = 0
p=0.99
s=4
t=20
max_inliers=-1000000
iteration=0
while N > counter:
    points = np.random.randint(len(good), size=4)
    counter= counter+1
    h,status = cv2.findHomography(np.array(
        [kp1[good[points[0]][1]].pt, kp1[good[points[1]][1]].pt, kp1[good[points[2]][1]].pt,
         kp1[good[points[3]][1]].pt]),
                           np.array(
                               [kp2[good[points[0]][2]].pt, kp2[good[points[1]][2]].pt, kp2[good[points[2]][2]].pt,
                                kp2[good[points[3]][2]].pt]),method=0)
    if status[0]==0:
        continue
    inliers_num = 0
    for d, i, j in good:
        x = np.array([kp1[i].pt[0], kp1[i].pt[1], 1]).reshape((3, 1))
        x_prime = np.matmul(h, x)

        y = np.array([kp2[j].pt[0], kp2[j].pt[1], 1]).reshape((3, 1))
        d = np.sqrt(np.sum((y - x_prime/x_prime[2]) ** 2))
        if (d < t):
            inliers_num = inliers_num + 1
    if inliers_num > max_inliers:
        h_final = h[:,:]
        max_inliers = inliers_num
        w= max_inliers/len(good)

    if w > w_min:
        w_min = w
        N = log(1 - p) // log(1 - w ** s)


last_inliers_num=max_inliers
h=h_final[:,:]

inliers_src = []
inliers_dst = []

for d, i, j in good:
    x = np.array([kp1[i].pt[0], kp1[i].pt[1], 1]).reshape((3, 1))
    x_prime = np.matmul(h, x)
    y = np.array([kp2[j].pt[0], kp2[j].pt[1], 1]).reshape((3, 1))
    d = np.sqrt(np.sum((y - x_prime/x_prime[2]) ** 2))
    if (d < t):
        inliers_src.append(kp1[i].pt)
        inliers_dst.append(kp2[j].pt)


while True:
    h,status = cv2.findHomography(np.array(inliers_src), np.array(inliers_dst),method=0)
    inliers_num = 0
    inliers_src = []
    inliers_dst = []
    for d, i, j in good:
        x = np.array([kp1[i].pt[0], kp1[i].pt[1], 1]).reshape((3, 1))
        x_prime = np.matmul(h, x)
        y = np.array([kp2[j].pt[0], kp2[j].pt[1], 1]).reshape((3, 1))
        d = np.sqrt(np.sum((y - x_prime/x_prime[2]) ** 2))
        if (d < t):
            inliers_num = inliers_num + 1
            inliers_src.append(kp1[i].pt)
            inliers_dst.append(kp2[j].pt)
    if inliers_num == last_inliers_num or iteration > 10 :
        print(iteration)
        break
    last_inliers_num = inliers_num
    iteration=iteration+1



tmp1 = np.zeros_like(tmp3)
tmp1[:,:,:] = tmp3[:,:,:]
tmp2 = np.zeros_like(tmp4)
tmp2[:,:,:] = tmp4[:,:,:]

for i,j in inliers_src:
    i = int(i)
    j = int(j)
    tmp1=cv2.circle(tmp1,(i,j), 3, (0, 0, 255), thickness=2)
for i, j in inliers_dst:
    i = int(i)
    j = int(j)
    tmp2=cv2.circle(tmp2, (i,j), 3, (0, 0, 255), thickness=2)

r18 = np.ones((n,m,3))*255
r18[:,0:img1.shape[1],:] = tmp1[:,:,:]
r18[floor(n/2-img2.shape[0]/2):floor(n/2+img2.shape[0]/2),img1.shape[1]+20:img1.shape[1]+img2.shape[1]+20,:] = tmp2[:,:,:]

cv2.imwrite('output/r18.jpg',r18)

r20 = cv2.warpPerspective(img1, h,(img2.shape[1]*2,img2.shape[0]*2))
cv2.imwrite('output/r20.jpg',r20[0:1025,0:1668,:])
