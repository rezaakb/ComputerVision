from math import log
import cv2
import numpy as np

def RANSAC(good,kp1,kp2):
    w = 0
    counter = 0
    N = 10000000
    w_min = 0
    p = 0.99
    s = 4
    t = 20
    max_inliers = -1000000
    iteration = 0
    while N > counter:
        points = np.random.randint(len(good), size=4)
        counter = counter + 1
        h, status = cv2.findHomography(np.array(
            [kp1[good[points[0]][1]].pt, kp1[good[points[1]][1]].pt, kp1[good[points[2]][1]].pt,
             kp1[good[points[3]][1]].pt]),
            np.array(
                [kp2[good[points[0]][2]].pt, kp2[good[points[1]][2]].pt, kp2[good[points[2]][2]].pt,
                 kp2[good[points[3]][2]].pt]), method=0)
        # h_inverse = np.linalg.inv(np.array(h))
        if status[0] == 0:
            continue
        inliers_num = 0
        for d, i, j in good:
            x = np.array([kp1[i].pt[0], kp1[i].pt[1], 1]).reshape((3, 1))
            x_prime = np.matmul(h, x)

            y = np.array([kp2[j].pt[0], kp2[j].pt[1], 1]).reshape((3, 1))
            # y_prime = np.matmul(h_inverse, y)

            d = np.sqrt(np.sum((y - x_prime / x_prime[2]) ** 2))  # + np.sqrt(np.sum((x - y_prime) ** 2))

            if (d < t):
                inliers_num = inliers_num + 1
        if inliers_num > max_inliers:
            h_final = h[:, :]
            max_inliers = inliers_num
            w = max_inliers / len(good)

        if w > w_min:
            w_min = w
            if w==1:
                break
            N = log(1 - p) // log(1 - w ** s)

    last_inliers_num = max_inliers
    h = h_final[:, :]

    inliers_src = []
    inliers_dst = []

    for d, i, j in good:
        x = np.array([kp1[i].pt[0], kp1[i].pt[1], 1]).reshape((3, 1))
        x_prime = np.matmul(h, x)
        y = np.array([kp2[j].pt[0], kp2[j].pt[1], 1]).reshape((3, 1))
        d = np.sqrt(np.sum((y - x_prime / x_prime[2]) ** 2))
        if (d < t):
            inliers_src.append(kp1[i].pt)
            inliers_dst.append(kp2[j].pt)

    while True:
        h, status = cv2.findHomography(np.array(inliers_src), np.array(inliers_dst), method=0)
        inliers_num = 0
        inliers_src = []
        inliers_dst = []
        for d, i, j in good:
            x = np.array([kp1[i].pt[0], kp1[i].pt[1], 1]).reshape((3, 1))
            x_prime = np.matmul(h, x)
            y = np.array([kp2[j].pt[0], kp2[j].pt[1], 1]).reshape((3, 1))
            # y_prime = np.matmul(h_inverse, y)
            d = np.sqrt(np.sum((y - x_prime / x_prime[2]) ** 2))  # + np.sqrt(np.sum((x - y_prime) ** 2))
            if (d < t):
                inliers_num = inliers_num + 1
                inliers_src.append(kp1[i].pt)
                inliers_dst.append(kp2[j].pt)
        if inliers_num == last_inliers_num or iteration > 10:
            print(iteration,inliers_num)
            break
        last_inliers_num = inliers_num
        iteration = iteration + 1
    return h

filenames=['input/04.JPG','input/03.JPG','input/05.JPG','input/06.JPG',
           'input/07.JPG','input/08.JPG','input/09.JPG','input/10.JPG']
base = np.zeros((828*3,1104*5,3),dtype='uint8')
img = np.zeros((8,828,1104,3),dtype='uint8')
kp =[]
d = []
sift = cv2.xfeatures2d.SIFT_create()

for i in range(8):
    tmp = cv2.imread(filenames[i])
    if (i==0):
        base[828:828 * 2, 1500:1500 +1104, :] = tmp[:, :, :]
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    else:
        img[i, :, :, :]= tmp[:,:,:]
        gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    keypoints, discriptor = sift.detectAndCompute(gray, None)
    kp.append(keypoints)
    d.append(discriptor)

resault = img[0,:,:,:]

list_matched = []
list_unmatched = []
bf = cv2.BFMatcher()

for i in range(1,8):
    matches = bf.knnMatch(d[i],d[0], k=2)
    good = []
    for a,b in matches:
        if a.distance < 0.6*b.distance:
             good.append([a.distance/b.distance,a.queryIdx,a.trainIdx])
    if (len(good)>200):
        h=RANSAC(good,kp[i],kp[0])
        list_matched.append((i,h))
    else:
        list_unmatched.append(i)

for i in list_unmatched:
    max_good_len = -1000000
    max_good = []
    max_h=[]
    p=0
    for k,h in list_matched:
        matches = bf.knnMatch(d[i], d[k], k=2)
        good = []
        for a, b in matches:
            if a.distance/b.distance < 0.7:
                good.append([a.distance / b.distance, a.queryIdx, a.trainIdx])
        if (len(good)>max_good_len):
            p=k
            max_good = good
            max_h = h
            max_good_len=len(good)
    h1 = RANSAC(max_good, kp[i], kp[p])
    h_final = np.dot(max_h,h1)
    list_matched.append((i,h_final))


def stiching(img1,img2,h):
    tmp1=img1
    tmp2 = cv2.warpPerspective(img2,h,(tmp1.shape[1],tmp1.shape[0]))
    mask1 = np.ones_like(tmp1,dtype='uint8')
    mask2 = np.ones_like(tmp1,dtype='uint8')
    mask1[tmp1==0] = 0
    mask2[tmp2==0] = 0
    mask_final = mask1 *mask2
    finalpicture =  (tmp1+ tmp2)*(1-mask_final) + (tmp1*(0.5)+ tmp2*(0.5)) *mask_final
    return finalpicture


for i,h in list_matched:
    base = stiching(base,img[i],h)

cv2.imwrite('r13.jpg', base[482:1655, 350:4121, :])
