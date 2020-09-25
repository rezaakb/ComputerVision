import cv2
import numpy as np
from scipy import signal


#tashkile tabe gauss
def gauss(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x +y*y) / (2. * sigma * sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h

#moshtaghe ofoghi ya amoodi migirad
def gradientGauss(x, i):
    a = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4.0
    if i == 0:
        return signal.convolve2d(x, np.transpose(a), boundary='symm', mode='same')
    else:
        return signal.convolve2d(x, a, boundary='symm', mode='same')

g=gauss((10,10),3)
gx = gradientGauss(g,0)
gy = gradientGauss(g,1)
scale_percent=50
r=7
k=0.06
nesbat =0.6



img1 = cv2.imread('01.JPG')
img1 = cv2.resize(img1, ((img1.shape[1] * scale_percent) // 100,(img1.shape[0] * scale_percent) // 100), interpolation = cv2.INTER_AREA)
img1_b_x = signal.convolve2d(img1[:,:,0], gx, boundary='symm', mode='same')
img1_g_x = signal.convolve2d(img1[:,:,1], gx, boundary='symm', mode='same')
img1_r_x = signal.convolve2d(img1[:,:,2], gx, boundary='symm', mode='same')
img1_x = np.maximum(img1_b_x,img1_g_x,img1_r_x)

img1_b_y = signal.convolve2d(img1[:,:,0], gy, boundary='symm', mode='same')
img1_g_y = signal.convolve2d(img1[:,:,1], gy, boundary='symm', mode='same')
img1_r_y = signal.convolve2d(img1[:,:,2], gy, boundary='symm', mode='same')
img1_y = np.maximum(img1_b_y,img1_g_y,img1_r_y)

img1_x2 = img1_x *img1_x
img1_y2 = img1_y * img1_y
img1_xy = img1_y *img1_x

cv2.imwrite('r01_grad.jpg', np.sqrt(img1_x2 + img1_y2))

s1_x2 = signal.convolve2d(img1_x2, g, boundary='symm', mode='same')
s1_y2 = signal.convolve2d(img1_y2, g, boundary='symm', mode='same')
s1_xy = signal.convolve2d(img1_xy, g, boundary='symm', mode='same')

harris1 = (s1_x2*s1_y2 - s1_xy) - k*(s1_x2+s1_y2)
tmp = ((harris1-np.min(harris1)) / (np.max(harris1)-np.min(harris1)))*255
cv2.imwrite('r03_score.jpg', tmp)

thresh1 = cv2.threshold(tmp,2.5,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('r05_thresh.jpg', thresh1)
thresh1= np.uint8(thresh1)
num_labels, labels_im = cv2.connectedComponents(thresh1)
harris1_dots = np.zeros_like(harris1)

feature1 = np.zeros((num_labels-1,r*r*3))
r07_harris = np.zeros_like(img1)
r07_harris[:,:,:] = img1[:,:,:]

for i in range(1,num_labels):
    maxValue = np.max(harris1[labels_im==i])
    index = np.where(harris1==maxValue)
    index_i = index[0][0]
    index_j = index[1][0]
    harris1_dots[index_i,index_j] = i
    r07_harris= cv2.circle(r07_harris,(index_j,index_i),6,(255,0,0),thickness=2)
    if (index_j<r//2+1):
        index_j = r//2+1
    if (index_i < r // 2 + 1):
        index_i = r // 2 + 1
    if (img1.shape[0]-index_i< r // 2):
        index_i =  img1.shape[0] - r // 2
    if (img1.shape[1]-index_j< r // 2 ):
        index_j =  img1.shape[1] - r // 2
    tmp = img1[index_i-r//2-1:index_i+r//2,index_j-r//2-1:index_j+r//2,:].flatten()
    tmp1 = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)))
    feature1[i-1,:] =tmp1[:]

cv2.imwrite('r07_harris.jpg', r07_harris)

img2 = cv2.imread('02.JPG')
img2 = cv2.resize(img2, ((img2.shape[1] * scale_percent) // 100,(img2.shape[0] * scale_percent) // 100), interpolation = cv2.INTER_AREA)
img2_b_x = signal.convolve2d(img2[:,:,0], gx, boundary='symm', mode='same')
img2_g_x = signal.convolve2d(img2[:,:,1], gx, boundary='symm', mode='same')
img2_r_x = signal.convolve2d(img2[:,:,2], gx, boundary='symm', mode='same')
img2_x = np.maximum(img2_b_x,img2_g_x,img2_r_x)

img2_b_y = signal.convolve2d(img2[:,:,0], gy, boundary='symm', mode='same')
img2_g_y = signal.convolve2d(img2[:,:,1], gy, boundary='symm', mode='same')
img2_r_y = signal.convolve2d(img2[:,:,2], gy, boundary='symm', mode='same')
img2_y = np.maximum(img2_b_y,img2_g_y,img2_r_y)

img2_x2 = img2_x *img2_x
img2_y2 = img2_y * img2_y
img2_xy = img2_y *img2_x

cv2.imwrite('r02_grad.jpg', np.sqrt(img2_x2 + img2_y2))

s2_x2 = signal.convolve2d(img2_x2, g, boundary='symm', mode='same')
s2_y2 = signal.convolve2d(img2_y2, g, boundary='symm', mode='same')
s2_xy = signal.convolve2d(img2_xy, g, boundary='symm', mode='same')

harris2 = (s2_x2*s2_y2 - s2_xy) - k*(s2_x2+s2_y2)
tmp = ((harris2-np.min(harris2)) / (np.max(harris2)-np.min(harris2)))*255
cv2.imwrite('r04_score.jpg', tmp)

thresh2 = cv2.threshold(tmp,2.5,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('r06_thresh.jpg', thresh2)
thresh2= np.uint8(thresh2)
num_labels, labels_im = cv2.connectedComponents(thresh2)
harris2_dots = np.zeros_like(harris2)

feature2 = np.zeros((num_labels-1,r*r*3))
r08_harris = np.zeros_like(img2)
r08_harris[:,:,:] = img2[:,:,:]

for i in range(1,num_labels):
    maxValue = np.max(harris2[labels_im==i])
    index = np.where(harris2==maxValue)
    index_i = index[0][0]
    index_j = index[1][0]
    harris2_dots[index_i,index_j] = i
    r08_harris= cv2.circle(r08_harris,(index_j,index_i),6,(255,0,0),thickness=2)
    if (index_j<r//2+1):
        index_j = r//2+1
    if (index_i < r // 2 + 1):
        index_i = r // 2 + 1
    if (img2.shape[0]-index_i< r // 2 ):
        index_i =  img2.shape[0] - r // 2
    if (img2.shape[1]-index_j< r // 2):
        index_j =  img2.shape[1]- r // 2
    tmp = img2[index_i-r//2-1:index_i+r//2,index_j-r//2-1:index_j+r//2,:].flatten()
    tmp1 = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)))
    feature2[i-1,:] =tmp1[:]

cv2.imwrite('r08_harris.jpg', r08_harris)


n=len(feature1)
m=len(feature2)
A = np.zeros((n,m))
f1 = np.zeros((n,m))
f2 = np.zeros((n,m))

list = []

for i in range(n):
    minDistance = 10000000
    p1 = 0
    p2 = 0
    for j in range(m):
        A[i,j]= np.sum((feature1[i,:] - feature2[j,:])**2)
        if A[i,j]<minDistance:
            p2=p1
            p1=j
            minDistance=A[i,j]
    if A[i,p1]/A[i,p2] < nesbat:
        list.append(p1)
        f1[i,p1]=1

print(len(np.where(f1==1)[0]))

for j in range(m):
    minDistance = 10000000
    p1 = 0
    p2 = 0
    for i in range(n):
        if A[i,j]<minDistance:
            p2=p1
            p1=i
            minDistance=A[i,j]
    if A[p1,j]/A[p2,j] < nesbat and f1[p1,j]==1:
        f2[p1,j]=1

print(len(np.where(f2==1)[0]))


for i in range(m):
    if (len(np.where(f2[:,i]==1)[0])>1):
        f2[:,i]=0

print(len(np.where(f2==1)[0]))


r09_corres = np.zeros_like(img1)
r09_corres[:,:,:] = img1[:,:,:]
r10_corres = np.zeros_like(img2)
r10_corres[:,:,:] = img2[:,:,:]
r11 = np.ones((img1.shape[0],(img1.shape[1]+img2.shape[1])+10,3))*255
r11[:,0:img1.shape[1],:]=img1[:,:,:]
r11[:,10+img1.shape[1]:10+img1.shape[1]+img2.shape[1],:]=img2[:,:,:]


best_lines = []
for i in range(n):
    tmp=np.where(f2[i]==1)[0]
    if (len(tmp)>0):
        index1 = np.where(harris1_dots == i+1)
        index1_i = index1[0][0]
        index1_j = index1[1][0]
        r09_corres = cv2.circle(r09_corres, (index1_j, index1_i), 6, (255, 0, 0), thickness=2)
        index2 = np.where(harris2_dots == tmp[0]+1)
        index2_i = index2[0][0]
        index2_j = index2[1][0]
        r10_corres = cv2.circle(r10_corres, (index2_j, index2_i), 6, (255, 0, 0), thickness=2)
        best_lines.append([abs(harris1[index1_i,index1_j]-harris2[index2_i,index2_j]),index1_i,index1_j,index2_i,index2_j])

best_lines.sort()
for i in range(min(len(best_lines),10)):
    r11 = cv2.circle(r11, (best_lines[i][2], best_lines[i][1]), 6, (255, 0, 0), thickness=2)
    r11 = cv2.circle(r11, (best_lines[i][4] + 10 + img1.shape[1], best_lines[i][3]), 6, (255, 0, 0), thickness=2)
    r11 = cv2.line(r11, (best_lines[i][2], best_lines[i][1]), (best_lines[i][4] + 10 + img1.shape[1], best_lines[i][3]), (0, 255, 0),thickness=2)

cv2.imwrite('r09_corres.jpg', r09_corres)
cv2.imwrite('r10_corres.jpg', r10_corres)
cv2.imwrite('r11.jpg', r11)









