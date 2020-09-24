import numpy as np
import cv2

def lineIntersectionWith2Point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def IntersectionBetweenPointandLine(m1,b1, line2) :
   m2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
   b2 = line2[0][1] - m2 * line2[0][0]
   x = (b2 - b1) / (m1 - m2)
   y = m1 * x + b1
   return [x,y]


def distance(p1,p2):
    return np.sqrt((p1[1]-p2[1])**2+(p1[0]-p2[0])**2)

img = cv2.imread('input/13.jpg')
points = [[688, 1073], [2153, 558], [831, 3273], [2590, 3194],[2286,1323],[2351,1376],[2513,2646],[2569,2640]]
B = lineIntersectionWith2Point((points[0], points[2]), (points[1], points[3]))
A = lineIntersectionWith2Point((points[0], points[1]), (points[2], points[3]))
C = lineIntersectionWith2Point((points[4], points[5]), (points[6], points[7]))

D = lineIntersectionWith2Point((points[0], points[3]), (B, A))

tmpb = (A[0]-C[0])/(A[1]-C[1]) *B[0] + B[1]
BH = IntersectionBetweenPointandLine(-(A[0]-C[1])/(A[1]-C[1]),tmpb,(C,A))

tmpb = (B[0]-A[0])/(B[1]-A[1]) *C[0] + C[1]
CH = IntersectionBetweenPointandLine(-(B[0]-A[0])/(B[1]-A[1]),tmpb,(B,A))

H = lineIntersectionWith2Point((CH, C), (BH, B))
f=np.sqrt(distance(B,CH)*distance(A,CH)-distance(H,CH)**2)
K=np.array([[f,0,H[0]],[0,f,H[1]],[0,0,1]])

v1 = np.array([D[0],D[1],1])
v2 =np.array([B[0],B[1],1])

w = np.dot(K,K.T)
w = np.linalg.inv(w)
tmp1=np.dot( v1,np.dot(w,v1))
tmp2=np.dot( v2,np.dot(w,v2))
tmp3 =np.sqrt(tmp1*tmp2)
tmp4= np.dot( v1,np.dot(w,v2))
ratio = np.tan(np.arccos(tmp4/tmp3))
print(ratio)

x =1000
y =  int(ratio*x)
s = np.array(points)
h, status = cv2.findHomography(np.array(
    [[points[0][1], points[0][0]], [points[1][1], points[1][0]], [points[2][1], points[2][0]],
     [points[3][1], points[3][0]]]), np.array([[y - 1, 0], [0, 0], [y - 1, x - 1], [0, x - 1]]))
im_dst = cv2.warpPerspective(img, h, (y, x))
cv2.imwrite('output/r21.jpg', im_dst)


img = cv2.imread('output/r20.jpg')
points =[[4,1174],[ 175,233], [885,1202], [765,120],[11,1365],[79,1656],[255,1383],[293,1645]]
B = lineIntersectionWith2Point((points[0], points[2]), (points[1], points[3]))
A = lineIntersectionWith2Point((points[0], points[1]), (points[2], points[3]))
C = lineIntersectionWith2Point((points[4], points[5]), (points[6], points[7]))

D = lineIntersectionWith2Point((points[1], points[2]), (B, A))


tmpb = (A[0]-C[0])/(A[1]-C[1]) *B[0] + B[1]
BH = IntersectionBetweenPointandLine(-(A[0]-C[1])/(A[1]-C[1]),tmpb,(C,A))

tmpb = (B[0]-A[0])/(B[1]-A[1]) *C[0] + C[1]
CH = IntersectionBetweenPointandLine(-(B[0]-A[0])/(B[1]-A[1]),tmpb,(B,A))

H = lineIntersectionWith2Point((CH, C), (BH, B))
f=np.sqrt(distance(B,CH)*distance(A,CH)-distance(H,CH)**2)

K=np.array([[f,0,H[0]],[0,f,H[1]],[0,0,1]])

v1 = np.array([D[0],D[1],1])
v2 =np.array([B[0],B[1],1])

w = np.dot(K,K.T)
w = np.linalg.inv(w)
tmp1=np.dot(v1,np.dot(w,v1))
tmp2=np.dot(v2,np.dot(w,v2))
tmp3 =np.sqrt(tmp1*tmp2)
tmp4= np.dot(v1,np.dot(w,v2))
ratio = np.tan(np.arccos(tmp4/tmp3))
print(ratio)
x =1000
y =  int(ratio*x)
s = np.array(points)
h, status = cv2.findHomography(np.array(
    [[points[0][1], points[0][0]], [points[1][1], points[1][0]], [points[2][1], points[2][0]],
     [points[3][1], points[3][0]]]), np.array([[y - 1, 0], [0, 0], [y - 1, x - 1], [0, x - 1]]))
im_dst = cv2.warpPerspective(img, h, (y, x))
cv2.imwrite('output/r22.jpg', im_dst)

