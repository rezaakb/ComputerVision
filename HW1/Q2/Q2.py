import numpy as np
import cv2

def makeRotation(theta1,theta2,theta3):
    c = np.cos(theta1)
    s = np.sin(theta1)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c = np.cos(theta2)
    s = np.sin(theta2)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c = np.cos(theta3)
    s = np.sin(theta3)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R=np.dot(Ry,Rx)
    R=np.dot(Rz,R)
    return R

#in tabe ra az jaie copy kardam
def equationPlane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    norm = np.sqrt(a**2+b**2+c**2)
    return a/norm,b/norm,c/norm,d

img = cv2.imread('14.png')
Px = img.shape[0]/2
Py = img.shape[1]/2
distance = np.sqrt(40**2-25**2)
zamin =20

theta2 =np.arctan(Px/500)
theta = np.arctan((distance-zamin)/25) + theta2

vatar2 = np.sqrt((distance+zamin)**2+25**2)
vatar3 = np.sqrt(500**2+Py**2)

vatar6=np.sqrt(25**2+(distance-zamin)**2)
a,b,c,d = equationPlane(0,np.sin(theta2)*vatar6,np.cos(theta2)*vatar6
                        ,0,-np.sin(theta2)*vatar2,np.cos(theta2)*vatar2,
                        -Px*(vatar2/vatar3),-np.sin(theta2)*vatar2,np.cos(theta2)*vatar2)
P2x =400
P2y = 800

theta3 = np.pi/2 - theta
C = np.array([0,np.sin(theta3)*distance,-np.cos(theta3)*distance])
print(C)
R=makeRotation(theta,0,0)
t = -1*np.dot(R,C)
K=np.array([[500,0,Px],[0,500,Py],[0,0,1]])
K_inverse  = np.linalg.inv(K)
K_2=np.array([[500,0,P2x],[0,500,P2y],[0,0,1]])
h = np.dot(np.dot(K_2,R-(np.dot(t,np.array([a,b,c]))/d)),K_inverse)
print(h)
cv2.imwrite('r12.jpg', cv2.warpPerspective(img, h, (2 * P2x, 2 * P2y)))