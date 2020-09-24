import cv2
import numpy as np
import matplotlib.pyplot as plt

def initialization():
    num = 20
    images = np.zeros((num,1000,1500,3))
    images = images.astype('uint8')
    for i in range(1,21):
        if i<10:
            name_file = 'inputs/im0' + str(i)+'.jpg'
        else:
            name_file = 'inputs/im'+str(i)+'.jpg'
        images[i-1,:,:,:]=cv2.imread(name_file)
    objpoints = []
    imgpoints = []
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*22
    objp = objp.reshape(-1, 1, 3)
    px = 750
    py = 500

    for i in range(len(images)):
        objpoints.append(objp)
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))
        imgpoints.append(corners2)
        '''
        imgpoints[i] = cv2.convertPointsToHomogeneous(imgpoints[i])
        objpoints[i][ :, :, 2] = 1
        right = findVanishingPoint(
            [imgpoints[i][0][0], imgpoints[i][8][0], imgpoints[i][27][0], imgpoints[i][35][0], imgpoints[i][45][0],
             imgpoints[i][53][0]])
        up = findVanishingPoint(
            [imgpoints[i][0][0], imgpoints[i][45][0], imgpoints[i][8][0], imgpoints[i][53][0], imgpoints[i][4][0],
             imgpoints[i][49][0]])

        cv2.line(images[i], right,(imgpoints[i][0][0][0],imgpoints[i][0][0][1]),color=(0,255,0),thickness=5)
        cv2.line(images[i], right,(imgpoints[i][45][0][0],imgpoints[i][45][0][1]),color=(0,255,0),thickness=5)
        cv2.line(images[i], right,(imgpoints[i][27][0][0],imgpoints[i][27][0][1]),color=(0,255,0),thickness=5)
        cv2.line(images[i], up,(imgpoints[i][0][0][0],imgpoints[i][0][0][1]),color=(255,0,0),thickness=5)
        cv2.line(images[i], up,(imgpoints[i][8][0][0],imgpoints[i][8][0][1]),color=(255,0,0),thickness=5)
        cv2.line(images[i], up,(imgpoints[i][4][0][0],imgpoints[i][4][0][1]),color=(255,0,0),thickness=5)
        cv2.imwrite('ssasd.jpg',images[i])
        q=0
        '''
    return objpoints,imgpoints

def cameraCalibration(i1,i2):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[i1-1:i2], imgpoints[i1-1:i2], (1000,1500), None, None)
    return mtx

def interSectionTwoLine(x1,x2):
    a = x1[1]*x2[2] - x1[2]*x2[1]
    b = x1[2]*x2[0] - x1[0]*x2[2]
    c = x1[0]*x2[1] - x1[1]*x2[0]
    return [a/c, b/c, 1]


def findVanishingPoint(points):
    l1 = interSectionTwoLine(points[0], points[1])
    l2 = interSectionTwoLine(points[2], points[3])
    l3 = interSectionTwoLine(points[4], points[5])
    [x1, y1, c] = interSectionTwoLine(l1, l2)
    [x2, y2, c] = interSectionTwoLine(l2, l3)
    [x3, y3, c] = interSectionTwoLine(l1, l3)
    return (int((x1 + x2 + x3) // 3),int((y1 + y2 + y3) // 3),1)


objpoints, imgpoints = initialization()
print('1 to 10: ')
print(cameraCalibration(1,10))
print('6 to 15: ')
print(cameraCalibration(6,15))
print('11 to 20: ')
print(cameraCalibration(11,20))
print('1 to 20: ')
print(cameraCalibration(1,20))

def findFocusLengths(objpoints, imgpoints):
    px = 750
    py = 500
    f = []
    v = []
    for i in range(len(objpoints)):
        imgpoints[i] = cv2.convertPointsToHomogeneous(imgpoints[i])
        objpoints[i][ :, :, 2] = 1
        h = cv2.findHomography(objpoints[i],imgpoints[i])
        h = h[0]
        tmp = px*h[0,0]/h[2,0]+px*h[0,1]/h[2,1]+py*h[1,0]/h[2,0]+py*h[1,1]/h[2,1]-(h[0,0]*h[0,1]+h[1,0]*h[1,1])/(h[2,0]*h[2,1])-px**2-py**2
        f.append(np.sqrt(tmp))

        right = findVanishingPoint(
            [imgpoints[i][0][0], imgpoints[i][8][0], imgpoints[i][27][0], imgpoints[i][35][0], imgpoints[i][45][0],
             imgpoints[i][53][0]])
        up = findVanishingPoint(
            [imgpoints[i][0][0], imgpoints[i][45][0], imgpoints[i][8][0], imgpoints[i][53][0], imgpoints[i][4][0],
             imgpoints[i][49][0]])


        tmp = px*up[0]/up[2]+px*right[0]/right[2]+py*up[1]/up[2]+py*right[1]/right[2]-(up[0]*right[0]+up[1]*right[1])/(up[2]*right[2])-px**2-py**2
        v.append(np.sqrt(tmp))
        print('pic ',i+1,'  *method 1:', f[i],'  *method 2:',v[i])

findFocusLengths(objpoints, imgpoints)