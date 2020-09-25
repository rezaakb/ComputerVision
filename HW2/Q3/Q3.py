import cv2

right = [[1330, 3617,1],[1383, 3858,1],[1783, 3640,1],[1806, 3802,1],[2422, 3671,1],[2426, 3833,1]]
left =[[577, 1169,1],[626, 771,1],[2473, 3569,1],[2496, 3130,1],[1302, 3510,1],[1335, 3137,1],[1267, 1193,1],[1306, 790,1],[217, 1158,1],[266, 794,1]]
up = [[1282, 3602,1],[2544, 3663,1],[1255, 2066,1],[2050, 2098,1],[198, 1159,1],[1401, 1199,1]]

def interSectionTwoLine(x1,x2):
    a = x1[1]*x2[2] - x1[2]*x2[1]
    b = x1[2]*x2[0] - x1[0]*x2[2]
    c = x1[0]*x2[1] - x1[1]*x2[0]
    return [a/c, b/c, 1]

border=200

def findVanishingPoint(points):

    l1 = interSectionTwoLine(points[0],points[1])
    l2 = interSectionTwoLine(points[2],points[3])
    l3 = interSectionTwoLine(points[4],points[5])
    [x1 , y1, c] = interSectionTwoLine(l1,l2)
    [x2 , y2, c] = interSectionTwoLine(l2,l3)
    [x3 , y3, c] = interSectionTwoLine(l1,l3)
    return (int((y1+y2+y3)//3)+border,int((x1+x2+x3)//3)+border)

right_vanishing_point = findVanishingPoint(right)
left_vanishing_point = findVanishingPoint(left)
up_vanishing_point = findVanishingPoint(up)

im = cv2.imread('vns.jpg')

double_im = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, None, [0,0,0])
double_im = cv2.line(double_im,right_vanishing_point,left_vanishing_point,(0,255,0),thickness=5)
print('up =', up_vanishing_point)
print('right = ', right_vanishing_point)
print('left = ', left_vanishing_point)
cv2.imwrite('r3.jpg', double_im)
