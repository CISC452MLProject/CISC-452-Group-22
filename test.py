import cv2

img = cv2.imread("img\\1_2018-09-13_1-24.jpg",1)
# cv2.imshow('image', img)
#24#744.8,560.3;729.2,493.9;678.4,505.6;629.6,515.4;678.4,492.0;631.5,566.2;658.9,509.6;666.7,415.8;670.6,406.0;687.2,366.0;760.4,435.3;737.0,449.0;701.8,421.7;633.5,404.1;619.8,452.9;668.6,454.9
img = cv2.circle(img, (744, 560), 5, (0, 0, 255), -1)
img = cv2.circle(img, (729, 493), 5, (0, 0, 255), -1)
img = cv2.circle(img, (678, 505), 5, (0, 0, 255), -1)
img = cv2.circle(img, (629, 515), 5, (0, 0, 255), -1)
img = cv2.circle(img, (678, 492), 5, (0, 0, 255), -1)
img = cv2.circle(img, (631, 566), 5, (0, 0, 255), -1)
img = cv2.circle(img, (658, 509), 5, (0, 0, 255), -1)
img = cv2.circle(img, (666, 415), 5, (0, 0, 255), -1)
img = cv2.circle(img, (670, 406), 5, (0, 0, 255), -1)
img = cv2.circle(img, (687, 366), 5, (0, 0, 255), -1)
img = cv2.circle(img, (760, 435), 5, (0, 0, 255), -1)
img = cv2.circle(img, (737, 449), 5, (0, 0, 255), -1)
img = cv2.circle(img, (701, 421), 5, (0, 0, 255), -1)
img = cv2.circle(img, (633, 404), 5, (0, 0, 255), -1)
img = cv2.circle(img, (619, 452), 5, (0, 0, 255), -1)
img = cv2.circle(img, (668, 454), 5, (0, 0, 255), -1)

cv2.imwrite("test.jpg", img)

# while(1):
#     continue