import numpy as np
import cv2

same_part = cv2.imread('1.jpg')
full_img = cv2.imread('2.jpg')
diff_part = cv2.imread('3.jpg')

# Convert it to grayscale
si_g = cv2.cvtColor(same_part, cv2.COLOR_BGR2GRAY)
fi_g = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
di_g = cv2.cvtColor(diff_part, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

sp_keypoints, sp_descriptors = orb.detectAndCompute(si_g, None)
fi_keypoints, fi_descriptors = orb.detectAndCompute(fi_g, None)
di_keypoints, di_descriptors = orb.detectAndCompute(di_g, None)

matcher = cv2.BFMatcher()
matches1 = matcher.knnMatch(fi_descriptors, sp_descriptors, k=2)
matches2 = matcher.knnMatch(fi_descriptors, di_descriptors, k=2)


good1 = []
good2 = []

for m, n in matches1:
    if m.distance < 0.75*n.distance:
        good1.append([m])
for m, n in matches2:
    if m.distance < 0.75*n.distance:
        good2.append([m])


final_img1 = cv2.drawMatchesKnn(
    full_img, fi_keypoints, same_part, sp_keypoints, good1, None, flags=2)
final_img2 = cv2.drawMatchesKnn(
    full_img, fi_keypoints, diff_part, di_keypoints, good2, None, flags=2)

final_img1 = cv2.resize(final_img1, (640, 480))
final_img2 = cv2.resize(final_img2, (640, 480))


# cv2.imshow("Full image", cv2.resize(full_img, (640,480)))
# cv2.imshow("Part image", cv2.resize(same_part ,(640,480)))
# cv2.imshow("Diff part image", cv2.resize(diff_part,(640,480)))
# cv2.imshow("kp1",  cv2.resize(cv2.drawKeypoints(same_part,sp_keypoints,None),(640,480)))
# cv2.imshow("kp2",  cv2.resize(cv2.drawKeypoints(full_img, fi_keypoints,None),(640,480)))
# cv2.imshow("kp3",  cv2.resize(cv2.drawKeypoints(diff_part, di_keypoints,None),(640,480)))
cv2.imshow("f1", final_img1)
cv2.imshow("f2", final_img2)

cv2.waitKey(0)
