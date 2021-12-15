from facial_landmarks import FaceLandmarks
from utils import *

# Load the Face landmarks class
fl = FaceLandmarks()
facefl = FaceLandmarks()

# Load and Resize Image
frame = cv2.imread("../Resources/Images/Nick.png")

# Clone Image for Detection
image = frame.copy()

# Resize if needed
h, w = image.shape[0], image.shape[1]
scaleFactor = 1
if h > 1000:
    scaleFactor = 1000 / h * 0.9

desired_size = (int(w * scaleFactor), int(h * scaleFactor))
image = cv2.resize(image, desired_size)


# 1. Face Landmarks Detection
landmarks = fl.get_facial_landmarks(image)
convexHull = cv2.convexHull(landmarks)

# 2. Face Extraction
face_x, face_y, face_w, face_h = cv2.boundingRect(convexHull)
offset_x = int(face_w * 0.1)  # 10% of width
offset_y = int(face_h * 0.1)  # 10% of height

crop = image[face_y - offset_y:face_y + face_h + offset_y, face_x - offset_x:face_x + face_w + offset_x]
face = crop.copy()

# 3. Re-acquire the Landmarks and the contour
facelm = facefl.get_facial_landmarks(face)
faceconvexHull = cv2.convexHull(facelm)

height, width, _ = crop.shape

# 4. Extract Iris to check colour
lIris, rIris = getIrises(face, facelm)

leftEyeClr = eye_color(lIris)
rightEyeClr = eye_color(rIris)

print("Left eye color is:", leftEyeClr)
print("Right eye color is:", rightEyeClr)

# cv2.imshow("Left Iris", lIris)
# cv2.imshow("Right Iris", rIris)

# 5. Mouth state
_, mouth = isOpen(face, 'MOUTH', threshold=13.5, display=False)
print("Mouth is:", mouth[0])

# 6. Create mask
mask = np.zeros((height, width, 1), dtype=np.uint8)
cv2.fillConvexPoly(mask, faceconvexHull, 255)


# 7. Extract the face
lipsList = list([0,13,14,146,17,178,181,185,191,267,269,270,291,308,310,311,312,314,317,318,321,324,37,375,39,40,402,
                 405,409,415,61,78,80,81,82,84,87,88,91,95])
lipsContour = cv2.convexHull(facelm[lipsList])
cv2.fillConvexPoly(mask, lipsContour, 0)

rightEyeList = list([105,107,133,144,145,153,154,155,157,158,159,160,161,163,173,246,33,46,52,53,55,63,65,66,7,70])
rightEyeContour = cv2.convexHull(facelm[rightEyeList])
cv2.fillConvexPoly(mask, rightEyeContour, 0)

leftEyeList = list([249,263,276,282,283,285,293,295,296,300,334,336,362,373,374,380,381,382,384,385,386,387,388,390,
                    398,466])
leftEyeContour = cv2.convexHull(facelm[leftEyeList])
cv2.fillConvexPoly(mask, leftEyeContour, 0)

# 8. Determine the skin tone
skin_tone = skin_color(face, mask)
print("Skin tone is:", skin_tone)


extracted = cv2.bitwise_and(face, face, mask=mask)

cv2.imshow('Face', face)
# cv2.imshow('Mask', mask)
# cv2.imshow('Extracted Skin', extracted)

# 9. Cleanup
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
