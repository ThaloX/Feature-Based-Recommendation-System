from facial_landmarks import FaceLandmarks
from utils import *
import os
import shutil


def detector(frame):
    # Load the Face landmarks class
    fl = FaceLandmarks()
    facefl = FaceLandmarks()

    # Clone Image for Detection
    image = frame.copy()

    # Resize if needed
    h, w = image.shape[0], image.shape[1]
    scale_factor = 1
    if h > 1000:
        scale_factor = 1000 / h * 0.9

    desired_size = (int(w * scale_factor), int(h * scale_factor))
    image = cv2.resize(image, desired_size)

    # 1. Face Landmarks Detection
    landmarks = fl.get_facial_landmarks(image)
    convex_hull = cv2.convexHull(landmarks)

    # 2. Face Extraction
    face_x, face_y, face_w, face_h = cv2.boundingRect(convex_hull)
    offset_x = int(face_w * 0.1)  # 10% of width
    offset_y = int(face_h * 0.1)  # 10% of height

    crop = image[face_y - offset_y:face_y + face_h + offset_y, face_x - offset_x:face_x + face_w + offset_x]
    face = crop.copy()

    # 3. Re-acquire the Landmarks and the contour
    facelm = facefl.get_facial_landmarks(face)
    face_convex_hull = cv2.convexHull(facelm)

    height, width, _ = crop.shape

    # 4. Extract Iris to check colour
    l_iris, r_iris = getIrises(face, facelm)

    left_eye_clr, lprc = eye_color(l_iris)
    right_eye_clr, rprc = eye_color(r_iris)

    # Table of Percentages
    head = ["Eye", "Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other"]

    # display table
    print(head)
    print(["L", lprc])
    print(["R", rprc])

    # print("Left eye color is:", left_eye_clr)
    # print("Right eye color is:", right_eye_clr)

    # 5. Mouth state
    mouth = isOpen(face, 'MOUTH', threshold=13.5, display=False)

    # 6. Create mask
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, face_convex_hull, 255)

    # 7. Extract the face
    lips_list = list(
        [0, 13, 14, 146, 17, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 37,
         375, 39, 40, 402,
         405, 409, 415, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95])
    lips_contour = cv2.convexHull(facelm[lips_list])
    cv2.fillConvexPoly(mask, lips_contour, 0)

    left_eye_list = list(
        [249, 263, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387,
         388, 390,
         398, 466])
    left_eye_contour = cv2.convexHull(facelm[left_eye_list])
    cv2.fillConvexPoly(mask, left_eye_contour, 0)

    right_eye_list = list(
        [105, 107, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 33, 46, 52, 53, 55, 63, 65, 66,
         7, 70])
    right_eye_contour = cv2.convexHull(facelm[right_eye_list])
    cv2.fillConvexPoly(mask, right_eye_contour, 0)



    # 8. Determine the skin tone
    skin_tone, sprc = skin_color(face, mask)
    # print("Skin tone is:", skin_tone)
    # Table of Percentages
    head = ["Pale", "Caucasian", "Tanned", "Brown", "Brown Black", "Other"]

    # display table
    print(head)
    print(sprc)

    # 9. Return values
    return left_eye_clr, right_eye_clr, mouth[0], skin_tone


def satisfied_compare(user_input, detected):
    count = 0

    # Only one count up for the eyes
    if user_input[0] == detected[0] or user_input[1] == detected[1]:
        count = count + 1

    # Rest of the characteristics
    for i in range(len(user_input[2:])):
        if user_input[i] == detected[i]:
            count = count + 1
        else:
            continue

    return count


def main():
    # Get requirements
    global eye_id, mouth_id, skin_id, min_req

    eye_options = ["Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray"]
    mouth_options = ["Open", "Closed"]
    skin_tone_options = ["Pale", "Caucasian", "Tanned", "Brown", "Brown Black"]

    # Eye input
    print("List of eye_options: ", eye_options)

    while True:
        try:
            eye_id = int(input("Choose a colour from the list by typing its index (0-6): "))
            if 0 <= eye_id <= 6:
                break
            else:
                print("The number is out of range")
        except ValueError:
            print("Invalid argument")
            continue

    eye_req = eye_options[eye_id]
    print(eye_req)

    # Mouth input
    print("List of mouth options: ", mouth_options)

    while True:
        try:
            mouth_id = int(input("Choose an option from the list by typing its index (0-1): "))
            if 0 <= mouth_id <= len(mouth_options) - 1:
                break
            else:
                print("The number is out of range")
        except ValueError:
            print("Invalid argument")
            continue

    mouth_req = mouth_options[mouth_id]
    print(mouth_req)

    # Skin input
    print("List of skin options: ", skin_tone_options)

    while True:
        try:
            skin_id = int(input("Choose a colour from the list by typing its index (0-4): "))
            if 0 <= skin_id <= len(skin_tone_options) - 1:
                break
            else:
                print("The number is out of range")
        except ValueError:
            print("Invalid argument")
            continue

    skin_req = skin_tone_options[skin_id]
    print(skin_req)

    # Minimum requirements
    while True:
        try:
            min_req = int(input("Choose the minimum number of requirements to be satisfied (1-3): "))
            if 1 <= min_req <= 3:
                break
            else:
                print("The number is out of range")
        except ValueError:
            print("Invalid argument")
            continue
    print("\n\n")

    required = [eye_req, eye_req, mouth_req, skin_req]

    # Get images
    directory = r'.\resources\images'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(directory, filename)
            frame = cv2.imread(path)

            l_eye, r_eye, mouth, skin = detector(frame)
            detected = [l_eye, r_eye, mouth, skin]

            satisfied_requirements = satisfied_compare(required, detected)

            print(filename, "\t\tMin required: ", min_req, "Satisfied: ", satisfied_requirements, "\n\n")

            if min_req <= satisfied_requirements:
                dest = '../Results/' + filename
                shutil.copyfile(path, dest)
            else:
                continue
        else:
            continue

    # 9. Cleanup
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
