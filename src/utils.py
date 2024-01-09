import cv2
import itertools
import numpy as np
import mediapipe as mp

# Initialize the mediapipe face mesh class.
mp_face_mesh = mp.solutions.face_mesh

# Setup the face landmarks function for images.
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)

# Setup the face landmarks function for videos.
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.3)

# Initialize the mediapipe drawing styles class.
mp_drawing_styles = mp.solutions.drawing_styles


# Get both irises
def getIrises(image, landmarks):
    # Get left eye iris from landmarks
    lEyeTop = landmarks[470][1]
    lEyeLeft = landmarks[471][0]
    lEyeBot = landmarks[472][1]
    lEyeRight = landmarks[469][0]
    lEye = image[lEyeTop:lEyeBot, lEyeLeft:lEyeRight].copy()

    # Get right eye iris from landmarks
    rEyeTop = landmarks[475][1]
    rEyeLeft = landmarks[476][0]
    rEyeBot = landmarks[477][1]
    rEyeRight = landmarks[474][0]
    rEye = image[rEyeTop:rEyeBot, rEyeLeft:rEyeRight].copy()

    return lEye, rEye


def getSize(image, face_landmarks, INDEXES):
    """
    This function calculate the height and width of a face part utilizing its landmarks.
    Args:
        image:          The image of person(s) whose face part size is to be calculated.
        face_landmarks: The detected face landmarks of the person whose face part size is to
                        be calculated.
        INDEXES:        The indexes of the face part landmarks, whose size is to be calculated.
    Returns:
        width:     The calculated width of the face part of the face whose landmarks were passed.
        height:    The calculated height of the face part of the face whose landmarks were passed.
        landmarks: An array of landmarks of the face part whose size is calculated.
    """

    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape

    # Convert the indexes of the landmarks of the face part into a list.
    INDEXES_LIST = list(itertools.chain(*INDEXES))

    # Initialize a list to store the landmarks of the face part.
    landmarks = []

    # Iterate over the indexes of the landmarks of the face part.
    for INDEX in INDEXES_LIST:
        # Append the landmark into the list.
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                          int(face_landmarks.landmark[INDEX].y * image_height)])

    # Calculate the width and height of the face part.
    _, _, width, height = cv2.boundingRect(np.array(landmarks))

    # Convert the list of landmarks of the face part into a numpy array.
    landmarks = np.array(landmarks)

    # Return the calculated width height and the landmarks of the face part.
    return width, height, landmarks


def isOpen(image, face_part, threshold=5, display=True):
    """
    This function checks whether the an eye or mouth of the person(s) is open,
    utilizing its facial landmarks.
    Args:
        image:             The image of person(s) whose an eye or mouth is to be checked.
        face_part:         The name of the face part that is required to check.
        threshold:         The threshold value used to check the isOpen condition.
        display:           A boolean value that is if set to true the function displays
                           the output image and returns nothing.
    Returns:
        output_image: The image of the person with the face part is opened  or not status written.
        status:       A dictionary containing isOpen statuses of the face part of all the
                      detected faces.
    """
    # The output of the facial landmarks detection on the image
    face_mesh_results = face_mesh_images.process(image)

    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape

    # Create a dictionary to store the isOpen status of the face part of all the detected faces.
    status = {}

    # Get the indexes of the mouth.
    INDEXES = mp_face_mesh.FACEMESH_LIPS


    # Iterate over the found faces.
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

        # Get the height of the face part.
        _, height, _ = getSize(image, face_landmarks, INDEXES)

        # Get the height of the whole face.
        _, face_height, _ = getSize(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)

        # Check if the face part is open.
        if (height / face_height) * 100 > threshold:

            # Set status of the face part to open.
            status[face_no] = 'Open'

        # Otherwise.
        else:
            # Set status of the face part to close.
            status[face_no] = 'Closed'

    return status

# Color
def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
            hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False

## Eyes

# Define HSV color ranges for eyes colors
eye_class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    eye_class_name[0]: ((166, 21, 50), (240, 100, 85)),
    eye_class_name[1]: ((166, 2, 25), (300, 20, 75)),
    eye_class_name[2]: ((2, 20, 20), (40, 100, 60)),
    eye_class_name[3]: ((20, 3, 30), (65, 60, 60)),
    eye_class_name[4]: ((0, 10, 5), (40, 40, 25)),
    eye_class_name[5]: ((60, 21, 50), (165, 100, 85)),
    eye_class_name[6]: ((60, 2, 25), (165, 20, 65))
}

# Assign pixel color in HSV space to a category
def find_eye_class(hsv):
    color_id = len(eye_class_name) - 1
    for i in range(len(eye_class_name) - 1):
        if check_color(hsv, EyeColor[eye_class_name[i]]) == True:
            color_id = i
    return color_id


def eye_color(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0], image.shape[1]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))

    eye_radius = h / 2 * 0.87  # approximate

    pupil_coords = (int(w / 2), int(h / 2))

    cv2.circle(imgMask, pupil_coords, int(eye_radius), (255, 255, 255), -1)

    # cv2.circle(image, pupil_coords, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(eye_class_name), float)

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_eye_class(imgHSV[y, x])] += 1

    main_color_index = np.argmax(eye_class[:len(eye_class) - 1])

    total_vote = eye_class.sum()
    #print("\n\nDominant Eye Color: ", eye_class_name[main_color_index])
    #print("\n **Eyes Color Percentage **")
    percent = []
    for i in range(len(eye_class_name)):
        #print(eye_class_name[i], ": ", round(eye_class[i] / total_vote * 100, 2), "%")
        percent.append(round(eye_class[i] / total_vote * 100, 2))

    return eye_class_name[main_color_index], percent

## Skin
# Define HSV color ranges for skin tones
skin_class_name = ("Pale", "Caucasian", "Tanned", "Brown", "Brown Black", "Other")
SkinTone = {
    skin_class_name[0]: ((18, 30, 94), (21, 33, 100)),
    skin_class_name[1]: ((18, 33, 71), (18, 34, 88)),
    skin_class_name[2]: ((17, 33, 53), (17, 34, 65)),
    skin_class_name[3]: ((18, 33, 35), (18, 34, 47)),
    skin_class_name[4]: ((16, 33, 17), (17, 34, 30))
}


# Assign pixel color in HSV space to a category
def find_skin_class(hsv):
    color_id = len(skin_class_name) - 1
    for i in range(len(skin_class_name) - 1):
        if check_color(hsv, SkinTone[skin_class_name[i]]) == True:
            color_id = i
    return color_id


def skin_color(image, skin_mask):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0], image.shape[1]

    skin_class = np.zeros(len(skin_class_name), float)

    for y in range(0, h):
        for x in range(0, w):
            if skin_mask[y, x] != 0:
                skin_class[find_skin_class(imgHSV[y, x])] += 1

    main_color_index = np.argmax(skin_class[:len(skin_class) - 1])

    total_vote = skin_class.sum()

    percent = []
    for i in range(len(skin_class_name)):
        percent.append(round(skin_class[i] / total_vote * 100, 2))

    return skin_class_name[main_color_index], percent
