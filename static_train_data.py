import mediapipe
import copy
import cv2
import numpy as np
import json
import itertools

from utils import cvfpsCalculator

print("imported...")

########### CONSTANTS #################

DEBUG = True
DIRECTORY = "data"
LABEL_FILE_NAME = "static_labels.json"
DATA_FILE_NAME = "static_data.json"
MODE = "W" # W -> WRITE DATA , APPEND DATA
PER_CLASS_LIMIT = 600
KEYPOINTS = {}

# CV2
CAP_DEVICE = 0
CAP_HEIGHT = 480
CAP_WIDTH = 480

#MEDIAPIPE HANDS
STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.7

############

def getclassNames():

	file_path = DIRECTORY + "/" + LABEL_FILE_NAME
	
	data = None

	with open(file_path,'r') as file:
		data = json.load(file)
		
	
	if data != None:
		return data['labels']
	else:
		return data

def init():

	global KEYPOINTS
	classNames = getclassNames()

	if classNames != None:
		for i in classNames:
			KEYPOINTS[i] = []
	
	print("sucessfully iniitalize")
	print(KEYPOINTS)


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def log(results,label,label_cnt):

	global KEYPOINTS

	res = get_xy_points_from_result(results)
	len_res = len(res['x'])
	landmark_list = []
	for i in range(len_res):
		landmark_list.append([res['x'][i],res['y'][i]])
	
	processed_landmark_list = pre_process_landmark(landmark_list)
	KEYPOINTS[label].append(processed_landmark_list)


	if label_cnt >= PER_CLASS_LIMIT:
		file_path = DIRECTORY + "/" + DATA_FILE_NAME
		with open(file_path,'w') as f:
			json.dump(KEYPOINTS,f)
		
		print("sucessfuly saved", label)

	# print(f"{label} , {label_cnt} , ")
	
	

def draw_information(image,information:str,y = 30):


	font = cv2.FONT_HERSHEY_SIMPLEX
	org = (10,y)
	fontScale = 0.5
	color = (255, 0, 0)
	thickness = 1
	
	image = cv2.putText(image,information, org, font, fontScale, color, thickness, cv2.LINE_AA)

	return image




def get_xy_points_from_result(results):

	# returns {x : [] , y : []}

	res = {

		"x" : [],
		"y" : []
	}

	if(results.multi_hand_landmarks):

		for p in results.multi_hand_landmarks[0].landmark:
			res['x'].append(p.x)
			res['y'].append(p.y)

	return res
			


def draw_bounding_rect(image,landmarks, padding = 15):


	image_width, image_height = image.shape[1], image.shape[0]

	landmark_array = np.empty((0, 2), int)


	for p in (landmarks.landmark):
		landmark_x = min(int(p.x * image_width), image_width - 1 )
		landmark_y = min(int(p.y * image_height), image_height - 1)

		landmark_point = [np.array((landmark_x, landmark_y))]
		landmark_array = np.append(landmark_array, landmark_point, axis=0)

	x, y, w, h = cv2.boundingRect(landmark_array)

	start_point = (x - padding, y - padding)
	end_point = (x + w + padding, y + h + padding)
	# BGR
	color = (0, 0, 0)
	thickness = 2
	
	image = cv2.rectangle(image, start_point, end_point, color, thickness)

	return (image,(x, y, w, h))
	

def crop_image(image,bound_rect,padding = 15):
	image_width, image_height = image.shape[1], image.shape[0]
	x,y,w,h = bound_rect
	cropped_image = image[max(y - padding,0) : min(y + h + padding,image_height), max(x - padding,0) : min(x + w + padding,image_width)]

	return cropped_image


def main():

	init()
	# setting cam window properties
	classNames = getclassNames()

	cap = cv2.VideoCapture(CAP_DEVICE)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    #loading mediapipe hands model

	mp_hands = mediapipe.solutions.hands
	hands = mp_hands.Hands(
	    static_image_mode= STATIC_IMAGE_MODE,
	    max_num_hands=1,
	    min_detection_confidence= MIN_DETECTION_CONFIDENCE,
	    min_tracking_confidence= MIN_TRACKING_CONFIDENCE,
	)

	mp_drawing = mediapipe.solutions.drawing_utils
	mp_drawing_styles = mediapipe.solutions.drawing_styles

	# cvfps

	cvfps = cvfpsCalculator()

	# starting cam
	for label in classNames:
		label_cnt = 0
		print('starting ', label)
		while(label_cnt <= PER_CLASS_LIMIT and cap.isOpened()):
			
			while cap.isOpened():
				
				if label_cnt > PER_CLASS_LIMIT:
					break

				fps = cvfps.get()
				# print(fps)

				ret,frame = cap.read()
				if not ret:
					print("empty frames")
					break

				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
				frame = cv2.flip(frame,1)
				debug_frame = frame.copy()
				

				#mediapipe inference on frame and drawing on debug frame

				results = hands.process(frame)
				landmark = get_xy_points_from_result(results)
				bound_rect = None
				# print(landmark)	

				if results.multi_hand_landmarks:
					
					label_cnt += 1
					for hand_landmarks in results.multi_hand_landmarks:

						mp_drawing.draw_landmarks(
						debug_frame,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())
					
					debug_frame,bound_rect = draw_bounding_rect(debug_frame,results.multi_hand_landmarks[0])
					log(results,label,label_cnt)



				information = str(fps) + " fps"
				debug_frame = draw_information(debug_frame,information)
				
				debug_frame = draw_information(debug_frame,f"label : {label} , count : {label_cnt}",45)
				
				debug_frame = cv2.cvtColor(debug_frame,cv2.COLOR_RGB2BGR)
				
				# if bound_rect != None and DEBUG == True:
				# cv2.imshow("cropped", crop_image(debug_frame,bound_rect))
				
				cv2.imshow("GestureRecognition",debug_frame)

				key = cv2.waitKey(1)

				if key == ord('q'):
					print('...quitting')
					cap.release()
					cv2.destroyAllWindows()
					break

				


	cap.release()
	cv2.destroyAllWindows()




main() 

