import mediapipe
import copy
import cv2
import json
import numpy as np
import itertools
from classifiers.keypointClassifier import keypointClassifier
from classifiers.pointHistoryClassifier import pointHistoryClassifier
from utils import cvfpsCalculator
from action.action import Action
from threading import Thread
from collections import deque
from collections import Counter


print("imported...")

########### CONSTANTS #################

DEBUG = False
DIRECTORY = "data"
LABEL_FILE_NAME = "static_labels.json"
DYNAMIC_LABEL_FILE_NAME = "dynamic_labels.json"
HISTORY_LEN = 16

# CV2
CAP_DEVICE = 0
CAP_HEIGHT = 480
CAP_WIDTH = 480

#MEDIAPIPE HANDS
STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.7

############
def getclassNames(name = "static"):


	file_path = DIRECTORY

	if name == "static":
		file_path = file_path  + "/" + LABEL_FILE_NAME
	else:
		file_path = file_path +  "/" + DYNAMIC_LABEL_FILE_NAME
	
	data = None

	with open(file_path,'r') as file:
		data = json.load(file)
		
	
	if data != None:
		return data['labels']
	else:
		return data


def get_landmark_list(image,results):

	image_width, image_height = image.shape[1], image.shape[0]
	res = get_xy_points_from_result(results)
	len_res = len(res['x'])
	landmark_list = []
	for i in range(len_res):
		landmark_list.append([res['x'][i]*image_width,res['y'][i]*image_height])
	
	return landmark_list



def pre_process_point_history(image,point_history):

	image_width, image_height = image.shape[1], image.shape[0]

	temp_point_history = copy.deepcopy(point_history)

	# Convert to relative coordinates
	base_x, base_y = 0, 0
	for index, point in enumerate(temp_point_history):
		if index == 0:
			base_x, base_y = point[0], point[1]

		temp_point_history[index][0] = (temp_point_history[index][0] -
										base_x) / image_width
		temp_point_history[index][1] = (temp_point_history[index][1] -
										base_y) / image_height

	# Convert to a one-dimensional list
	temp_point_history = list(
		itertools.chain.from_iterable(temp_point_history))

	return temp_point_history


def pre_process_landmark(results):

	res = get_xy_points_from_result(results)
	len_res = len(res['x'])
	landmark_list = []
	for i in range(len_res):
		landmark_list.append([res['x'][i],res['y'][i]])
	
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

def draw_information(image,information:str, y = 30):


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

def draw_point_history(image, point_history):
	for index, point in enumerate(point_history):
		#print(type(point[0]),type(point[1]))
		if int(point[0]) != 0 and int(point[1]) != 0:
			cv2.circle(image, (int(point[0]), int(point[1])), 1 + int(index / 2),
					  (235, 52, 195), 2)

	return image

def main():

	classNames = getclassNames()
	dynamicClassNames = getclassNames(name = "dynamic")
	action = Action()
	# setting cam window properties

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
	keypoint_classifier = keypointClassifier()
	point_history_classifier = pointHistoryClassifier()

	# cvfps

	cvfps = cvfpsCalculator()


	#
	points_history = deque(maxlen = HISTORY_LEN)
	id_history = deque(maxlen = 3)

	# starting cam

	while cap.isOpened():


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

			for hand_landmarks in results.multi_hand_landmarks:

				mp_drawing.draw_landmarks(
				debug_frame,
				hand_landmarks,
				mp_hands.HAND_CONNECTIONS,
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())
			
			debug_frame,bound_rect = draw_bounding_rect(debug_frame,results.multi_hand_landmarks[0])
			landmark_list = get_landmark_list(debug_frame,results)
			processed_landmark_list = pre_process_landmark(results)

			hand_id = keypoint_classifier(processed_landmark_list)
			curr_label = classNames[hand_id]

			if hand_id == 0:
				points_history.append([landmark_list[8][0],landmark_list[8][1]])
			else:
				t = Thread( target = action.doAction,args = (curr_label,))
				t.start()
				# t.join()
				points_history.append([0,0])
			
			debug_frame = draw_information(debug_frame,classNames[hand_id],y = 45)
			pre_processed_point_history_list = pre_process_point_history(debug_frame,points_history)

			finger_gesture_id = 0
			point_history_len = len(pre_processed_point_history_list)
			if point_history_len == (HISTORY_LEN * 2):
				finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
			
			id_history.append(finger_gesture_id)
			most_common_fg_id = Counter(id_history).most_common()[0][0]
			
			debug_frame = draw_information(debug_frame,classNames[hand_id],y = 45)
			debug_frame = draw_information(debug_frame,dynamicClassNames[most_common_fg_id],y = 60)

			t1 = Thread(target = action.doAction, args = (dynamicClassNames[most_common_fg_id],))
			t1.start()
		
		else:
			points_history.append([0,0])



		information = str(fps) + " fps" + ", press q to quit"
		debug_frame = draw_information(debug_frame,information)
		debug_frame = draw_point_history(debug_frame,points_history)

		debug_frame = cv2.cvtColor(debug_frame,cv2.COLOR_RGB2BGR)
		
		if bound_rect != None and DEBUG == True:
			cv2.imshow("cropped", crop_image(debug_frame,bound_rect))
		
		cv2.imshow("GestureRecognition",debug_frame)
		cv2.setWindowProperty("GestureRecognition", cv2.WND_PROP_TOPMOST, 1)
		key = cv2.waitKey(1)

		if key == ord('q'):
			print('...quitting')
			break

		


	cap.release()
	cv2.destroyAllWindows()




main() 

