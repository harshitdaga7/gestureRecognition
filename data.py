import cv2
import numpy as np
import os

NO_OF_CAPTURES = 40
DATA_DIRECTORY = "data"

gestures = ['palm','v','close_fist','thumbs_up','thumbs_down']

for g in gestures:
    if not os.path.exists(g):
        os.mkdir(g)

current_index = 0
captures_cnt = 0

cap = cv2.VideoCapture(0)



while True:

	if captures_cnt >= NO_OF_CAPTURES:
		cv2.destroyWindow(gestures[current_index]) 
		current_index += 1
		captures_cnt = 0 

	if current_index >= len(gestures):
		print("process completed")
		break

	ret, frame = cap.read()
	cv2.imshow(gestures[current_index], frame)

	key = cv2.waitKey(1)

	if key == ord('c'):
		captures_cnt += 1
		filename = str(captures_cnt) + ".png"
		file_path = f"{gestures[current_index]}/{filename}"
		cv2.imwrite(file_path,frame)
		print(f"captured and saved to{gestures[current_index]}/{filename}")

	if(key == ord('q')):
		print("quit")
		break
        
cap.release()
cv2.destroyAllWindows()