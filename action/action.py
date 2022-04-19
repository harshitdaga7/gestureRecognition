import json
import pyautogui

FILE_NAME = "config/gestureActionMap.json"

class Action:

    def __init__(self):
        self.gesture_action_map = None

        with open(FILE_NAME,'r') as f:
            self.gesture_action_map = json.load(f)
        
    def performHotkey(self,action_arr):
        
        for i in action_arr:
            pyautogui.keyDown(i)
        
        for i in action_arr[::-1]:
            pyautogui.keyUp(i)

    def doAction(self,gestureName):
        
        action = self.gesture_action_map[gestureName]
        #print(action)

        if action != "":
            action_arr = list(action.split("+"))
            #print(action_arr)
            self.performHotkey(action_arr)

        