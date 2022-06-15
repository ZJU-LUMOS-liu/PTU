import json
import os
import random

from torch.utils.data import Dataset

def Number_to_String(number):
    if number == '0':
        return 'zero'
    if number == '1':
        return 'one'
    if number == '2':
        return 'two'
    if number == '3':
        return 'three'
    if number == '4':
        return 'four'
    if number == '5':
        return 'five'
    if number == '6':
        return 'six'
    if number == '7':
        return 'seven'
    if number == '8':
        return 'eight'
    if number == '9':
        return 'nine'
    if number == '10':
        return 'ten'

def Add_Prompt(number,start_string,goal_string):
    return 'Taking ' + Number_to_String(str(number)) + ' steps, ' + 'from ' + start_string + ' to ' + goal_string + ', we need to '

class coin_text_dataset(Dataset):
    def __init__(self, filename):       
        self.filename=filename 
        with open(self.filename,'r',encoding='utf-8') as file:
             self.action_list=json.load(file)
        
    def __len__(self):
        return len(self.action_list)
    
    def __getitem__(self, index):    
        action=self.action_list[index]
        total_step=action['Action_Len']
        predict_step_number=total_step-2
        Start_Message=action['Action_Label'][0]
        Goal_Message=action['Action_Label'][total_step-1]
        #Predicted='[sos]'
        Predict=""
        for i in range(1,total_step-1):
            Predict=Predict+action['Action_Label'][i]+'.'
        Prompt=Add_Prompt(predict_step_number,Start_Message,Goal_Message)
        return Prompt,Predict
    