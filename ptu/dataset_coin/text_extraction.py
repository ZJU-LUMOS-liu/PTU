import pandas as pd
import json
import random

df = pd.read_excel('taxonomy.xls',sheet_name='target_action_mapping')
data=df.values
#print(df)
#print(data)
print(data.shape)

Result_list=[]
for i in range (180):
    temp_dict={}
    temp_list=[]
    for k in range (778):
        if data[k][0] == i:
            temp_list.append(data[k][3])
            Target_label = data[k][1]
    temp_dict={'ID':i, 'Target_Label':Target_label, 
               'Action_Len':len(temp_list),'Action_Label':temp_list}
    Result_list.append(temp_dict)
print(Result_list)

random.shuffle(Result_list)
train_list= Result_list[0:160]
test_list=Result_list[160:180]

print(len(train_list))
print(test_list)
with open("./coin_train_text.json","w") as f:
    json.dump(train_list,f)

with open("./coin_test_text.json","w") as f:
    json.dump(test_list,f)