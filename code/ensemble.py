import os
import json
from collections import defaultdict,Counter

file_path ='/opt/ml/level2_mrc_nlp-level2-nlp-07/code/ensemble/nbest'           # nbest_prediction.json 파일 경로

json_file_path = []
json_files=[]
em = []

for json_path in os.listdir(file_path):
    if json_path.endswith('.json'):                    
        json_file_path.append(json_path)

for file in json_file_path:
    fp=os.path.join(file_path,file)
    em_count = int(file.split(".")[0])/100         # em을 기준으로 가중치 생성
    # em_count = f"{em_count:.4f}"
    em.append(float(em_count))
    with open(fp,"r",encoding='utf-8') as json_file:
        json_data=json.load(json_file)
        json_files.append(json_data)

key_list=list(json_files[0].keys())
text_list=defaultdict(dict)
result=defaultdict(list)
ensemble_dict = defaultdict(list)

for i in range(len(key_list)):
    text_list[key_list[i]]
    result[key_list[i]]
    ensemble_dict[key_list[i]]


for idx, json_file in enumerate(json_files):
    for kl in key_list:
        for i in range(len(json_file[kl])):
            x = json_file[kl][i]['text']
            # y = json_file[kl][i]['probability']               # em 가중치 x
            # y = json_file[kl][i]['probability'] + em[idx]     # probability 값 + em 가중치 (em / 1000)
            y = json_file[kl][i]['probability'] * em[idx]       # probability 값 * em 가중치 (em / sum(em))

            try:                                                # default dict 형태로 text에 해당하는 probability 값 넣어줌
                if text_list[kl][x]:
                    z = text_list[kl][x]
                    z.append(y)
                    text_list[kl][x] = z

            except:
                text_list[kl][x] = [y]
    

for kl in key_list:
    calculate={}

    for i in text_list[kl].keys():

        if len(text_list[kl][i]) == 1:
            calculate[i] = text_list[kl][i][0]
        else:
            # prob = sum(text_list[kl][i]) / (2 * len(text_list[kl][i]))        # 1차시도
            # prob = sum(text_list[kl][i]) / len(text_list[kl][i])              # 2차시도
            # key_sum = prob + (len(text_list[kl][i]) * 0.03)                   # 4차? 시도
            prob = sum(text_list[kl][i])
            key_sum = prob + (len(text_list[kl][i]) * 0.01)
            calculate[i] = key_sum

        # calculate[i] = sum(text_list[kl][i])                                  # text 가중치 x
    

    max_answer=max(calculate.keys(),key= lambda x : calculate[x])
    # x = sorted(calculate.items(),key= lambda x : calculate[x[0]], reverse = True)[:20]        # 앙상블 된 probability 값 20개 추출

    result[kl]=max_answer

    # for key, via in x:
    #     ensemble_dict[kl].append({"text" : key,"probability" : via})                          # 앙상블 된 값들을 {ID : {text, prob}} 형태로 만들어 주는 과정


print(result)
# print(len(result))

new_nbest_json_path = "mix_ensemble.json"
ensemble_path = "enensemble.json"

with open(new_nbest_json_path, 'w', encoding='utf-8') as file:                  # 앙상블 된 predictions.json
    json.dump(result, file)

# with open(ensemble_path, 'w', encoding='utf-8') as file:                      # 앙상블 된 nbest_predictions.json
#     json.dump(ensemble_dict, file)