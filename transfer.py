import json 
import os
import pandas as pd
folder = '/brtx/605-nvme2/ylu174/VQA/ag_qwen_omni_Yi_train'


results = []
video_id_dict = []
for idx, file in enumerate(os.listdir(folder)):
    with open(os.path.join(folder, file), 'r') as f:
        data = json.load(f)
        answer = data['answers']
        video_id_dict.append(data['video_url'].split('=')[-1])
        for idx2, candidate in enumerate(answer):
            results.append({
                "Q_ID": idx+1,
                "Video_ID": data['video_url'].split('=')[-1],
                "Rank": idx2 + 1,
                "Answer": candidate,
            })

new_file = './submissions/omni_full.csv'
submission_df = pd.DataFrame(results)
submission_df.to_csv(new_file, index=False,encoding='utf-8')
print(len(results))

# file1 = './submissions/qwen.csv'

# import csv
# comp_results = []
# with open(file1, 'r') as f1, open(new_file, 'r') as f2:
#     reader1 = csv.reader(f1)
#     reader2 = csv.reader(f2)
#     for idx, row1 in enumerate(reader1):
#         if idx == 0:
#             continue

#         if row1[1] in video_id_dict:

#             comp_results.append({
#                 "Q_ID": row1[0],
#                 "Video_ID": row1[1],
#                 "Rank": row1[2],
#                 "Answer": row1[3],
#             })
# new_file2 = './submissions/qwen_comp.csv'
# comp_results = pd.DataFrame(comp_results)
# comp_results.to_csv(new_file2, index=False,encoding='utf-8')