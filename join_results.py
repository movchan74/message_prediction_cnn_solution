# coding: utf-8
import json
import numpy as np
import sys

if len(sys.argv) != 5:
    print ('usage: join_result.py input_1.json input_2.json input_3.json output.json')
    exit()

input_1_filename = sys.argv[1]
input_2_filename = sys.argv[2]
input_3_filename = sys.argv[3]
output_filename = sys.argv[4]

with open(input_1_filename) as f:
    submission_1 = json.load(f)

with open(input_2_filename) as f:
    submission_2 = json.load(f)

with open(input_3_filename) as f:
    submission_3 = json.load(f)

if set(submission_1.keys()) != set(submission_2.keys()):
    print ('error')
    exit()

if set(submission_1.keys()) != set(submission_3.keys()):
    print ('error')
    exit()

submission_id_list = submission_1.keys()
final_submission = {}

for submission_id in submission_id_list:
    pred = []
    pred.append(submission_1[submission_id])
    pred.append(submission_2[submission_id])
    pred.append(submission_3[submission_id])
    # final_submission.append({'id': submission_id, 'predictions': pred}) 
    final_submission[submission_id] = pred

with open(output_filename, 'w') as f:
    json.dump(final_submission, f, sort_keys=True, indent=4, separators=(',', ': '))