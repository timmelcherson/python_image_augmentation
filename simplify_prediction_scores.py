import os
import glob
import json


def simplify_confidence(src):

    confidence_values = {}

    with open(src) as f:
        data = json.load(f)

    # Save coordinates in a list of 4 as: left top right bottom
    for entry in data['result_data']:

        confidence_list = []
        filename = (entry['filename'].split('/')[2]).split('.')[0]

        for item in entry['objects']:
            confidence = item['confidence']
            confidence_list.append(confidence)

        confidence_values[filename] = confidence_list

    with open('simplified_prediction_scores.json', 'w') as fp:
        json.dump(confidence_values, fp,  indent=4)

    return confidence_values


def main():

    json_src = 'result_from_test_set_3.json'

    simplify_confidence(json_src)

    
if __name__ == "__main__":
    main()
