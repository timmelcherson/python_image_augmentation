import os
import numpy as np
import glob
import json

def read_txt_file_content(src):

    counter = 0

    coordinates = []

    file_list = glob.glob("augmented_test_set_2_txtfiles/*.txt")

    # Save coordinates in a list of 4 as: left top right bottom
    for file_path in file_list:

        filename = (file_path.split('\\')[1]).split('.')[0]

        if counter >= 3:
            break

        with open(file_path, 'r') as file_input:
            data = file_input.read().split()
            # print(data)
            coordinates.append((filename, [data[1], data[2], data[3], data[4]]))
        
        counter += 1

    return coordinates
    


def read_json_file(src):

    
    coordinates = []
    
    with open(src) as f:
        data = json.load(f)
    
    # Save coordinates in a list of 4 as: left top right bottom
    for entry in data['data']:
        filename = (entry['filename'].split('/')[2]).split('.')[0]
        for item in entry['objects']:
            left = item['relative_coordinates']['center_x'] - (item['relative_coordinates']['width'] / 2)
            top = item['relative_coordinates']['center_y'] + (item['relative_coordinates']['height'] / 2)
            right = item['relative_coordinates']['center_x'] + (item['relative_coordinates']['width'] / 2)
            bottom = item['relative_coordinates']['center_y'] - (item['relative_coordinates']['height'] / 2)
            coordinates.append((filename, [left, top, right, bottom]))
    
    return coordinates


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle 
    # remember (0, 0) is top left and (1, 1) is bottom right for  our data
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
    
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def main():
    
    txtSrc1 = 'augmented_test_set_2_txtfiles'
    txtSrc2 = 'augmented_test_set_3_txtfiles'
    jsonSrc1 = 'result_from_test_set_2.json'
    jsonSrc2 = 'result_from_test_set_3.json'
    testJson = 'test.json'

    groundTruthCoordinates = read_txt_file_content(txtSrc1)
    predictedCoordinates = read_json_file(testJson)
    predictedAccuracies = []

    print(groundTruthCoordinates)
    print(predictedCoordinates)

    


if __name__ == "__main__":
    main()
