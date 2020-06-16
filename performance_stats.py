import os
import numpy as np
import glob
import json

def read_txt_file_content(src):

    # counter = 0

    coordinates = {}

    filename = (src.split('/')[1]).split('.')[0]

    # file_list = glob.glob("augmented_test_set_2_txtfiles/*.txt")
    with open(src, 'r') as file_input:
        data = file_input.read().split()
        print(len(file_input.readlines()))
        left = float(data[1]) - float(data[3]) / 2
        top = float(data[2]) - float(data[4]) / 2
        right = float(data[1]) + float(data[3]) / 2
        bottom = float(data[2]) + float(data[4]) / 2
        coordinates[filename] = [left, top, bottom, right]
    # Save coordinates in a list of 4 as: left top right bottom
    # for file_path in file_list:

    #     filename = (file_path.split('\\')[1]).split('.')[0]

    #     # if counter >= 3:
    #     #     break

    #     # center_x, center_y, width, height
    #     with open(file_path, 'r') as file_input:
    #         data = file_input.read().split()
    #         # print(data)
    #         left = float(data[1]) - float(data[3]) / 2
    #         top = float(data[2]) - float(data[4]) / 2
    #         right = float(data[1]) + float(data[3]) / 2
    #         bottom = float(data[2]) + float(data[4]) / 2
    #         coordinates[filename] = [left, top, bottom, right]
        
        # counter += 1
        
    return coordinates
    


def read_json_file(src):

    coordinates = {}
    
    with open(src) as f:
        data = json.load(f)
    

    # Save coordinates in a list of 4 as: left top right bottom
    for entry in data['result_data']:
        filename = (entry['filename'].split('/')[2]).split('.')[0]
        for item in entry['objects']:
            left = item['relative_coordinates']['center_x'] - (item['relative_coordinates']['width'] / 2)
            top = item['relative_coordinates']['center_y'] - (item['relative_coordinates']['height'] / 2)
            right = item['relative_coordinates']['center_x'] + (item['relative_coordinates']['width'] / 2)
            bottom = item['relative_coordinates']['center_y'] + (item['relative_coordinates']['height'] / 2)
            coordinates[filename] = [left, top, right, bottom]

    return coordinates

def calc_all_iou_values(dictA, dictB):

    print("in calc function")
    setA = set(dictA)
    setB = set(dictB)
    print("set A: " + str(setA))

    print("set B: " + str(setB))
    iou_dict = {}

    for key in setA.intersection(setB):
        print("Key: " + key)
        iou = bb_intersection_over_union(dictA[key], dictB[key])
        iou_dict[key] = iou
        setA.remove(key)
        setB.remove(key)
        # print("setA len: " + str(len(setA)) + ", and setB len: " + str(len(setB)))

    return iou_dict

def bb_intersection_over_union(boxA, boxB):

    print("in intersect function")
	# # determine the (x, y)-coordinates of the intersection rectangle
    print("Box A: " + str(boxA))
    print("Box B: " + str(boxB))
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    print("(xA, yA): " + "(" + str(xA) + ", " + str(yA) + ")")
    print("(xB, yB): " + "(" + str(xB) + ", " + str(yB) + ")")
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    print("interArea: " + str(interArea))
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    print("boxAArea: " + str(boxAArea))
    print("boxBArea: " + str(boxBArea))
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    print("iou: " + str(iou))
	# return the intersection over union value
    return iou


def main():
    
    txtSrc1 = 'augmented_test_set_2_txtfiles'
    txtSrc2 = 'augmented_test_set_3_txtfiles'
    jsonSrc1 = 'result_from_test_set_2.json'
    jsonSrc2 = 'result_from_test_set_3.json'
    testJson = 'test.json'
    testTxt = 'augmented_test_set_2_txtfiles/ebb0c82da028690d_gr.txt'

    groundTruthCoordinates1 = read_txt_file_content(testTxt)
    # groundTruthCoordinates2 = read_txt_file_content(txtSrc2)
    # predictedCoordinates1 = read_json_file(jsonSrc1)
    # predictedCoordinates2 = read_json_file(jsonSrc2)
    testCoordinates = read_json_file(testJson)
    predictedAccuracies = {}
    
    # all_iou_values1 = calc_all_iou_values(groundTruthCoordinates1, predictedCoordinates1)
    # all_iou_values2 = calc_all_iou_values(
    #     groundTruthCoordinates2, predictedCoordinates2)
    
    # test_values = calc_all_iou_values(
        # groundTruthCoordinates1, testCoordinates)
    
    # print("From list 1:")
    # for i in range(5):
    #     print(all_iou_values1.values()[i])

    # print("From list 2:")
    # for i in range(5):
    #     print(all_iou_values2.values()[i])

    # print("total avg of list 1: " + str(sum(all_iou_values1.values()) /
    #                           len(all_iou_values1.values())))

    # print("total avg of list 2: " + str(sum(all_iou_values2.values()) /
    #                                     len(all_iou_values2.values())))
    
    # print("total avg of list 2: " + str(sum(test_values.values()) /
    #                                     len(test_values.values())))
    # print(a)
    # print(b)

    # print(groundTruthCoordinates)
    # print(predictedCoordinates)

    


if __name__ == "__main__":
    main()
