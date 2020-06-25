import os
import numpy as np
import glob
import json


######### Read ground truth values from each images respective txt file
def read_txt_file_content(src):

    coordinates = {}
    coordinate_list = []

    file_list = glob.glob(src + "*.txt")
    
    for file_path in file_list:

        filename = (file_path.split('\\')[1]).split('.')[0]

        print(filename)

        with open(file_path, 'r') as file_input:
            
            lines = file_input.read().splitlines()

            # coordinates are stoored as center_x, center_y, width, height
            for entry in lines:
                entry_split = entry.split()
                # print("entry at index: " + str(index) + ", is: " + str(entry_split[1]))
                left = float(entry_split[1]) - float(entry_split[3]) / 2
                top = float(entry_split[2]) - float(entry_split[4]) / 2
                right = float(entry_split[1]) + float(entry_split[3]) / 2
                bottom = float(entry_split[2]) + float(entry_split[4]) / 2
                coordinate_list.append([left, top, right, bottom])

            coordinates[filename] = coordinate_list
    
    return coordinates
    

######### Read the results from test batch
def read_json_file(src):

    coordinates = {}
    coordinate_list = []
    
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
            coordinate_list.append([left, top, right, bottom])

        coordinates[filename] = coordinate_list

    return coordinates


######### Calculate the IoU for each image
def calc_all_iou_values(groundTruth, prediction):

    setA = set(groundTruth)
    setB = set(prediction)

    iou_dict = {}

    for key in setA.intersection(setB):
        # print("Key: " + key)
        iou = bb_intersection_over_union(groundTruth[key], prediction[key])
        iou_dict[key] = iou
        print("File: " + key + ", has total IoU: " + str(iou))
        setA.remove(key)
        setB.remove(key)

    return iou_dict

def bb_intersection_over_union(groundTruthBoxes, predictionBoxes):
    # determine the (x, y)-coordinates of the intersection rectangle 
    # remember (0, 0) is top left and (1, 1) is bottom right for  our data
    
    total_groundtruth_area = 0
    total_prediction_area = 0
    total_intersection_area = 0

    for index, prediction in enumerate(predictionBoxes):
        for index, groundTruth in enumerate(groundTruthBoxes):

            xA = max(groundTruth[0], prediction[0])
            yA = max(groundTruth[1], prediction[1])

            xB = min(groundTruth[2], prediction[2])
            yB = min(groundTruth[3], prediction[3])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            # compute the area of both the prediction and ground-truth
            # rectangles
            groundTruthArea = (groundTruth[2] - groundTruth[0]) * (groundTruth[3] - groundTruth[1])
            predictionArea = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
            # print("groundTruth area: " + str(groundTruthArea))
            # print("prediction area: " + str(predictionArea))
            
            total_groundtruth_area += groundTruthArea
            total_prediction_area += predictionArea
            total_intersection_area += interArea
    
    total_iou = total_intersection_area / float(total_groundtruth_area + total_prediction_area - total_intersection_area)
    # print("Total IoU: " + str(total_iou) + "\n")
    return total_iou


def main():
    
    txtSrc1 = 'augmented_test_set_2_txtfiles/'   # Ran 2 test batches to get even more spread of results, 
    txtSrc2 = 'augmented_test_set_3_txtfiles/'   # since test set 2 was used for mAP during training
    jsonSrc1 = 'result_from_test_set_2.json'    # Results from test set 2
    jsonSrc2 = 'result_from_test_set_3.json'    # Results from test set 3
    testJson = 'test.json'
    testTxt = 'test_files/'

    # groundTruthCoordinates1 = read_txt_file_content(txtSrc1)
    # groundTruthCoordinates2 = read_txt_file_content(txtSrc2)
    # predictedCoordinates1 = read_json_file(jsonSrc1)
    # predictedCoordinates2 = read_json_file(jsonSrc2)

    # print(groundTruthCoordinates1)
    # print("From list 1:")
    # for i in range(5):
    #     print(groundTruthCoordinates1[i])

    # print("From list 2:")
    # for i in range(5):
    #     print(predictedCoordinates1[i])

    testGroundtruth = read_txt_file_content(testTxt)
    testCoordinates = read_json_file(testJson)
    
    predictedAccuracies = {}
    
    # all_iou_values1 = calc_all_iou_values(groundTruthCoordinates1, predictedCoordinates1)
    # all_iou_values2 = calc_all_iou_values(
    #     groundTruthCoordinates2, predictedCoordinates2)
    
    # test_values = calc_all_iou_values(
    #     groundTruthCoordinates1, testCoordinates)
    
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
