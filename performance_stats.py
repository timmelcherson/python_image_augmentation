import os
import numpy as np
import glob
import json
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
from timeit import default_timer as timer
from itertools import chain
from skimage.metrics import structural_similarity as ssim

# Read ground truth values from each images respective txt file


def read_txt_file_content(src):

    coordinates = {}

    file_list = glob.glob(src + "*.txt")

    for file_path in file_list:

        coordinate_list = []
        filename = (file_path.split('\\')[1]).split('.')[0]

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

            # close file
            file_input.close()

    # print(str(coordinates))
    return coordinates


# Read the results from test batch
def read_json_file_coordinates(src):

    coordinates = {}

    with open(src) as f:
        data = json.load(f)

    # Save coordinates in a list of 4 as: left top right bottom
    for entry in data['result_data']:

        coordinate_list = []
        filename = (entry['filename'].split('/')[2]).split('.')[0]

        for item in entry['objects']:
            left = item['relative_coordinates']['center_x'] - \
                (item['relative_coordinates']['width'] / 2)
            top = item['relative_coordinates']['center_y'] - \
                (item['relative_coordinates']['height'] / 2)
            right = item['relative_coordinates']['center_x'] + \
                (item['relative_coordinates']['width'] / 2)
            bottom = item['relative_coordinates']['center_y'] + \
                (item['relative_coordinates']['height'] / 2)
            coordinate_list.append([left, top, right, bottom])

        coordinates[filename] = coordinate_list

    return coordinates


def read_json_file_confidence(src):

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

    return confidence_values


# Calculate the IoU for all images
def calc_all_iou_values(groundTruth, prediction):

    setA = set(groundTruth)
    setB = set(prediction)

    iou_dict = {}

    start = timer()
    for key in setA.intersection(setB):
        # print("\nKey: " + key)
        iou = bb_intersection_over_union(groundTruth[key], prediction[key])
        iou_dict[key] = iou
        # print("Total IoU: " + str(iou) + "\n")
        # print("groundtruth get len: " + str(len(groundTruth.get(key))))

        setA.remove(key)
        setB.remove(key)

    end = timer()
    print("exec time of bb_intersection_over_union: " + str(end - start) + "\n")
    return iou_dict


def bb_intersection_over_union(groundTruthBoxes, predictionBoxes):
    # determine the (x, y)-coordinates of the intersection rectangle
    # remember (0, 0) is top left and (1, 1) is bottom right for  our data

    total_groundtruth_area = 0
    total_prediction_area = 0
    total_intersection_area = 0
    corresponding_prediction_area = 0
    is_groundtruth_area_calculated = False
    is_prediction_area_calculated = False
    skip_indices = []

    if len(predictionBoxes) == 0:
        return 0

    for prediction in predictionBoxes:

        # print("new prediction")
        # print("prediction object nr: " + str(index))
        previous_distance = 1  # set highest possible distance at first

        # 0 = left, 1 = top, 2 = right, 3 = bottom
        predictionWidth = prediction[2] - prediction[0]
        predictionHeight = prediction[3] - prediction[1]
        predictionCenter_x = prediction[0] + predictionWidth / 2
        predictionCenter_y = prediction[1] + predictionHeight / 2

        predictionArea = predictionWidth * predictionHeight
        total_prediction_area += predictionArea

        for index, groundTruth in enumerate(groundTruthBoxes):

            # Ground truth area is always calculated in first prediction iteration, so skipping
            # does not effect any area calculation
            if index in skip_indices:
                continue

            # print("ground truth object nr: " + str(index))
            groundTruthWidth = groundTruth[2] - groundTruth[0]
            groundTruthHeight = groundTruth[3] - groundTruth[1]
            groundTruthCenter_x = groundTruth[0] + groundTruthWidth / 2
            groundTruthCenter_y = groundTruth[1] + groundTruthHeight / 2

            new_euclidean_distance = np.sqrt(np.square(
                groundTruthCenter_x - predictionCenter_x) + np.square(groundTruthCenter_y - predictionCenter_y))

            # print("new distance: " + str(new_euclidean_distance) + ", previous distance: " + str(previous_distance))

            if new_euclidean_distance < previous_distance:
                # print("Shorter distance detected")
                xA = max(groundTruth[0], prediction[0])
                yA = max(groundTruth[1], prediction[1])
                xB = min(groundTruth[2], prediction[2])
                yB = min(groundTruth[3], prediction[3])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA) * max(0, yB - yA)
                # print("interArea: " + str(interArea))
                total_intersection_area += interArea
                # print("skip index: " + str(index))
                skip_indices.append(index)
                previous_distance = new_euclidean_distance

            interArea = 0

            # compute the area of both the prediction and ground-truth
            # rectangles
            # print("groundTruth area: " + str(groundTruthArea))
            # print("prediction area: " + str(predictionArea))
            # iou = interArea / float(groundTruthArea + predictionArea - interArea)
            # print("iou: " + str(iou))

            if not is_groundtruth_area_calculated:
                groundTruthArea = groundTruthWidth * groundTruthHeight
                total_groundtruth_area += groundTruthArea

        # set flag that ground truth area is done calculated (after 1st prediction comparison)
        is_groundtruth_area_calculated = True

    # print("total ground truth area: " + str(total_groundtruth_area))
    # print("total prediction area: " + str(total_prediction_area))
    # print("total intersection area: " + str(total_intersection_area))

    # Either do this, or try and calculate only IoU for a prediction to its closest ground truth,
    # and then skip all the ground truths that does not have a corresponding prediction
    # this could give a kind of falsely good result, as not even detecting a ground truth object is
    # not good at all.
    total_iou = total_intersection_area / \
        float(total_groundtruth_area +
              total_prediction_area - total_intersection_area)
    return total_iou


def bar_plot_iou_for_number_of_objects(groundTruthDict, totalIoUDict, **kwargs):

    gSet = set(groundTruthDict)
    iouSet = set(totalIoUDict)

    single_obj_iou_values = []
    multiple_obj_iou_values = []

    for key in gSet.intersection(iouSet):

        gt_values = groundTruthDict.get(key)
        iou_value = totalIoUDict.get(key)

        if len(gt_values) > 1:
            multiple_obj_iou_values.append(iou_value)
        else:
            single_obj_iou_values.append(iou_value)

    single_obj_avg_iou = sum(single_obj_iou_values) / \
        len(single_obj_iou_values)
    multiple_obj_avg_iou = sum(
        multiple_obj_iou_values) / len(multiple_obj_iou_values)

    avg_ious = [single_obj_avg_iou, multiple_obj_avg_iou]

    nr_of_objects_label = ['1', '>1']
    y = [len(single_obj_iou_values), len(multiple_obj_iou_values)]

    y_pos = np.arange(len(nr_of_objects_label))

    # Plot the figure
    bars = plt.bar(y_pos, avg_ious, width=0.5)

    # Create bars
    # plt.bar(y_pos, avg_ious, width=0.5)

    # for i, v in enumerate(nr_of_objects):
    #     plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    for index, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.2, yval + .005, "n = " + str(y[index]))

    # create axis labels
    for ar in kwargs:
        if ar == 'test_set_nr':
            plt.title("Test Set nr. {}".format(kwargs.get('test_set_nr')))
    # Create names on the x-axis
    plt.xticks(y_pos, nr_of_objects_label)
    plt.xlabel('Number of ground truth objects in image')
    plt.ylabel('Average IoU score')

    plt.show()


# def PSNR(original, compressed):
#     mse = np.mean((original - compressed) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#         # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
#     return psnr


def calculate_all_ssim(original_images_src, augmented_images_src):

    original_file_list = sorted(glob.glob(original_images_src + "*.jpg"))
    augmented_file_list = sorted(glob.glob(augmented_images_src + "*.jpg"))
    
    # for i in range(20):
    #     print(original_file_list[i])
    # print(type(original_file_list))

    # for i in range(20):
    #     print(sorted(original_file_list)[i])
    # print(type(sorted(original_file_list)))
    ssim_dict = {}

    for file_path_1 in original_file_list:
        
        augmented_score = {}
        original_filename = (file_path_1.split('\\')[1]).split('.')[0]
        original_image = cv2.imread(file_path_1)

        print(original_filename)
        for file_path_2 in augmented_file_list:

            augmented_filename = (file_path_2.split('\\')[1]).split('.')[0]
            
            if original_filename in augmented_filename:
                print("TRUE IS IN IT")
                print(augmented_filename)
                augmented_image = cv2.imread(file_path_2)
                
                print(type(augmented_image))

                ssim_score = ssim(original_image, augmented_image, multichannel=True)
                print(ssim_score)
                augmented_score[augmented_filename] = ssim_score
                

            else:
                continue
            

        ssim_dict[original_filename] = augmented_score
        # print(ssim_dict)
        
    return ssim_dict


def scatter_plot_confidence_over_augmentation_type(ssim_dict, predictionDict):

    # extract dictionary keys
    keys = [*predictionDict]
    ssim_nested_keys = [*ssim_dict]

    # define groups
    g1 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_gn_01' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_gn_01' in key])))
    g2 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_gn_02' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_gn_02' in key])))
    g3 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_gn_03' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_gn_03' in key])))
    g4 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_ga_03' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_ga_03' in key])))
    g5 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_ga_07' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_ga_07' in key])))
    g6 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_ga_20' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_ga_20' in key])))
    g7 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_ga_30' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_ga_30' in key])))
    g8 = (list(chain(ssim_dict[key][nested_key] for key in ssim_nested_keys for nested_key in ssim_dict[key] if '_gr' in nested_key)),
          list(chain(*[predictionDict[key] for key in keys if '_gr' in key])))
    g9 = ([1],
          list(chain(*[predictionDict[key] for key in keys if not '_' in key])))

    # Make sure dimensions are symmetrical by multiplying ssim score list with length of precision list
    g1 = (g1[0] * len(g1[1]), g1[1])
    g2 = (g2[0] * len(g2[1]), g2[1])
    g3 = (g3[0] * len(g3[1]), g3[1])
    g4 = (g4[0] * len(g4[1]), g4[1])
    g5 = (g5[0] * len(g5[1]), g5[1])
    g6 = (g6[0] * len(g6[1]), g6[1])
    g7 = (g7[0] * len(g7[1]), g7[1])
    g8 = (g8[0] * len(g8[1]), g8[1])
    g9 = (g9[0] * len(g9[1]), g9[1])

    # b: blue
    # g: green
    # r: red
    # c: cyan
    # m: magenta
    # y: yellow
    # k: black
    # w: white
    

    data = (g1, g2, g3, g4, g5, g6, g7, g8, g9)
    colors = ("b", "g", "r", "c", "m", "y", "k", "#7777FF", "#99FF33")
    groups = ("gn_01", "gn_02", "gn_03", "ga_03",
              "ga_07", "ga_20", "ga_30", "gr", "original")

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Matplot scatter plot')
    plt.xlabel('SSIM score')
    plt.ylabel('Prediction precision')
    # plt.legend(loc=2)
    plt.show()


def main():

    # Ran 2 test batches to get even more spread of results,
    txtSrc1 = 'augmented_test_set_2_txtfiles/'
    # since test set 2 was used for mAP during training
    txtSrc2 = 'augmented_test_set_3_txtfiles/'
    jsonSrc1 = 'result_from_test_set_2.json'    # Results from test set 2
    jsonSrc2 = 'result_from_test_set_3.json'    # Results from test set 3
    testJson = 'test.json'
    testTxt = 'test_files/'
    # original_images = 'images2/'
    # augmented_images = 'images_dst/'
    original_images = 'C:/Users/A560655/Documents/datasets/augmented_test_images2/'
    augmented_images = 'C:/Users/A560655/Documents/datasets/bird_polar bear/'

    ssim = calculate_all_ssim(original_images, augmented_images)

    print(ssim)
    # groundTruthCoordinates1 = read_txt_file_content(txtSrc1)
    # groundTruthCoordinates2 = read_txt_file_content(txtSrc2)
    # predictedCoordinates1 = read_json_file_coordinates(jsonSrc1)
    # predictedCoordinates2 = read_json_file_coordinates(jsonSrc2)

    # testGroundtruth = read_txt_file_content(testTxt)
    # testCoordinates = read_json_file_coordinates(testJson)

    # test_predicted_confidence_scores = read_json_file_confidence(testJson)
    predicted_confidence_scores_1 = read_json_file_confidence(jsonSrc1)

    # all_iou_values1 = calc_all_iou_values(groundTruthCoordinates1, predictedCoordinates1)
    # all_iou_values2 = calc_all_iou_values(
    #     groundTruthCoordinates2, predictedCoordinates2)

    # bar_plot_iou_for_number_of_objects(
    #     groundTruthCoordinates1, all_iou_values1, test_set_nr=1)
    # bar_plot_iou_for_number_of_objects(
    #     groundTruthCoordinates2, all_iou_values2, test_set_nr=2)

    scatter_plot_confidence_over_augmentation_type(
        ssim, predicted_confidence_scores_1)

    # scatter_plot_confidence_over_augmentation_type(
    #     ssim, test_predicted_confidence_scores)
    # test_values = calc_all_iou_values(testGroundtruth, testCoordinates)

    # print(test_values)
    # plot_iou_for_number_of_objects(testGroundtruth, test_values)

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
