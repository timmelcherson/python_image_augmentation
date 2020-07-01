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
from itertools import chain
# from skimage.metrics import structural_similarity as ssim

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


# Calculate the IoU for all images
def calc_all_iou_values(groundTruth, prediction):

    setA = set(groundTruth)
    setB = set(prediction)

    iou_dict = {}

    for key in setA.intersection(setB):
        # print("\nKey: " + key)
        iou = bb_intersection_over_union(groundTruth[key], prediction[key])
        iou_dict[key] = iou
        

        setA.remove(key)
        setB.remove(key)

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



def bar_plot_nr_of_objects_vs_iou(groundTruthDict, totalIoUDict, **kwargs):

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



def scatter_plot_ssim_vs_confidence(ssim_src, prediction_src):

    with open(ssim_src) as f:
        data = json.load(f)
    
    with open(prediction_src) as f:
        pred_data = json.load(f)

    # extract dictionary keys
    pred_keys = sorted([*pred_data])
    ssim_keys = sorted([*data])

    g1 = ([], [])
    g2 = ([], [])
    g3 = ([], [])
    g4 = ([], [])
    g5 = ([], [])
    g6 = ([], [])
    g7 = ([], [])
    g8 = ([], [])
    g9 = ([], [])

    counter = 0
    for key in ssim_keys:
        for nested_key in [*data[key]]:

            if nested_key in pred_keys:

                if len(pred_data[nested_key]) is not 0:
                    
                    value = sum(pred_data[nested_key]) / \
                        len(pred_data[nested_key])
                    # print("calculated value: " + str(value) +
                    #       " for key: " + nested_key)
                else:
                    value = 0
                # for value in predictionDict[key]:
                counter += 1

                # gX[0] = ssim values for the X-axis
                # gX[1] = prediction values for the Y-axis

                if "_gn_01" in nested_key:
                    # g1[0].append(value)
                    # g1[1].append(data[key][nested_key])
                    g1[1].append(value)
                    g1[0].append(data[key][nested_key])
                    print("appended 1 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_gn_02" in nested_key:
                    # g2[0].append(value)
                    # g2[1].append(data[key][nested_key])
                    g2[1].append(value)
                    g2[0].append(data[key][nested_key])
                    print("appended 2 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_gn_03" in nested_key:
                    # g3[0].append(value)
                    # g3[1].append(data[key][nested_key])
                    g3[1].append(value)
                    g3[0].append(data[key][nested_key])
                    print("appended 3 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_ga_03" in nested_key:
                    # g4[0].append(value)
                    # g4[1].append(data[key][nested_key])
                    g4[1].append(value)
                    g4[0].append(data[key][nested_key])
                    print("appended 4 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_ga_07" in nested_key:
                    # g5[0].append(value)
                    # g5[1].append(data[key][nested_key])
                    g5[1].append(value)
                    g5[0].append(data[key][nested_key])
                    print("appended 5 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_ga_20" in nested_key:
                    # g6[0].append(value)
                    # g6[1].append(data[key][nested_key])
                    g6[1].append(value)
                    g6[0].append(data[key][nested_key])
                    print("appended 6 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_ga_30" in nested_key:
                    # g7[0].append(value)
                    # g7[1].append(data[key][nested_key])
                    g7[1].append(value)
                    g7[0].append(data[key][nested_key])
                    print("appended 7 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

                elif "_gr" in nested_key:
                    # g8[0].append(value)
                    # g8[1].append(data[key][nested_key])
                    g8[1].append(value)
                    g8[0].append(data[key][nested_key])
                    print("appended 8 for counter: " + str(counter) + " and values: " +
                          "(" + str(value) + ", " + str(data[key][nested_key]) + ")")

    data = (g1, g2, g3, g4, g5, g6, g7, g8)
    colors = ("b", "g", "r", "c", "m", "y", "k", "#7777FF")
    groups = ("gn_01", "gn_02", "gn_03", "ga_03",
              "ga_07", "ga_20", "ga_30", "gr")

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color,
                   edgecolors='none', s=30, label=group)

    plt.ylim(0.8, 1.0)
    plt.title('Matplot scatter plot')
    plt.xlabel('SSIM score')
    plt.ylabel('Prediction precision')
    plt.legend(loc=3)
    plt.show()


def scatter_plot_ssim_vs_iou(ssim_src, iou_src):
    
    with open(ssim_src) as f:
        data = json.load(f)

    # extract dictionary keys
    ssim_keys = sorted([*data])
    iou_keys = sorted([*iou_src])

    g1 = ([], [])
    g2 = ([], [])
    g3 = ([], [])
    g4 = ([], [])
    g5 = ([], [])
    g6 = ([], [])
    g7 = ([], [])
    g8 = ([], [])
    g9 = ([], [])

    for key in ssim_keys:
        for nested_key in [*data[key]]:

            if nested_key in iou_keys:
                iou = iou_src[nested_key]
                ssim = data[key][nested_key]
                # print("Key: " + key + " and nested key: " + nested_key + ", has values:\n" + 
                #       "SSIM: " + str(data[key][nested_key]) + "\nIoU: " + str(iou_src[nested_key]))
                          
                # for value in predictionDict[key]:

    #             # gX[0] = ssim values for the X-axis
    #             # gX[1] = prediction values for the Y-axis

                if "_gn_01" in nested_key:
                    # g1[0].append(value)
                    # g1[1].append(data[key][nested_key])
                    g1[1].append(iou)
                    g1[0].append(ssim)

                elif "_gn_02" in nested_key:
                    # g2[0].append(value)
                    # g2[1].append(data[key][nested_key])
                    g2[1].append(iou)
                    g2[0].append(ssim)

                elif "_gn_03" in nested_key:
                    # g3[0].append(value)
                    # g3[1].append(data[key][nested_key])
                    g3[1].append(iou)
                    g3[0].append(ssim)

                elif "_ga_03" in nested_key:
                    # g4[0].append(value)
                    # g4[1].append(data[key][nested_key])
                    g4[1].append(iou)
                    g4[0].append(ssim)

                elif "_ga_07" in nested_key:
                    # g5[0].append(value)
                    # g5[1].append(data[key][nested_key])
                    g5[1].append(iou)
                    g5[0].append(ssim)

                elif "_ga_20" in nested_key:
                    # g6[0].append(value)
                    # g6[1].append(data[key][nested_key])
                    g6[1].append(iou)
                    g6[0].append(ssim)

                elif "_ga_30" in nested_key:
                    # g7[0].append(value)
                    # g7[1].append(data[key][nested_key])
                    g7[1].append(iou)
                    g7[0].append(ssim)

                elif "_gr" in nested_key:
                    # g8[0].append(value)
                    # g8[1].append(data[key][nested_key])
                    g8[1].append(iou)
                    g8[0].append(ssim)

    g1_avg = (sum(g1[0])/len(g1[0]), sum(g1[1])/len(g1[1]))
    g2_avg = (sum(g2[0])/len(g2[0]), sum(g2[1])/len(g2[1]))
    g3_avg = (sum(g3[0])/len(g3[0]), sum(g3[1])/len(g3[1]))
    g4_avg = (sum(g4[0])/len(g4[0]), sum(g4[1])/len(g4[1]))
    g5_avg = (sum(g5[0])/len(g5[0]), sum(g5[1])/len(g5[1]))
    g6_avg = (sum(g6[0])/len(g6[0]), sum(g6[1])/len(g6[1]))
    g7_avg = (sum(g7[0])/len(g7[0]), sum(g7[1])/len(g7[1]))
    g8_avg = (sum(g8[0])/len(g8[0]), sum(g8[1])/len(g8[1]))

    data = (g1, g2, g3, g4, g5, g6, g7, g8, g1_avg, g2_avg,
            g3_avg, g4_avg, g5_avg, g6_avg, g7_avg, g8_avg)
    colors = ("b", "g", "r", "c", "m", "y", "k", "#7777FF",
              "b", "g", "r", "c", "m", "y", "k", "#7777FF")
    groups = ("gn_01", "gn_02", "gn_03", "ga_03",
              "ga_07", "ga_20", "ga_30", "gr", "gn_01_avg", "gn_02_avg", "gn_03_avg", "ga_03_avg",
              "ga_07_avg", "ga_20_avg", "ga_30_avg", "gr_avg")

    
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    z = np.polyfit([g1_avg[0], g2_avg[0], g3_avg[0], g4_avg[0],
                    g5_avg[0], g6_avg[0], g7_avg[0], g8_avg[0]],
                   [g1_avg[1], g2_avg[1], g3_avg[1], g4_avg[1],
                    g5_avg[1], g6_avg[1], g7_avg[1], g8_avg[1]],
                   1)
    p = np.poly1d(z)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        if "_avg" in group:
            area = 200
        else:
            area = 20
        ax.scatter(x, y, alpha=0.8, c=color,
                   edgecolors='none', s=area, label=group)
        ax.plot(x, p(x), color="#e08283", linestyle="-.")

    # plt.ylim(0.6, 1)
    plt.title('SSIM vs Intersection over Union with Group Averages')
    plt.xlabel('SSIM from corresponding original image')
    plt.ylabel('Intersection over Union')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()


def box_plot_confidence(confidence_src):
    return 0

def box_plot_iou(iou_src):

    
    # extract dictionary keys
    iou_keys = sorted([*iou_src])

    # fake up some data
    spread = np.random.rand(10) * 100
    center = np.ones(5) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100

    g1 = []
    g2 = []
    g3 = []
    g4 = []
    g5 = []
    g6 = []
    g7 = []
    g8 = []
    g9 = []

    for key in iou_keys:
        
        iou = iou_src[key]

        if "_gn_01" in key:
            g1.append(iou)

        elif "_gn_02" in key:
            g2.append(iou)

        elif "_gn_03" in key:
            g3.append(iou)

        elif "_ga_03" in key:
            g4.append(iou)

        elif "_ga_07" in key:
            g5.append(iou)

        elif "_ga_20" in key:
            g6.append(iou)

        elif "_ga_30" in key:
            g7.append(iou)

        elif "_gr" in key:
            g8.append(iou)

        else:
            g9.append(iou)

            
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)

    data = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    # Colors from http://www.flatuicolorpicker.com
    # Color and names (in index-order): 
    # Blue: Hummingbird (0), Chambray (1), Shark (2)
    # Red: Pomegranate (3), Old Brick (4)
    # Green: Light Green (5), Salem (6)
    # Grey: Edward (7)
    # Yellow: Ripe Lemon (8)
    colors = ("#c5eff7", "#3a539b", "#24252a",
              "#f03434", "#96281b", "#7befb2", "#1e824c", "#abb7b7", "#f7ca18")

    ## Custom x-axis labels
    ax.set_xticklabels(
        ['gn_01', 'gn_02', 'gn_03', 'ga_03', 'ga_07', 'ga_20', 'ga_30', 'gr', 'original'])

    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_title('Box Plot')
    
    bp = ax.boxplot(data, patch_artist=True, whis=[10,90])
    # bp = ax.boxplot(data, patch_artist=True)

    # print(len(bp['boxes']))
    # print(bp['boxes'])

    for index, box in enumerate(bp['boxes']):
        print(bp)
        box.set(facecolor=colors[index])
    
    # print(bp['whiskers'])
    # print(len(bp['whiskers']))


    for index, whisker in enumerate(bp['whiskers']):
        if index % 2 == 0:
            whisker.set(color=colors[index - int(index/2)], linewidth=2)
        else:
            whisker.set(color=colors[index - int(index/2 + 1)], linewidth=2)

    for index, cap in enumerate(bp['caps']):
        if index % 2 == 0:
            cap.set(color=colors[index - int(index/2)], linewidth=2)
        else:
            cap.set(color=colors[index - int(index/2 + 1)], linewidth=2)

    plt.ylabel('Intersection over Union')
    plt.show()


    return 0
# NEXT STEP:
# scatter plot with ssim vs IoU
# box plot over confidence and IoU


# MORE EXAMPLES OF NICE PLOTS
# line plot with error bars
# or scatter plot with error bars
# box plot for each category


def main():

    # Ran 2 test batches to get even more spread of results,
    # since test set 2 was used for mAP during training

    txtSrc = 'augmented_test_set_3_txtfiles/'
    jsonSrc = 'result_from_test_set_3.json'    # Results from test set 3
    image_src = 'C:/Users/A560655/Documents/datasets/augmented_bird_polar_bear/'
    original_images = 'C:/Users/A560655/Documents/datasets/bird_polar_bear/'

    ssim_src = 'ssim_result_test_set_3.json'
    prediction_score_src = 'simplified_prediction_scores_test_set_3.json'

    # Read ground truth bounding box coordinates from given txt file
    groundTruthCoordinates = read_txt_file_content(txtSrc)

    # Read coordinates for predicted bounding boxes from acquired json file
    predictedCoordinates = read_json_file_coordinates(jsonSrc)

    # Calculate intersection over union for all images
    iou_values = calc_all_iou_values(groundTruthCoordinates, predictedCoordinates)
    
    # # Create bar plot for images with a single object vs images with multiple objects and their corresponding 
    # # Intersection over Union scores
    # bar_plot_nr_of_objects_vs_iou(
    #     groundTruthCoordinates, iou_values, test_set_nr=1)

    # # Create scatter plot for ssim vs prediction confidence for test set 2 results
    # scatter_plot_ssim_vs_confidence(ssim_src_2, prediction_score_src_2)

    # Create a scatter plot for ssim vs intersection over union for test set 2 results
    # scatter_plot_ssim_vs_iou(ssim_src, iou_values)
    
    # Create a box plot for iou values for each category of augmentation
    box_plot_iou(iou_values)

if __name__ == "__main__":
    main()
