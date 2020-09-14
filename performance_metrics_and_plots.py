import os
import numpy as np
import glob
import json
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
from itertools import chain

import performance_metrics_utils as utils
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

                left = float(entry_split[1]) - float(entry_split[3]) / 2
                top = float(entry_split[2]) - float(entry_split[4]) / 2
                right = float(entry_split[1]) + float(entry_split[3]) / 2
                bottom = float(entry_split[2]) + float(entry_split[4]) / 2
                coordinate_list.append([left, top, right, bottom])

            coordinates[filename] = coordinate_list

            # close file
            file_input.close()

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
        iou = bb_intersection_over_union(groundTruth[key], prediction[key])
        iou_dict[key] = iou

        setA.remove(key)
        setB.remove(key)

    return iou_dict


# COMMENT THIS
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

            groundTruthWidth = groundTruth[2] - groundTruth[0]
            groundTruthHeight = groundTruth[3] - groundTruth[1]
            groundTruthCenter_x = groundTruth[0] + groundTruthWidth / 2
            groundTruthCenter_y = groundTruth[1] + groundTruthHeight / 2

            new_euclidean_distance = np.sqrt(np.square(
                groundTruthCenter_x - predictionCenter_x) + np.square(groundTruthCenter_y - predictionCenter_y))

            if new_euclidean_distance < previous_distance:

                xA = max(groundTruth[0], prediction[0])
                yA = max(groundTruth[1], prediction[1])
                xB = min(groundTruth[2], prediction[2])
                yB = min(groundTruth[3], prediction[3])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA) * max(0, yB - yA)
                total_intersection_area += interArea
                skip_indices.append(index)
                previous_distance = new_euclidean_distance

            interArea = 0

            if not is_groundtruth_area_calculated:
                groundTruthArea = groundTruthWidth * groundTruthHeight
                total_groundtruth_area += groundTruthArea

        # set flag that ground truth area is done calculated (after 1st prediction comparison)
        is_groundtruth_area_calculated = True

    # Either do this, or try and calculate only IoU for a prediction to its closest ground truth,
    # and then skip all the ground truths that does not have a corresponding prediction
    # this could give a kind of falsely good result, as not even detecting a ground truth object is
    # not good at all.
    total_iou = total_intersection_area / \
        float(total_groundtruth_area +
              total_prediction_area - total_intersection_area)
    return total_iou


# COMMENT THIS
def bar_plot_nr_of_objects_vs_iou(groundTruthDict, totalIoUDict):

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

    # Create names on the x-axis
    plt.title("Test Set nr. {}".format(kwargs.get('test_set_nr')))
    plt.xticks(y_pos, nr_of_objects_label)
    plt.xlabel('Number of ground truth objects in image')
    plt.ylabel('Average IoU score')

    plt.show()


# COMMENT THIS
def scatter_plot_ssim_vs_confidence(ssim_src, prediction_src):

    data = utils.extract_ssim_vs_confidence_groups(
        ssim_src, prediction_src)

    colors = ("#c5eff7", "#3a539b", "#24252a",
              "#f03434", "#96281b", "#7befb2", "#1e824c", "#abb7b7", "#f7ca18")
    groups = ("GN 0.1", "GN 0.2", "GN 0.3", "GA 0.3",
              "GA 0.7", "GA 2.0", "Ga 3.0", "GR")

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
    plt.legend(bbox_to_anchor=(1.12, 1), markerscale=0.4)
    plt.show()


# COMMENT THIS
def scatter_plot_ssim_vs_iou(ssim_src, iou_src):

    data = utils.extract_ssim_vs_iou_groups(ssim_src, iou_src)

    colors = ("#2c82c9", "#3a539b", "#24252a", "#f03434", "#96281b", "#7befb2", "#1e824c", "#abb7b7",
              "#2c82c9", "#3a539b", "#24252a", "#f03434", "#96281b", "#7befb2", "#1e824c", "#abb7b7")

    groups = ("GN 0.1", "GN 0.2", "GN 0.3", "GA 0.3",
              "GA 0.7", "GA 2.0", "GA 3.0", "GR", "GN 0.1 avg", "GN 0.2 avg", "GN 0.3 avg", "GA 0.3 avg",
              "GA 0.7 avg", "GA 2.0 avg", "GA 3.0 avg", "GR avg")

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)

    z = np.polyfit([data[8][0], data[9][0], data[10][0], data[11][0],
                    data[12][0], data[13][0], data[14][0], data[15][0]],
                   [data[8][1], data[9][1], data[10][1], data[11][1],
                    data[12][1], data[13][1], data[14][1], data[15][1]],
                   1)
    p = np.poly1d(z)

    for data, color, group in zip(data, colors, groups):
        x, y = data  # variable data is a tuple of ([x vals], [y vals])
        label = ""
        if "avg" in group:
            area = 200
            label = None
        else:
            area = 20
            label = group
        ax.scatter(x, y, alpha=0.8, c=color,
                   edgecolors='none', s=area, label=label)
        ax.plot(x, p(x), color="#e08283", linestyle="-.")

    # plt.ylim(0.6, 1)
    plt.title('SSIM vs Intersection over Union with Group Averages')
    plt.xlabel('SSIM from corresponding original image')
    plt.ylabel('Intersection over Union')
    plt.xlim(0, 1.01)
    plt.ylim(0.5, 1.05)
    plt.legend(bbox_to_anchor=(1, 1), markerscale=2)
    plt.show()


# COMMENT THIS
def box_plot_confidence(confidence_src):

    data = utils.extract_confidence_groups(confidence_src)

    # Colors from http://www.flatuicolorpicker.com
    # Color and names (in index-order):
    # Blue: Mariner (0), Chambray (1), Shark (2)
    # Red: Pomegranate (3), Old Brick (4)
    # Green: Light Green (5), Salem (6)
    # Grey: Edward (7)
    # Yellow: Ripe Lemon (8)
    colors = ("#2c82c9", "#3a539b", "#24252a",
              "#f03434", "#96281b", "#7befb2", "#1e824c", "#abb7b7", "#f7ca18")

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)

    # Custom x-axis labels
    ax.set_xticklabels(
        ['gn_01', 'gn_02', 'gn_03', 'ga_03', 'ga_07', 'ga_20', 'ga_30', 'gr', 'original'])

    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_title('Box Plot')

    bp = ax.boxplot(data, patch_artist=True, whis=[10, 90])
    # bp = ax.boxplot(data, patch_artist=True)

    for index, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[index])

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

    for median in bp['medians']:
        median.set(linewidth=2)

    plt.ylim(0, 1.05)
    plt.ylabel('Prediction Confidence')
    plt.show()


# COMMENT THIS
def box_plot_iou(iou_src):

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)

    data = utils.extract_iou_groups(iou_src)

    # Colors from http://www.flatuicolorpicker.com
    # Color and names (in index-order):
    # Blue: Hummingbird (0), Chambray (1), Shark (2)
    # Red: Pomegranate (3), Old Brick (4)
    # Green: Light Green (5), Salem (6)
    # Grey: Edward (7)
    # Yellow: Ripe Lemon (8)
    colors = ("#c5eff7", "#3a539b", "#24252a",
              "#f03434", "#96281b", "#7befb2", "#1e824c", "#abb7b7", "#f7ca18")

    # Custom x-axis labels
    ax.set_xticklabels(
        ['gn_01', 'gn_02', 'gn_03', 'ga_03', 'ga_07', 'ga_20', 'ga_30', 'gr', 'original'])

    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_title('Box Plot')

    bp = ax.boxplot(data, patch_artist=True, whis=[10, 90])
    # bp = ax.boxplot(data, patch_artist=True)

    for index, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[index])

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

    for median in bp['medians']:
        median.set(linewidth=2)

    plt.ylabel('Intersection over Union')
    plt.show()


# COMMENT THIS
def scatter_iou_vs_ssim_with_errors(ssim_src, iou_src):
    data = utils.extract_ssim_vs_iou_groups(ssim_src, iou_src)

    values = data[0:9]
    avg_values = [value for value in data[9:]]
    avg_list = [[], []]
    for entry in avg_values:
        avg_list[0].append(entry[0])
        avg_list[1].append(entry[1])

    colors = ("#c5eff7", "#3a539b", "#24252a", "#f03434",
              "#96281b", "#7befb2", "#1e824c", "#abb7b7", "#f7ca18")
    groups = ("gn_01_avg", "gn_02_avg", "gn_03_avg", "ga_03_avg",
              "ga_07_avg", "ga_20_avg", "ga_30_avg", "gr_avg", "original_avg")

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)

    # x = (1, 2, 3, 4)
    # y = (1, 2, 3, 4)
    # print(np.array([[0.3]*len(x), [0.17]*len(x)]))
    # plt.errorbar(x, y, yerr=np.array([[0.3]*len(x), [0.17]*len(x)]), fmt='r^')

    # yerr= [[0.3  0.3  0.3  0.3 ]
    #       [0.17 0.17 0.17 0.17]]   is the shape of yerr parameter, shape=(2,N)

    group_std = []
    group_10_90_percentile_x = [[], []]
    group_10_90_percentile_y = [[], []]

    for group in values:
        # Standard deviation of x ([0])and y ([1])
        group_std.append((np.std(group[0]), np.std(group[1])))

        group_10_90_percentile_x[0].append(np.percentile(group[0], 10))
        group_10_90_percentile_x[1].append(np.percentile(group[0], 90))
        group_10_90_percentile_y[0].append(np.percentile(group[1], 10))
        group_10_90_percentile_y[1].append(np.percentile(group[1], 90))

    group_10_90_percentile_x[0] = np.subtract(
        avg_list[0], group_10_90_percentile_x[0])
    group_10_90_percentile_x[1] = np.subtract(
        group_10_90_percentile_x[1], avg_list[0])
    group_10_90_percentile_y[0] = np.subtract(
        avg_list[1], group_10_90_percentile_y[0])
    group_10_90_percentile_y[1] = np.subtract(
        group_10_90_percentile_y[1], avg_list[1])

    # aa = np.array([3.581, -0.721, 0.137, 0.645, 0.12,
    #                0., -3.236, 0.248, -5.687, 0.816])
    # np.random.rand(aa.size, 3)

    # ax.errorbar(x=avg_list[0], y=avg_list[1], xerr=group_10_90_percentile_x, yerr=group_10_90_percentile_y,
    #             fmt='o', ecolor=colors, capsize=8.0, capthick=2.0, elinewidth=2.0, marker="None")
    # ax.scatter(x=avg_list[0], y=avg_list[1], s=300,
    #            color=colors, marker="o", label=groups)

    # for avg_value, percentile_x_10, percentile_x_90, percentile_y_10, percentile_y_90, color, group in zip(avg_values, group_10_90_percentile_x[0], group_10_90_percentile_x[1], group_10_90_percentile_y[0], group_10_90_percentile_y[1], colors, groups):
    #     print([percentile_x_10, percentile_x_90])
    #     print(avg_value[0])
    #     ax.errorbar(x=avg_value[0], y=avg_value[1], xerr=[[percentile_x_10], [percentile_x_90]], yerr=[[percentile_y_10], [percentile_y_90]],
    #                 fmt='o', ecolor=color, capsize=8.0, capthick=2.0, elinewidth=2.0, marker="None")
    #     ax.scatter(x=avg_value[0], y=avg_value[1], s=300, color=color, marker="o", label=group)

    for avg_value, st_dev, color, group in zip(avg_values, group_std, colors, groups):
        ax.errorbar(x=avg_value[0], y=avg_value[1], xerr=st_dev[0], yerr=st_dev[1],
                    fmt='o', ecolor=color, capsize=8.0, capthick=2.0, elinewidth=2.0, marker="None")
        ax.scatter(x=avg_value[0], y=avg_value[1], s=300,
                   color=color, marker="o", label=group)

    plt.legend(bbox_to_anchor=(1.12, 1), markerscale=0.4)
    plt.title('SSIM vs Intersection with standard deviation')
    plt.xlabel('SSIM from corresponding original image')
    plt.ylabel('Intersection over Union')
    plt.show()


def calculate_percentiles_and_metrics_iou(iou_values):

    gn_01 = [iou_values[key] for key in iou_values.keys() if '_gn_01' in key]
    gn_02 = [iou_values[key] for key in iou_values.keys() if '_gn_02' in key]
    gn_03 = [iou_values[key] for key in iou_values.keys() if '_gn_03' in key]
    ga_03 = [iou_values[key] for key in iou_values.keys() if '_ga_03' in key]
    ga_07 = [iou_values[key] for key in iou_values.keys() if '_ga_07' in key]
    ga_20 = [iou_values[key] for key in iou_values.keys() if '_ga_20' in key]
    ga_30 = [iou_values[key] for key in iou_values.keys() if '_ga_30' in key]
    gr = [iou_values[key] for key in iou_values.keys() if '_gr' in key]

    data = (gn_01, gn_02, gn_03, ga_03, ga_07, ga_20, ga_30, gr)
    quantiles = [0.9, 0.75, 0.5, 0.25, 0.1]
    all_quantiles = []
    for quantile in quantiles:
        temp = []
        for group in data:
            temp.append(str(round(np.quantile(group, quantile), 4)))

        all_quantiles.append(temp)

    all_quantiles = np.array(all_quantiles)

    columns = ('gn_01', 'gn_02', 'gn_03', 'ga_03',
               'ga_07', 'ga_20', 'ga_30', 'gr')
    rows_quantiles = (r'$q_{0.90}$', r'$q_{0.75}$',
                      r'$q_{0.50}$', r'$q_{0.25}$', r'$q_{0.10}$')

    rows_metrics = ("Mean", "Variance", "STD")

    mean = []
    variance = []
    std = []

    for group in data:
        mean.append(str(round(np.mean(group), 4)))
        variance.append(str(round(np.var(group), 4)))
        std.append(str(round(np.std(group), 4)))

    metrics = [mean, variance, std]

    # Plot table for quantiles
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    cellColors = [["w"]*len(columns) for _ in range(len(rows_quantiles))]
    row_max_indices = [np.argmax(row) for row in all_quantiles]
    row_min_indices = [np.argmin(row) for row in all_quantiles]

    for min_index, max_index, i in zip(row_min_indices, row_max_indices, range(len(rows_quantiles))):
        cellColors[i][min_index] = (1, 0, 0, 0.3)
        cellColors[i][max_index] = (0, 1, 0, 0.3)

    row_colours = [(0.3, 0.6, 1, 0.1)]*len(rows_quantiles)
    col_colours = [(0.3, 0.6, 1, 0.1)]*len(columns)

    table = ax.table(cellText=all_quantiles,
                     cellColours=cellColors,
                     cellLoc='center',
                     rowLabels=rows_quantiles,
                     rowColours=row_colours,
                     colLabels=columns,
                     colColours=col_colours,
                     loc="upper center")

    table.set_fontsize(14)
    table.scale(1, 3)
    ax.axis('off')
    plt.show()

    # Plot table for metrics
    fig_metrics = plt.figure(figsize=(10, 3))
    ax_metrics = fig_metrics.add_subplot(1, 1, 1)

    cellColors = [["w"]*len(columns) for _ in range(len(rows_metrics))]
    row_max_indices = [np.argmax(row) for row in metrics]
    row_min_indices = [np.argmin(row) for row in metrics]

    for min_index, max_index, i in zip(row_min_indices, row_max_indices, range(len(rows_metrics))):
        if i == 0:
            cellColors[i][min_index] = (1, 0, 0, 0.3)
            cellColors[i][max_index] = (0, 1, 0, 0.3)
        else:
            cellColors[i][max_index] = (1, 0, 0, 0.3)
            cellColors[i][min_index] = (0, 1, 0, 0.3)

    row_colours = [(0.3, 0.6, 1, 0.1)]*len(rows_quantiles)
    col_colours = [(0.3, 0.6, 1, 0.1)]*len(columns)

    table_metrics = ax_metrics.table(cellText=metrics,
                                     cellColours=cellColors,
                                     cellLoc='center',
                                     rowLabels=rows_metrics,
                                     rowColours=row_colours,
                                     colLabels=columns,
                                     colColours=col_colours,
                                     loc="upper center")

    table_metrics.set_fontsize(14)
    table_metrics.scale(1, 3)
    ax_metrics.axis('off')
    plt.show()


def calculate_percentiles_and_metrics_prediction_confidence(prediction_values):
    
    # returns list of lists in order gn_01, gn_02, gn_03, ga_03, ga_07, ga_20, ga_30, gr, original
    data = utils.extract_confidence_groups(prediction_values)

    # gn_01 = [prediction_values[key] for key in prediction_values.keys() if '_gn_01' in key]
    # gn_02 = [prediction_values[key] for key in prediction_values.keys() if '_gn_02' in key]
    # gn_03 = [prediction_values[key] for key in prediction_values.keys() if '_gn_03' in key]
    # ga_03 = [prediction_values[key] for key in prediction_values.keys() if '_ga_03' in key]
    # ga_07 = [prediction_values[key] for key in prediction_values.keys() if '_ga_07' in key]
    # ga_20 = [prediction_values[key] for key in prediction_values.keys() if '_ga_20' in key]
    # ga_30 = [prediction_values[key] for key in prediction_values.keys() if '_ga_30' in key]
    # gr = [prediction_values[key] for key in prediction_values.keys() if '_gr' in key]

    # data = (gn_01, gn_02, gn_03, ga_03, ga_07, ga_20, ga_30, gr)

    quantiles = [0.9, 0.75, 0.5, 0.25, 0.1]
    all_quantiles = []
    for quantile in quantiles:
        temp = []
        for group in data:
            temp.append(str(round(np.quantile(group, quantile), 4)))

        all_quantiles.append(temp)

    all_quantiles = np.array(all_quantiles)

    columns = ('gn_01', 'gn_02', 'gn_03', 'ga_03',
               'ga_07', 'ga_20', 'ga_30', 'gr', 'original')
    rows_quantiles = (r'$q_{0.90}$', r'$q_{0.75}$',
                      r'$q_{0.50}$', r'$q_{0.25}$', r'$q_{0.10}$')

    rows_metrics = ("Mean", "Variance", "STD")

    mean = []
    variance = []
    std = []

    for group in data:
        mean.append(str(round(np.mean(group), 4)))
        variance.append(str(round(np.var(group), 4)))
        std.append(str(round(np.std(group), 4)))

    metrics = [mean, variance, std]

    # Plot table for quantiles
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    cellColors = [["w"]*len(columns) for _ in range(len(rows_quantiles))]
    row_max_indices = [np.argmax(row) for row in all_quantiles]
    row_min_indices = [np.argmin(row) for row in all_quantiles]

    for min_index, max_index, i in zip(row_min_indices, row_max_indices, range(len(rows_quantiles))):
        cellColors[i][min_index] = (1, 0, 0, 0.3)
        cellColors[i][max_index] = (0, 1, 0, 0.3)

    row_colours = [(0.3, 0.6, 1, 0.1)]*len(rows_quantiles)
    col_colours = [(0.3, 0.6, 1, 0.1)]*len(columns)

    table = ax.table(cellText=all_quantiles,
                     cellColours=cellColors,
                     cellLoc='center',
                     rowLabels=rows_quantiles,
                     rowColours=row_colours,
                     colLabels=columns,
                     colColours=col_colours,
                     loc="upper center")

    table.set_fontsize(14)
    table.scale(1, 3)
    ax.axis('off')
    plt.show()

    # Plot table for metrics
    fig_metrics = plt.figure(figsize=(10, 3))
    ax_metrics = fig_metrics.add_subplot(1, 1, 1)

    cellColors = [["w"]*len(columns) for _ in range(len(rows_metrics))]
    row_max_indices = [np.argmax(row) for row in metrics]
    row_min_indices = [np.argmin(row) for row in metrics]

    for min_index, max_index, i in zip(row_min_indices, row_max_indices, range(len(rows_metrics))):
        if i == 0:
            cellColors[i][min_index] = (1, 0, 0, 0.3)
            cellColors[i][max_index] = (0, 1, 0, 0.3)
        else:
            cellColors[i][max_index] = (1, 0, 0, 0.3)
            cellColors[i][min_index] = (0, 1, 0, 0.3)

    row_colours = [(0.3, 0.6, 1, 0.1)]*len(rows_quantiles)
    col_colours = [(0.3, 0.6, 1, 0.1)]*len(columns)

    table_metrics = ax_metrics.table(cellText=metrics,
                                     cellColours=cellColors,
                                     cellLoc='center',
                                     rowLabels=rows_metrics,
                                     rowColours=row_colours,
                                     colLabels=columns,
                                     colColours=col_colours,
                                     loc="upper center")

    table_metrics.set_fontsize(14)
    table_metrics.scale(1, 3)
    ax_metrics.axis('off')
    plt.show()


# COMMENT THIS
def main():

    # Ran 2 test batches to get even more spread of results,
    # since test set 2 was used for mAP during training

    txtSrc = 'augmented_test_set_3_txtfiles/'
    jsonSrc = 'result_from_test_set_3.json'    # Results from test set 3
    image_src = 'C:/Users/A560655/Documents/datasets/augmented_bird_polar_bear/'
    original_images = 'C:/Users/A560655/Documents/datasets/bird_polar_bear/'

    script_dir = os.path.dirname(__file__)

    original_relative_path = './original_images/'
    grey_relative_path = './grey_images/'
    gamma_relative_path = './gamma_images/'
    grey_noise_relative_path = './grey_noise_images/'

    original_src = os.path.join(script_dir, original_relative_path)
    grey_src = os.path.join(script_dir, grey_relative_path)
    gamma_src = os.path.join(script_dir, gamma_relative_path)
    grey_noise_src = os.path.join(script_dir, grey_noise_relative_path)

    ssim_src = 'ssim_result_test_set_3.json'
    prediction_score_src = 'simplified_prediction_scores_test_set_3.json'

    # Read ground truth bounding box coordinates from given txt file
    groundTruthCoordinates = read_txt_file_content(txtSrc)

    # Read coordinates for predicted bounding boxes from acquired json file
    predictedCoordinates = read_json_file_coordinates(jsonSrc)

    # Calculate intersection over union for all images
    iou_values = calc_all_iou_values(
        groundTruthCoordinates, predictedCoordinates)

    # # Create bar plot for images with a single object vs images with multiple objects and their corresponding
    # # Intersection over Union scores
    # bar_plot_nr_of_objects_vs_iou(
    #     groundTruthCoordinates, iou_values)

    # # Create scatter plot for ssim vs prediction confidence for test set results
    # scatter_plot_ssim_vs_confidence(ssim_src, prediction_score_src)

    # # Create a scatter plot for ssim vs intersection over union for test set results
    # scatter_plot_ssim_vs_iou(ssim_src, iou_values)

    # Calculate quantiles used in box plots and mean, variance and std groupwise for IoU
    calculate_percentiles_and_metrics_iou(iou_values)

    # # Calculate quantiles used in box plots and mean, variance and std groupwise for prediction confidence
    # calculate_percentiles_and_metrics_prediction_confidence(prediction_score_src)

    # Create a box plot for iou values for each category of augmentation
    # box_plot_iou(iou_values)

    # # Create a box plot for iou values for each category of augmentation
    # box_plot_confidence(prediction_score_src)

    # Create a line plot with error bars for ssim vs iou
    # scatter_iou_vs_ssim_with_errors(ssim_src, iou_values)


if __name__ == "__main__":
    main()
