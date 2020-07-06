import os
import numpy as np
import glob
import json

def extract_ssim_vs_confidence_groups(ssim_src, prediction_src):

    with open(ssim_src) as f:
        data = json.load(f)

    with open(prediction_src) as f:
        pred_data = json.load(f)

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

    for key in ssim_keys:
        for nested_key in [*data[key]]:

            if nested_key in pred_keys:

                if len(pred_data[nested_key]) is not 0:

                    value = sum(pred_data[nested_key]) / \
                        len(pred_data[nested_key])

                if "_gn_01" in nested_key:
                    g1[1].append(value)
                    g1[0].append(data[key][nested_key])

                elif "_gn_02" in nested_key:
                    g2[1].append(value)
                    g2[0].append(data[key][nested_key])

                elif "_gn_03" in nested_key:
                    g3[1].append(value)
                    g3[0].append(data[key][nested_key])

                elif "_ga_03" in nested_key:
                    g4[1].append(value)
                    g4[0].append(data[key][nested_key])

                elif "_ga_07" in nested_key:
                    g5[1].append(value)
                    g5[0].append(data[key][nested_key])

                elif "_ga_20" in nested_key:
                    g6[1].append(value)
                    g6[0].append(data[key][nested_key])

                elif "_ga_30" in nested_key:
                    g7[1].append(value)
                    g7[0].append(data[key][nested_key])

                elif "_gr" in nested_key:
                    g8[1].append(value)
                    g8[0].append(data[key][nested_key])

    data = (g1, g2, g3, g4, g5, g6, g7, g8)
    return data


def extract_ssim_vs_iou_groups(ssim_src, iou_src):

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

    for key in ssim_keys:
        for nested_key in [*data[key]]:

            if nested_key in iou_keys:
                iou = iou_src[nested_key]
                ssim = data[key][nested_key]

                if "_gn_01" in nested_key:
                    g1[1].append(iou)
                    g1[0].append(ssim)

                elif "_gn_02" in nested_key:
                    g2[1].append(iou)
                    g2[0].append(ssim)

                elif "_gn_03" in nested_key:
                    g3[1].append(iou)
                    g3[0].append(ssim)

                elif "_ga_03" in nested_key:
                    g4[1].append(iou)
                    g4[0].append(ssim)

                elif "_ga_07" in nested_key:
                    g5[1].append(iou)
                    g5[0].append(ssim)

                elif "_ga_20" in nested_key:
                    g6[1].append(iou)
                    g6[0].append(ssim)

                elif "_ga_30" in nested_key:
                    g7[1].append(iou)
                    g7[0].append(ssim)

                elif "_gr" in nested_key:
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

    data = (g1, g2, g3, g4, g5, g6, g7, g8, g1_avg, g2_avg, g3_avg, g4_avg, g5_avg, g6_avg, g7_avg, g8_avg)

    return data


def extract_confidence_groups(confidence_src):
    with open(confidence_src) as f:
        data = json.load(f)

    confidence_keys = sorted([*data])

    g1 = []
    g2 = []
    g3 = []
    g4 = []
    g5 = []
    g6 = []
    g7 = []
    g8 = []
    g9 = []

    for key in confidence_keys:

        confidence = data[key]

        if "_gn_01" in key:
            for value in confidence:
                g1.append(value)

        elif "_gn_02" in key:
            for value in confidence:
                g2.append(value)

        elif "_gn_03" in key:
            for value in confidence:
                g3.append(value)

        elif "_ga_03" in key:
            for value in confidence:
                g4.append(value)

        elif "_ga_07" in key:
            for value in confidence:
                g5.append(value)

        elif "_ga_20" in key:
            for value in confidence:
                g6.append(value)

        elif "_ga_30" in key:
            for value in confidence:
                g7.append(value)

        elif "_gr" in key:
            for value in confidence:
                g8.append(value)

        else:
            for value in confidence:
                g9.append(value)

    data = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    return data


def extract_iou_groups(iou_src):

    # extract dictionary keys
    iou_keys = sorted([*iou_src])

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

    data = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    return data
