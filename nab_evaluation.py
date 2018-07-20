"""
    Evaluation module for point-wise prediction algorithms in terms of NAB (Numenta Anomaly Benchmark) score.
"""
import numpy as np

def scaledSigmoid(relativePositionInWindow):
    if relativePositionInWindow > 3.0:
        # FP well behind window
        return -1.0
    else:
        return 2/(np.exp(5*relativePositionInWindow) + 1) - 1.0


def label_anomaly_windows(labels):
    """
    Converts the point wise anomaly labels to windows
    :param label (type: 1D array): true labels 1/0 for each point
    :return (type: list): set of intervals of anomaly (start time, end time) sorted in increasing order of start time
    """
   
    labeled_anomaly_window = []
    #if the anomaly window starts from the beginning
    start = 0
    for i in range(1, labels.shape[0]):
        if labels[i] == 1 and labels[i-1] == 0:
            start = i
        elif labels[i-1] == 1 and labels[i] == 0:
            end = i-1
            labeled_anomaly_window.append((start, end))
        elif i == len(labels)-1 and labels[i] == 1:
            #if the anomaly window extends till the end
            labeled_anomaly_window.append((start, i))
              
    return labeled_anomaly_window


def getCorrespondingWindow(index, windows):
    """
    Finds the corresponding window to the each predicted anomaly
    :param index (type:int): index of predicted anomaly point
    :param windows (type list of tuples (start, end)): list of true anomaly windows
    :return (typ: tuple): anomaly window if point lies inside it else the preceding window
    """
    
    for i, window in enumerate(windows):
        if index <= window[1] and index >= window[0]:
            return window
        elif index > window[1] and index < windows[i+1][0]:
            return window


def add_buffer_to_label(sparsity, label_windows, min_label, max_label, window_scale_limit=2, goal_sparsity=0.01):
    """
    Add buffers to windows to produce the desired anomaly sparsity. If the current sparsity is greater 
    than the goal_sparsity, then this function does not do anything. This will not expand windows beyond
    window_scale_limit * (current window size). Merge any overlapping windows. 
    
    @param sparsity (float): The current sparsity (total length of label windows)/(length of timeseries)
    @param label_windows (list): List of (start, end) indices. Should be sorted by start index. 
    @param min_label (int): Don't allow windows to expand before this index
    @param max_label (int): Don't allow windows to expand after this index
    @param window_scale_limit (float): The largest amount the windows to be expanded by. For example, for 
                                       window_scale_limit=2, the new windows will be at most 
                                       2 * (current window size). 
    @param goal_sparsity (int): The goal sparsity of the window after increasing the window size. 
    """
    if sparsity == 0:  
        return label_windows

    new_windows = []
    expand_amount = min(goal_sparsity/sparsity, window_scale_limit) - 1
    
    if expand_amount > 0:
        for i in range(len(label_windows)):
            start = label_windows[i][0]
            end = label_windows[i][1]
            buff_amount = math.ceil(expand_amount/2 * (end - start+1))
            new_start = max(min_label, start - buff_amount)
            new_end = min(max_label, end + buff_amount)

            # Merge overlapping windows. Overlapping windows do not happen frequently and 
            # usually only overlap with adjacent window. 
            while len(new_windows) > 0 and new_windows[-1][1] >= new_start-1:
                new_start = min(new_windows[-1][0], new_start)
                new_end = max(new_windows[-1][1], new_end)
                del new_windows[-1]
            new_windows.append((new_start, new_end))
    else:
        new_windows = label_windows
    return new_windows


def nab_score(y_true, y_pred):
    """
    Computes the NAB score for evaluating the given predictions. 
    (Ref: https://arxiv.org/ftp/arxiv/papers/1510/1510.03336.pdf)
    Scoring section (i) handles TP and FN, (ii) handles FP, and TN are 0.
    (i) Calculate the score for each window. Each window will either have one
    or more true positives or no predictions (i.e. a false negative). FNs
    lead to a negative contribution, TPs a positive one.
    (ii) Go through each false positive and score it. Each FP leads to a negative
    contribution dependent on how far it is from the previous window.
    :param y_true: true labels 1/0 for each point
    :param y_pred: prediction 1/0 for each point by the algorithm
    :return: nab score
    """
    #weights for the standard profile 
    tp_weight, fp_weight, fn_weight = 1.0, 0.11, 1.0
    label_windows = label_anomaly_windows(np.array(y_true))

    sparsity = sum(y_true)/float(len(y_true))
    label_windows = add_buffer_to_label(sparsity, label_windows, 0, len(y_true))

    detection_info = {}
    for window in label_windows:
        detection_info[window] = 0
    tp_score = 0
    fp_score = 0
    fn_score = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            #if index comes before the first window or after last window
            if i < label_windows[0][0]:
                fp_score += -1.0*fp_weight
            elif i > label_windows[-1][1]:
                position = abs(label_windows[-1][1]-i)/float(label_windows[-1][1]-label_windows[-1][0])
                fp_score += scaledSigmoid(position)*fp_weight
                
            else:
                cWindow = getCorrespondingWindow(i, label_windows)
                if i <= cWindow[1] and i >= cWindow[0] and detection_info[cWindow] == 0:
                    detection_info[cWindow] = 1
                    position = -(cWindow[1]-i)/float(cWindow[1]-cWindow[0])
                    #normalization so that scaledSigmoid(-1.0) = 1 
                    tp_score += scaledSigmoid(position)*tp_weight/scaledSigmoid(-1.0)
                elif detection_info[cWindow] == 0:
                    position = abs(cWindow[1] - i)/float(cWindow[1]-cWindow[0])
                    fp_score += scaledSigmoid(position)*fp_weight
                    
    for key in detection_info:
        if detection_info[key] == 0:
            fn_score += -fn_weight
    
    return tp_score+fp_score+fn_score
