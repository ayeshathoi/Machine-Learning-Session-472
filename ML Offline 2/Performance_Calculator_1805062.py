
# PERFORMANCE METRICS CALCULATOR

def print_Calculation(precision, recall, specificity, f1, accuracy, false_discovery_rate):
    print("PRECISION            : ", precision)
    print("RECALL               : ", recall)
    print("SPECIFICITY          : ", specificity)
    print("F1                   : ", f1)
    print("ACCURACY             : ", accuracy)
    print("FALSE DISCOVERY RATE : ", false_discovery_rate)
    
def calculation(tn, fp, fn, tp):
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    specificity = tn / (tn + fp) if tn + fp != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    false_discovery_rate = fp / (fp + tp) if fp + tp != 0 else 0
    return print_Calculation(precision, recall, specificity, f1, accuracy, false_discovery_rate)

def performance_metrics(y_hat, y_true):
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(y_hat)):
        if y_hat[i] == 0 and y_true[i] == 0:
            tn += 1
        elif y_hat[i] == 1 and y_true[i] == 0:
            fp += 1
        elif y_hat[i] == 0 and y_true[i] == 1:
            fn += 1
        elif y_hat[i] == 1 and y_true[i] == 1:
            tp += 1
    return calculation(tn, fp, fn, tp)
