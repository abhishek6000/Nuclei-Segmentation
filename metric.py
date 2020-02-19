def IoU(y_true, y_pred): 
    iou = tf.py_function(calculate_iou, [y_true, y_pred], tf.float32)
    return iou