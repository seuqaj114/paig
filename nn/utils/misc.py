import os
import time
import inspect
import numpy as np
import zipfile

def log_metrics(logger, prefix, metrics):
    metrics_string = " ".join([k+"=%s"%metrics[k] for k in sorted(metrics.keys())])
    string = prefix + " " + metrics_string
    logger.info(string)

def classes_in_module(module):
    classes = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if obj.__module__ == module.__name__:
                classes[name] = obj
    return classes

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def zipdir(path, save_dir):
    zipf = zipfile.ZipFile(os.path.join(save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)

    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "py":
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

    zipf.close()