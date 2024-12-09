import logging
import time
import os
import os.path as osp

def get_logger(log_dir="./logs"):
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(name='DPS')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler for logging to a file
    localtime = time.strftime("%Y_%m_%d_%H_%M_%S")
    log_file = osp.join(log_dir, localtime + ".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger