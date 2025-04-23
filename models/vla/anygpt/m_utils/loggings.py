import os
import logging


class HttpxFilter(logging.Filter):
    def filter(self, record):
        if record.name == "httpx":
            return False
        return True

    
def get_logger(local_rank, save_path, log_level='info', log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', **kwargs):
    logger = logging.getLogger('logger')
    levels = {
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }
    print('logger file saved to : ', save_path)
    if log_level.lower() not in levels:
        log_level = 'info'
        print(f"Invalid log level, set to {log_level}")
    else:
        log_level = log_level.lower()
    
    logger.setLevel(levels[log_level])
    
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(levels[log_level])
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if local_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(levels[log_level])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.addFilter(HttpxFilter())
    return logger