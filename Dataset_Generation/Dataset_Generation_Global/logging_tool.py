import logging
from inspect import currentframe 

def Setup_Logger(name, log_file, level=logging.INFO):
    """
    name: logger name, ex) 'first logger'
    log_file: save path

    Example---------------------------
    # first file logger
    logger = Setup_Logger('first_logger', 'first_logfile.log')
    logger.info('This is just info message')

    # second file logger
    super_logger = Setup_Logger('second_logger', 'second_logfile.log')
    super_logger.error('This is an error message')
    ----------------------------------
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def Line_Checking():
    cf = currentframe()
    print(f"No issue at the current Line: {cf.f_lineno}")  

if __name__ == '__main__':
    logger = Setup_Logger('test', '/home/han/test.log')
    logger.info("efkwjfekwj")

    logger_2 = Setup_Logger('test2', '/home/han/test_2.log')
    logger_2.info("wefhkae")

