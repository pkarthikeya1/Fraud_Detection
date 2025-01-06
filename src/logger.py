import os
import logging
import sys


LOGGING_STR  =  "[%(asctime)s, %(levelname)s, %(module)s, %(message)s]"

logging_dir = "logs"

os.makedirs(logging_dir, exist_ok=True)

file_name= os.path.join(logging_dir, 'logfile.log')

logging.basicConfig(
    level=logging.INFO,
    format= LOGGING_STR,
    filename= file_name,
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(file_name)
    ]
)

logger = logging.getLogger("Fraud_Detection_Logger")