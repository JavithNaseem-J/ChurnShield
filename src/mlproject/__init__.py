import os
import sys
import logging

logging_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"

# Create the logs directory if it does not exist
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running.log")
os.makedirs(log_dir, exist_ok=True)

# Create handlers
file_handler = logging.FileHandler(log_filepath)
stream_handler = logging.StreamHandler(sys.stdout)

# Set the logging format for each handler
formatter = logging.Formatter(logging_str)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Get the logger and configure it
logger = logging.getLogger('mlproject')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Test the logger
logger.info("Logger has been set up successfully!")
