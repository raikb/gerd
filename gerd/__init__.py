'''Common abbreviations:
    res: result
    var: variable'''
import logging
import logging.config
from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)
