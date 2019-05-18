'''
Created on 25 Jan 2019

@author: Christian Ovesen, Kenneth Sverre Verlo Jacobsen
'''

# Allows for installing missing imports   
import subprocess as sp
import sys
from setuptools.command.easy_install import sys_executable

# Logging for error and debugging purpose
try:
    import logging
except ModuleNotFoundError:
    sp.check_call([sys.executable, '-m','pip','install','logging'])
    import keras
    logging.debug('Installed logging')
# Setting up logging, easily changeable between logging error and debug.
logging.basicConfig(datefmt='%d-%m %H:%M:%S',format='[%(asctime)s] %(levelname)s:%(message)s', level=logging.DEBUG)
# logging.basicConfig(filename='error.out', filemode='w', datefmt='%d-%m %H:%M:%S', format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR)

# GUI related imports
try:
    import tkinter
except ModuleNotFoundError:
    try:
        sp.check_call([sys.executable, '-m','pip','install','tkinter'])
        import tkinter
        logging.debug('Installed TKinter')
    except:
        logging.debug("Failed to install TKinter")
except:
    logging.debug("Unknown error importing TKinter")

# XML file read and write
try:
    import xml.etree.ElementTree as ET
except ModuleNotFoundError:
    try:
        sp.check_call([sys_executable, "-m", "pip", "install", "xml"])
        import xml.etree.ElementTree as ET
        logging.debug("Installed ElementTree")
    except:
        logging.debug("Failed to install ElementTree")
except:
    logging.debug("Unknown error importing ElementTree")
    
# Data storage and database related imports
try:
    import pathlib
except ModuleNotFoundError:
    try:
        sp.check_call([sys_executable, "-m", "pip", "install", "pathlib"])
        import pathlib
        logging.debug("Installed pathlib")
    except:
        logging.debug("Failed to install pathlib")
except:
    logging.debug("Unknown error importing pathlib")
try:
    import sqlite3
except ModuleNotFoundError:
    try:
        sp.check_call([sys_executable, "-m", "pip", "install", "sqlite3"])
        import sqlite3
        logging.debug("Installed sqlite3")
    except:
        logging.debug("Failed to install sqlite3")
except:
    logging.debug("Unknown error importing sqlite3")

# For supporting multi-dimensional arrays and matrices, and high level mathematical functions
try:
    import numpy
except ModuleNotFoundError:
    try:
        sp.check_call([sys.executable, '-m','pip','install','numpy'])
        import numpy
        logging.debug('Installed NumPy')
    except:
        logging.debug("Failed to install NumPy")
except:
    logging.debug('Unknown error importing NumPy')

# AI releated imports
try:
    import tensorflow
    from tensorflow import keras as kerasTwo
except ModuleNotFoundError:
    try:
        sp.check_call([sys.executable, '-m','pip','install','tensorflow'])
        import tensorflow
        from tensorflow import keras as kerasTwo
        logging.debug('Installed tensorflow')
    except:
        logging.debug("Failed to install tensorflow")
except:
    logging.error('Unknown error importing tensorflow')
try:
    import theano
except ModuleNotFoundError:
    try:
        sp.check_call([sys.executable, '-m','pip','install','theano'])
        import theano
        logging.debug('Installed theano')
    except:
        logging.debug("Failed to install theano")
except:
    logging.error('Unknown error importing theano')