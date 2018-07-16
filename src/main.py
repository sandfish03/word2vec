import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy.random as nr
import sys
import h5py
import math

from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.initializers import global_uniform
from keras.initializers import uniform
from keras.optimizers import RMSprop
from keras.utils import np_utils
