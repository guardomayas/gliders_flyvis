import torch
import numpy as np
import flyvis
import pandas as pd
# from flyvis.datasets.rendering import BoxEye
from flyvis.analysis.visualization import plt_utils, plots
from flyvis.analysis import animations
from flyvis.datasets.datasets import SequenceDataset

from typing import List
from tqdm import tqdm


from pathlib import Path
from datamate import root, Directory

import matplotlib.pyplot as plt
import seaborn as sns


# sns.set_context('notebook')