from preprocess.embed import EmbedModel
from preprocess.search import VectorIndex
from model.instruct import DSPyLanguageModel
from model.retreive import DSPyRetrieverModel
from model.RAG import DSPyRAG
import dspy
from dspy.teleprompt import BootstrapFewShot

import json
import numpy as np
from preprocess.parse import parse_folder