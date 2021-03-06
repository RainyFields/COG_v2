# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Store all the constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import string
import os
import numpy as np

AVG_MEM = 4
MAX_MEMORY = 4
LASTMAP = {}
for k in range(MAX_MEMORY + 1):
    LASTMAP["last%d" % (k)] = k

ALLWHENS = []
for k in range(MAX_MEMORY + 1):
    ALLWHENS.append("last%d" % (k))
ALLWHENS_PROB = [1 / (MAX_MEMORY + 1)] * len(ALLWHENS)


def compare_when(when_list):
    return max(list(map(lambda x: LASTMAP[x], when_list)))


def get_target_value(t):
    # Convert target t to string and convert True/False target values
    # to lower case strings for consistency with other uses of true/false
    # in vocabularies.
    t = t.value if hasattr(t, 'value') else str(t)
    if t is True or t == 'True':
        return 'true'
    if t is False or t == 'False':
        return 'false'
    return t


# RGB, from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
WORD2COLOR = {
    'red': (230, 25, 75),
    'green': (60, 180, 75),
    'blue': (0, 130, 200),
    'yellow': (255, 225, 25),
    'purple': (145, 30, 180),
    'orange': (245, 130, 48),
    'cyan': (70, 240, 240),
    'magenta': (240, 50, 230),
    'lime': (210, 245, 60),
    'pink': (250, 190, 190),
    'teal': (0, 128, 128),
    'lavender': (230, 190, 255),
    'brown': (170, 110, 40),
    'beige': (255, 250, 200),
    'maroon': (128, 0, 0),
    'mint': (170, 255, 195),
    'olive': (128, 128, 0),
    'coral': (255, 215, 180),
    'navy': (0, 0, 128),
    'grey': (128, 128, 128),
    'white': (255, 255, 255)
}

OBJECT2ID = {
    "watercraft": 0,
    "rifle": 3,
    "display": 4,
    "lamp": 5,
    "speaker": 6,
    "cabinet": 7,
    "chair": 8,
    "bench": 9,
    "car": 10,
    "airplane": 11,
    "sofa": 12,
    "table": 1,
    "phone": 2
}

ALLSPACES = ['left', 'right', 'top', 'bottom']
ALLCOLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
ALLSHAPES = ['circle', 'square', 'cross', 'triangle', 'vbar', 'hbar']
ALLOBJECTS = list(OBJECT2ID.keys())
# Comment out the following to use a smaller set of colors and shapes
# ALLCOLORS += [
#     'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige',
#     'maroon', 'mint', 'olive', 'coral', 'navy', 'grey', 'white']
ALLSHAPES += list(string.ascii_lowercase)
ALLSHAPES += list(OBJECT2ID.keys())

ALLCOLORSHAPES = [x for x in itertools.product(ALLCOLORS, ALLSHAPES)]

# When the stimuli are invalid for a task
INVALID = 'invalid'

# Allowed vocabulary, the first word is invalid
INPUTVOCABULARY = [
                      'invalid',
                      '.', ',', '?',
                      'object', 'color', 'shape',
                      'loc', 'on',
                      'if', 'then', 'else',
                      'exist',
                      'equal', 'and',
                      'the', 'of', 'with',
                      'point',
                  ] + ALLSPACES + ALLCOLORS + ALLSHAPES + ALLWHENS
# For faster str -> index lookups
INPUTVOCABULARY_DICT = dict([(k, i) for i, k in enumerate(INPUTVOCABULARY)])

INPUTVOCABULARY_SIZE = len(INPUTVOCABULARY)

OUTPUTVOCABULARY = ['true', 'false'] + ALLCOLORS + ALLSHAPES

# Maximum number of words in a sentence
MAXSEQLENGTH = 25
dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        'data')
if not os.path.exists(dir_path):
    print('Data folder does not exist.')

# If use popvec out_type
def get_prefs(grid_size):
    prefs_y, prefs_x = (np.mgrid[0:grid_size, 0:grid_size]) / (grid_size - 1.)
    prefs_x = prefs_x.flatten().astype('float32')
    prefs_y = prefs_y.flatten().astype('float32')

    # numpy array (Grid_size**2, 2)
    prefs = (np.array([prefs_x, prefs_y]).astype('float32')).T
    return prefs


GRID_SIZE = 7
PREFS = get_prefs(GRID_SIZE)

config = {'dataset': 'yang',
          'pnt_net': True,
          'in_voc_size': len(INPUTVOCABULARY),
          'grid_size': GRID_SIZE,
          'out_voc_size': len(OUTPUTVOCABULARY),
          'maxseqlength': MAXSEQLENGTH,
          'prefs': PREFS,
          }
