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

"""Script for generating a COG dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import functools
import gzip
import itertools
import json
import multiprocessing
import os
import random
import shutil
import traceback

import numpy as np
import tensorflow.compat.v1 as tf

from cognitive import stim_generator as sg
import cognitive.task_bank as task_bank
from cognitive import task_generator as tg
from cognitive.convert import TaskInfoConvert

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('max_memory', 3, 'maximum memory duration')
tf.flags.DEFINE_integer('max_distractors', 1, 'maximum number of distractors')
tf.flags.DEFINE_integer('epochs', 4, 'number of epochs')
tf.flags.DEFINE_boolean('compress', True, 'whether to gzip the files')
tf.flags.DEFINE_integer('examples_per_family', 2,
                        'number of examples to generate per task family')
tf.flags.DEFINE_string('output_dir', '/tmp/cog',
                       'Directory to write output (json or tfrecord) to.')
tf.flags.DEFINE_integer('parallel', 0,
                        'number of parallel processes to use. Only training '
                        'dataset is generated in parallel.')

try:
    range_fn = xrange  # py 2
except NameError:
    range_fn = range  # py 3


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


def generate_example(max_memory, max_distractors, task_family):
    # random.seed(1)
    task = task_bank.random_task(task_family)
    epochs = task.n_frames

    # To get maximum memory duration, we need to specify the following average
    # memory value
    avg_mem = round(max_memory / 3.0 + 0.01, 2)
    if max_distractors == 0:
        objset = task.generate_objset(n_epoch=epochs,
                                      average_memory_span=avg_mem)
    else:
        objset = task.generate_objset(n_epoch=epochs,
                                      n_distractor=random.randint(1, max_distractors),
                                      average_memory_span=avg_mem)
    # Getting targets can remove some objects from objset.
    # Create example fields after this call.
    targets = task.get_target(objset)
    example = {
        'family': task_family,
        'epochs': epochs,  # saving an epoch explicitly is needed because
        # there might be no objects in the last epoch.
        'question': str(task),
        'objects': [o.dump() for o in objset],
        'answers': [get_target_value(t) for t in targets]
    }
    return example, objset, task


def generate_temporal_example(max_memory, max_distractors, n_tasks):

    # sample n_tasks
    families = list(task_bank.task_family_dict.keys())

    task_examples = []
    task_objsets = []
    task_list = []
    for i in range(n_tasks):
        example, objset, task = generate_example(max_memory, max_distractors, families)
        task_examples.append(example)
        task_objsets.append(objset)
        task_list.append(task)

    # temporal combination
    combo_frames = None
    for i, task in enumerate(task_list):
        example_current_task= task_examples[i]
        if combo_frames == None:
            combo_frames = TaskInfoConvert(example_current_task)
        combo_frames = combo_frames.merge(example_current_task)

    return combo_frames


def log_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            print('Exception in ' + func.__name__)
            traceback.print_exc()
            raise e

    return wrapped_func


def main(argv):
    # go shape: point at last sth

    max_distractors = 0
    max_memory = 10
    families = list(task_bank.task_family_dict.keys())
    task_family = families[0]

    # example, objset, task = generate_example(max_memory, max_distractors, task_family)
    # frameinfo = TaskInfoConvert(example)
    # print(frameinfo)
    # print("example", example)
    # print("objset", objset)
    # print("task", task)

    combo_frames = generate_temporal_example(max_memory,max_distractors,1)
    print(combo_frames)

if __name__ == '__main__':
    tf.app.run(main)
