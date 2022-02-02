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

"""A bank of available tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import random
import tensorflow.compat.v1 as tf

from cognitive import stim_generator as sg
from cognitive import task_generator as tg
from cognitive.task_generator import Task
from cognitive.task_generator import TemporalTask
from cognitive import constants as const

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('task_family', 'all', 'name of the task to be trained')


class GoShape(Task):
    """Go to shape X."""

    def __init__(self, select_op_set = None):
        self.select_op = []
        shape1 = sg.random_shape()
        when1 = sg.random_when()
        objs1 = tg.Select(shape=shape1, when=when1)
        ### shape
        self.select_op.append(objs1)

        self._operator = tg.Go(objs1)

        ### todo: make it simple and consistent with others
        self.n_frames = const.LASTMAP[when1]+1

        # backtrack operator based on epoch index
        self.track_op = {}
        for slt_op in self.select_op:
            self.track_op[self.n_frames - 1 - const.ALLWHENS.index(slt_op.when)] = slt_op

    def reinit(self, select_op_index, restrictions):
        # copy and update operators
        pass
        # retrun GoShape(select_op_set)

    @property
    def instance_size(self):
        return sg.n_random_shape() * sg.n_random_when()



class GoShapeTemporal(TemporalTask):
    """Go to shape X."""

    def __init__(self):
        super(GoShapeTemporal, self).__init__()
        shape1 = sg.random_shape()
        when1 = sg.random_when()
        objs1 = tg.Select(shape=shape1, when=when1)
        self._operator = tg.Go(objs1)

        ### todo: make it simple and consistent with others
        self.n_frames = const.LASTMAP[when1]+1

    @property
    def instance_size(self):
        return sg.n_random_shape() * sg.n_random_when()

class GoShapeTemporalComposite(tg.TemporalCompositeTask):
    def __init__(self, n_tasks):
        tasks = [GoShapeTemporal() for i in range(n_tasks)]
        super(GoShapeTemporalComposite, self).__init__(tasks)
        self.n_frames = sum([task.n_frames for task in tasks])

task_family_dict = OrderedDict([
    ('GoShape', GoShape),
])


def random_task(task_family):
    """Return a random question from the task family."""
    return task_family_dict[task_family[random.randint(0,len(task_family)-1)]]()
