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
# TODO: n_epochs
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


class ExistShapeOf(Task):
    """Check if exist object with shape of a colored object."""

    def __init__(self):
        color1, color2 = sg.sample_color(2)
        objs1 = tg.Select(color=color1, when=sg.random_when())
        shape1 = tg.GetShape(objs1)
        objs2 = tg.Select(color=color2, shape=shape1, when='now')
        self._operator = tg.Exist(objs2)
        self.n_frames = 4

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * sg.n_random_when()


class GoShape(Task):
    """Go to shape X."""

    def __init__(self, select_op_set=None):
        self.select_collection = []
        shape1 = sg.random_shape()
        when1 = sg.random_when()
        objs1 = tg.Select(shape=shape1, when=when1)
        inherent_attr = {"shape": shape1,
                         "when":when1}


        self._operator = tg.Go(objs1)
        select_tuple = (when1, inherent_attr, objs1, self._operator, "Go")
        self.select_collection.append(select_tuple)

        ### todo: make it simple and consistent with others
        self.n_frames = const.LASTMAP[when1] + 1

        # update epoch index
        for select_op in self.select_collection:
            select_op[0] = self.n_frames - 1 - const.ALLWHENS.index(select_op[0])

    ##### todo: maybe move it to Task class? how to deal with self._operator?
    def reinit(self, select_epoch_index, restrictions):
        for i, epoch_index in enumerate(select_epoch_index):
            select_index = self.op_index(epoch_index,)
            # update inherent attributes based on restrictions
            self.select_collection[select_index][2].update(self.select_collection[select_index][1],restrictions[i])
            # update _operator
            if self.select_collection[select_index][4] == "Go":
                self.select_collection[select_index][3] = tg.Go(self.select_collection[select_index][2])

    def op_index(self, epoch_index):
        for i, select_op in enumerate(self.select_collection):
            if epoch_index == select_op[0]:
                return i
        return "no select operator is found"

    @property
    def instance_size(self):
        return sg.n_random_shape() * sg.n_random_when()


class GoShapeTemporal(TemporalTask):
    """Go to shape X."""

    def __init__(self, n_frames):
        super(GoShapeTemporal, self).__init__(n_frames)
        shape1 = sg.random_shape()
        when1 = sg.random_when()
        objs1 = tg.Select(shape=shape1, when=when1)
        self._operator = tg.Go(objs1)

        ### todo: make it simple and consistent with others
        self.n_frames = const.LASTMAP[when1] + 1
        self.n_frames = n_frames
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
    ('ExistShapeOf', ExistShapeOf)
])


def random_task(task_family):
    """Return a random question from the task family."""
    return task_family_dict[task_family[random.randint(0, len(task_family) - 1)]]()
