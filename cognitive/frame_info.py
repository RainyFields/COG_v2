import stim_generator as sg
import numpy as np

class FrameInfo(object):
    def __init__(self, n_epochs, relative_tasks,  task_question, task_answers, objset=None, shareable = True):
        """
        used for combining multiple temporal tasks, initialize with 1 task
        stores each frame object in frame_list
        :param n_epochs: int, number of epochs of the task
        :param shareable: whether the task is shareable
        :param task_question: string
        :param task_answers: list of string of a task target value
        :param relative_tasks: set of tasks that use the frame
        :param objset: objset related to the task
        """
        # TODO: first beginning shareable flag
        # TODO: define p for each frame? if not shareable, then p=0
        # within shareables, p is initialized as uniform
        assert len(relative_tasks) == 1
        self.objset = objset

        self.frame_list = []

        self.n_epochs = n_epochs
        if shareable:
            self.first_shareable = 0
        else:
            self.first_shareable = -1
        for i in range(n_epochs):
            description = []
            if i == self.n_epochs - 1:
                description.append(["ending of task %d" % 0])
                self.frame_list.append(self.Frame(i, shareable, relative_tasks, description, task_answers))
            else:
                if i == 0:
                    description.append(["start of task %d" % 0])
                    self.frame_list.append(self.Frame(i, shareable, relative_tasks, description, None))

        if objset:
            for obj in objset:
                if obj.epoch[0] + 1 == self.epoch[1]:
                    self.frame_list[obj.epoch[0]].objs.add(obj)
                else:
                    for epoch in range(obj.epoch[0], obj.epoch[1]):
                        self.frame_list[epoch].objs.add(obj)

        self.p = np.zeros(len(self.frame_list))

    def __len__(self):
        return len(self.frame_list)

    def __iter__(self):
        return self.frame_list.__iter__()

    def __getitem__(self, item):
        return self.frame_list.__getitem__(item)

    def get_start_frame(self):
        return np.random.choice(self.frame_list,self.p)

    # @property
    # def first_shareable(self):
    #     if self.first_shareable == -1:
    #         return self.first_shareable
    #     assert self.first_shareable == next(i for i, frame in enumerate(self.frame_list) if frame.shareable)
    #     return self.first_shareable
    #
    # @first_shareable.setter
    # def first_shareable(self, idx = 0):
    #     self.first_shareable = idx

    class Frame(object):
        """ frame object within frame_info list"""

        def __init__(self, idx, relative_tasks, shareable, description=None, action=None, objs=None):
            self.idx = idx
            self.relative_tasks = relative_tasks
            self.description = description
            self.action = action
            self.objs = set()

            self.relative_task_epoch_idx = {}
            for task in self.relative_tasks:
                self.relative_task_epoch_idx[task] = self.idx

        def merge(self, new_frame):
            assert isinstance(new_frame, FrameInfo.Frame)
            self.idx = max(self.idx, new_frame.idx)
            self.relative_tasks = self.relative_tasks | new_frame.relative_tasks
            self.description = self.description + new_frame.description
            self.action = self.action + new_frame.action

            # TODO(mbai): change object epoch?
            # TODO(mbai): resolve conflict
            self.objs = self.objs | new_frame.objs
            temp = self.relative_task_epoch_idx.copy()
            temp.update(new_frame.relative_task_epoch_idx)
            self.relative_task_epoch_idx = temp
