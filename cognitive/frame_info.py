import cognitive.stim_generator as sg
import numpy as np
import cognitive.convert as cv


class FrameInfo(object):
    def __init__(self, n_epochs, relative_tasks, task_question, task_answers, objset=None, first_shareable=None):
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
        # TODO: make first_shareable a hyper for each task
        # TODO: define p for each frame? if not shareable, then p=0
        # within shareables, p is initialized as uniform
        assert len(relative_tasks) == 1
        assert isinstance(objset, sg.ObjectSet)

        self.objset = objset

        self.frame_list = []
        self.objset = objset

        self.n_epochs = n_epochs

        for i in range(n_epochs):
            description = []
            if i == self.n_epochs - 1:
                description.append(["ending of task %d" % 0])
                self.frame_list.append(self.Frame(self, i, relative_tasks, description, task_answers))
            else:
                if i == 0:
                    description.append(["start of task %d" % 0])
                    self.frame_list.append(self.Frame(self, i, relative_tasks, description, None))

        if objset:
            # iterate all objects in objset and add to each frame
            for obj in objset:
                if obj.epoch[0] + 1 == obj.epoch[1]:
                    self.frame_list[obj.epoch[0]].objs.append(obj)
                else:
                    for epoch in range(obj.epoch[0], obj.epoch[1]):
                        self.frame_list[epoch].objs.append(obj)

        if first_shareable is None:
            self.first_shareable = np.random.choice(np.arange(0,len(self.frame_info)+1))
        else:
            self.first_shareable = first_shareable

    def __len__(self):
        return len(self.frame_list)

    def __iter__(self):
        return self.frame_list.__iter__()

    def __getitem__(self, item):
        return self.frame_list.__getitem__(item)

    def add_new_frames(self, i, relative_tasks, new_task_info):
        '''
        add new empty frames and update objset and p
        :param i: number of new frames
        :param relative_tasks: the tasks assiociated with the new frames
        :param new_task_info:
        :return:
        '''
        for j in range(i):
            self.frame_list.append(self.Frame(self,
                                              len(self.frame_list),
                                              relative_tasks
                                              ))
        self.objset.increase_epoch(len(self.objset)+i)
        self.first_shareable = len(self.frame_list)

    def get_start_frame(self, new_task_info, dist=None):
        '''
        randomly sample a starting frame for merging
        check length of both, then starting first based on first_shareable
        sample from p, if start at the same frame, but new task ends earlier,
        then start later, otherwise, new task can end earlier than existing task
        overall, task order is predetermined such that new task is after existing task
        :param: distribution of
        :return: index of frame from existing frame_info
        '''
        assert isinstance(new_task_info,cv.TaskInfoConvert)
        if dist is None:
            dist = np.array([1/len(new_task_info.frame_info)]*len(new_task_info.frame_info))

        if np.all(self.p == 0):
            return -1

        assert np.sum(self.p) == 1
        return np.random.choice(np.arange(self.first_shareable, len(new_task_info.frame_info)),dist)

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

        def __init__(self, fi, idx, relative_tasks, description=None, action=None, objs=None):
            assert isinstance(fi,FrameInfo)

            self.fi = fi
            self.idx = idx
            self.relative_tasks = relative_tasks
            self.description = description
            self.action = action
            self.objs = objs if objs else list()

            self.relative_task_epoch_idx = {}
            for task in self.relative_tasks:
                self.relative_task_epoch_idx[task] = self.idx

        def compatible_merge(self, new_frame):
            assert isinstance(new_frame, FrameInfo.Frame)

            self.relative_tasks = self.relative_tasks | new_frame.relative_tasks
            self.description = self.description + new_frame.description
            self.action = self.action + new_frame.action

            # TODO(mbai): change object epoch?
            # TODO(mbai): resolve conflict
            for old_obj, new_obj in zip(self.objs,new_frame.objs):
                self.fi.objset.add(new_obj, self.idx)

            for epoch, obj_list in self.fi.objset.dict.items():
                if epoch == self.idx:
                    self.objs = obj_list

            # update the dictionary for relative_task_epoch
            temp = self.relative_task_epoch_idx.copy()
            temp.update(new_frame.relative_task_epoch_idx)
            self.relative_task_epoch_idx = temp
