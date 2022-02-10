import numpy as np

import cognitive.stim_generator as sg
import cognitive.task_generator as tg
import cognitive.task_bank as task_bank
from cognitive.helper import get_target_value


class TaskInfoCombo(object):
    """
    Storage of composition information,
    including task_frame_info, task_example, task and objset
    :param task_frame_info: FrameInfo
    :param task_example: including 'family'(task_family), 'epochs'(epochs),'question','objects' and 'answer'
    :param task: task family
    :param objset: objset
    """

    def __init__(self, frame_info = None, task_example=None, task=None, objset=None):
        " combining with a second task should be implemented incrementally"
        if frame_info is None or objset is None:
            raise ValueError("task information is incomplete")
        else:
            assert isinstance(frame_info, fi.FrameInfo)
            assert isinstance(objset, sg.ObjectSet)
            self.frame_info = frame_info
            self.objset = objset
            self.task_example = task_example
            self.task = task

    def __len__(self):
        # return number of tasks involved
        return len(self.frame_info)

    @property
    def n_epochs(self):
        return len(self.frame_info)


class FrameInfo(object):
    def __init__(self, first_shareable=None, task_example=None, task=None, objset=None):
        """
        used for combining multiple temporal tasks, initialize with 1 task,
        stores each frame object in frame_list
        :param first_shareable: the frame at which the task is first shareable.
                                if the task is non-shareable, first_shareable = len(task)
                                if no input, start at random frame, including the possibility of non-shareable
        :param task_example: example dict from main.generate_example
        :param task: stim_generator.Task object
        :param objset: objset related to the task
        """
        # TODO: first beginning shareable flag
        # TODO: make first_shareable a hyper for each task
        # TODO: define p for each frame? if not shareable, then p=0

        if task is None or objset is None:
            if task_example is None:
                raise ValueError("no tasks is provided")
            else:
                n_epochs = task_example["epochs"]
                task_question = task_example["question"]
                task_answers = task_example["answers"]
        else:
            assert isinstance(objset, sg.ObjectSet)
            assert isinstance(task, tg.TemporalTask)

            if task.n_frames != objset.n_epoch:
                raise ValueError('Task epoch does not equal objset epoch')

            n_epochs = task.n_frames
            # TODO: decide if task_question needs to be kept
            task_question = str(task)
            task_answers = [get_target_value(t) for t in task.get_target(objset)]

        relative_tasks = {0}
        self.objset = objset
        self.frame_list = list()
        self.n_epochs = n_epochs

        for i in range(n_epochs):
            description = list()
            if i == self.n_epochs - 1:
                description.append(["ending of task %d" % 0])
                self.frame_list.append(self.Frame(self, i, relative_tasks, description, task_answers))
            else:
                if i == 0:
                    description.append(["start of task %d" % 0])
                self.frame_list.append(self.Frame(self, i, relative_tasks, description))

        if objset:
            # iterate all objects in objset and add to each frame
            for obj in objset:
                if obj.epoch[0] + 1 == obj.epoch[1]:
                    self.frame_list[obj.epoch[0]].objs.append(obj)
                else:
                    for epoch in range(obj.epoch[0], obj.epoch[1]):
                        self.frame_list[epoch].objs.append(obj)

        if first_shareable is None:
            self.first_shareable = np.random.choice(np.arange(0, len(self.frame_list) + 1))
        else:
            self.first_shareable = first_shareable

        self.last_task = 0
        self.last_task_end = 0

    def __len__(self):
        return len(self.frame_list)

    def __iter__(self):
        return self.frame_list.__iter__()

    def __getitem__(self, item):
        return self.frame_list.__getitem__(item)

    def add_new_frames(self, i, relative_tasks):
        '''
        add new empty frames and update objset and p
        :param i: number of new frames
        :param relative_tasks: the tasks associated with the new frames
        :return:
        '''
        if i <= 0:
            return
        for j in range(i):
            self.frame_list.append(self.Frame(self,
                                              len(self.frame_list),
                                              relative_tasks
                                              ))
        self.objset.increase_epoch(self.objset.n_epoch + i)

    def get_start_frame(self, new_task_info, relative_tasks):
        '''
        randomly sample a starting frame to start merging add new frames if needed
        check length of both, then starting first based on first_shareable
        sample from p, if start at the same frame, but new task ends earlier,
        then start later, otherwise, new task can end earlier than existing task
        overall, task order is predetermined such that new task appears or finishes after the existing task
        avoid overlapping response frames
        :param dist: distribution of the
        :param new_task_info:
        :return: index of frame from existing frame_info
        '''
        assert isinstance(new_task_info, TaskInfoCombo)

        # if multiple tasks are shareable, then start from the last task
        # to maintain preexisting order

        first_shareable = self.first_shareable

        shareable_frames = self.frame_list[first_shareable:]

        new_task_len = new_task_info.n_epochs
        # TODO: provide first_shareable from new_task
        # new_task_first_shareable = new_task_info.first_shareable

        if len(shareable_frames) == 0:
            # queue, add frames and start merging
            assert first_shareable == len(self.frame_list)

            self.add_new_frames(new_task_len, relative_tasks)
            self.last_task_end = len(self.frame_list) - 1
            self.last_task = list(relative_tasks)[0]
            return first_shareable
        else:
            # add more frames
            if len(shareable_frames) == new_task_len:
                self.add_new_frames(1, relative_tasks)
                print('added one frame')
                first_shareable += 1
            else:
                self.add_new_frames(new_task_len - len(shareable_frames), relative_tasks)
                # print(f'added {new_task_len - len(shareable_frames)} frames')
            shareable_frames = self.frame_list[first_shareable:]
            # first check if start of alignment
            # if the alignment starts with start of existing task, then check if new task ends earlier
            aligned = False
            while not aligned:
                # inserting t3 into T = {t1,t2} (all shareable) in cases where t1 ends before t2 but appears before t2,
                # last_task_start is after t1, last_task_end is after t1.
                # suppose shareable_frames[new_task_len] is the index of end of t1
                # then shift to the right, at next loop, t2 ends, then add new frame

                # check where the response frame of the new task would be in shareable_frames
                # if the response frames are overlapping, then shift alignment to "right"
                if any('end of task' in d for d in shareable_frames[new_task_len - 1].description):
                    first_shareable += 1
                    print('response frame overlap')
                # if the sample frames are overlapping, then check if the new task ends before the last task
                # if so, then shift starting frame
                elif any('start of task {self.last_task}' in d for d in self.frame_list[first_shareable].description) \
                        and first_shareable + new_task_len - 1 <= self.last_task_end:
                    # TODO: check last_task_end
                    first_shareable += 1
                    print('sample frame overlap')
                else:
                    aligned = True

                if len(shareable_frames) < new_task_len:
                    self.add_new_frames(1, relative_tasks)
                shareable_frames = self.frame_list[first_shareable:]
            self.last_task_end = first_shareable + new_task_len - 1
            self.last_task = list(relative_tasks)[0]

            return first_shareable

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

        def __init__(self, fi, idx, relative_tasks, description=list(), action=list(), objs=None):
            assert isinstance(fi, FrameInfo)

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

            # TODO(mbai): change object epoch, add new input to objset.add(merge_idx)
            # TODO(mbai): resolve conflict
            for new_obj in new_frame.objs:
                self.fi.objset.add(new_obj, self.idx, merge_idx=self.idx)

            for epoch, obj_list in self.fi.objset.dict.items():
                if epoch == self.idx:
                    self.objs = obj_list

            # update the dictionary for relative_task_epoch
            temp = self.relative_task_epoch_idx.copy()
            temp.update(new_frame.relative_task_epoch_idx)
            self.relative_task_epoch_idx = temp

        def __str__(self):
            return 'frame: ' + str(self.idx) + ', relative tasks: ' + \
                   ','.join([str(i) for i in self.relative_tasks])\
                   + ' objects: ' + ','.join([str(o) for o in self.objs])


