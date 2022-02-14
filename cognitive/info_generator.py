import numpy as np

import cognitive.constants as CONST
import cognitive.stim_generator as sg
import cognitive.task_generator as tg
from cognitive.helper import get_target_value
import cognitive.constants as CONST


class TaskInfoCompo(object):
    """
    Storage of composition information,
    including task_frame_info, task_example, task and objset
    :param frame_info: FrameInfo
    :param task_example: including 'family'(task_family), 'epochs'(epochs),'question','objects' (objset) and 'answer'
    :param task: task family
    """

    def __init__(self, task, frame_info=None):
        " combining with a second task should be implemented incrementally"
        assert task is not None
        assert isinstance(task, tg.TemporalTask)

        self.tasks = [task]

        if frame_info is None:
            objset = task.generate_objset(task.n_frames)
            self.frame_info = FrameInfo(task, objset)

    def __len__(self):
        # return number of tasks involved
        return len(self.frame_info)

    def get_example(self, task_idx):
        task = self.tasks[task_idx]
        objset = task.generate_objset(task.n_frames)
        targets = task.get_target(objset)
        example = {
            'family': task.__class__.__name__,
            'epochs': task.n_frames,  # saving an epoch explicitly is needed because
            # there might be no objects in the last epoch.
            'question': str(task),
            'objects': [o.dump() for o in objset],
            'answers': [CONST.get_target_value(t) for t in targets],
            'first_shareable': task.first_shareable
        }
        return example

    @property
    def n_epochs(self):
        return len(self.frame_info)

    def merge(self, new_task_info, reuse=None):
        '''

        :param new_task_info: TaskInfoCombo object
        :return: None if no change, and the new task if merge needed change
        '''
        # TODO(mbai): change task instruction here
        # TODO: location conflict, feature (shape, color) conflict

        assert isinstance(new_task_info, TaskInfoCompo)
        ##### todo: where did we store the number of tasks?
        if len(new_task_info.tasks) > 1:
            raise NotImplementedError('Currently cannot support adding new composite tasks')

        if reuse is None:
            reuse = 0.5

        # correct task index in new_task_info
        #### todo: what is going on here?
        next_task_idx = len(self.tasks)
        for frame in new_task_info.frame_info:
            # todo: what if the old frame is associated with the previous task? # I updated the previous task frame relative_task with new task_index
            # frame.relative_tasks = [next_task_idx + i for i, _ in enumerate(frame.relative_tasks)]
            frame.relative_tasks = [next_task_idx]
            for i, task in enumerate(frame.relative_tasks):
                # frame.relative_task_epoch_idx[task] =
                frame.relative_task_epoch_idx[next_task_idx + i] = frame.relative_task_epoch_idx[task].pop()
                task = next_task_idx + i

        start = self.frame_info.get_start_frame(new_task_info, relative_tasks=[0])

        curr_abs_idx = start

        for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start:], new_task_info.frame_info)):
            assert isinstance(old_frame, FrameInfo.Frame)
            assert isinstance(new_frame, FrameInfo.Frame)
            # if there are no objects in the frame, then freely merge
            if not old_frame.objs or not new_frame.objs:
                try:
                    old_frame.compatible_merge(new_frame)  # always update frame descriptions
                except:
                    continue
            else:
                if np.random.random() < reuse:  # use curr frame info from previous task and reinit new task
                    attr_expected = dict()
                    # random select one obj from old frame objs
                    old_obj = np.random.choice(old_frame.objs)
                    attr_expected["loc"] = old_obj.loc
                    attr_expected["when"] = old_obj.when
                    attr_expected["shape"] = old_obj.shape
                    attr_expected["color"] = old_obj.color

                    new_task_info.tasks[0].reinit(i, attr_expected)
                    new_task_info.task_info[-1] = str(new_task_info.curr_task)
                    new_task_info.objset = new_task_info.curr_task.generate_objset(n_epoch=new_task_info.task.n_frames,
                                                                                   average_memory_span=CONST.AVG_MEM)
                    new_task_info.example["question"] = str(new_task_info.curr_task)
                    new_task_info.example["objects"] = [o.dump() for o in
                                                        new_task_info.objset]  ## todo: has this been updated?
                    targets = new_task_info.curr_task.get_target(self.frame_info.objset)
                    new_task_info.example["answers"] = [get_target_value(t) for t in targets]
                    ##### todo: does compatible merge include updating frame_info?
                    old_frame.compatible_merge(new_task_info.frame_info[i])
                else:
                    old_frame.compatible_merge(new_frame)

            curr_abs_idx += 1


class FrameInfo(object):
    def __init__(self, task_example=None, task=None, objset=None):
        """
        used for combining multiple temporal tasks, initialize with 1 task,
        stores each frame object in frame_list
        :param task_example: example dict from main.generate_example
        :param task: stim_generator.Task object
        :param objset: objset related to the task
        """
        if task is None or objset is None:
            if task_example is None:
                raise ValueError("no tasks is provided")
            else:
                n_epochs = task_example["epochs"]
                task_question = task_example["question"]
                task_answers = task_example["answers"]
                first_shareable = task_example['first_shareable']
        else:
            assert isinstance(objset, sg.ObjectSet)
            assert isinstance(task, tg.TemporalTask)

            if task.n_frames != objset.n_epoch:
                raise ValueError('Task epoch does not equal objset epoch')

            n_epochs = task.n_frames
            # TODO: decide if task_question needs to be kept
            task_question = str(task)
            task_answers = [get_target_value(t) for t in task.get_target(objset)]
            first_shareable = task.first_shareable

        relative_tasks = [0]  ### todo: why it was originally init as dict?
        self.objset = objset
        self.frame_list = list()
        self.n_epochs = n_epochs

        for i in range(n_epochs):
            description = list()
            if i == self.n_epochs - 1:
                description.append(["ending of task %d" % 0])
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

        self.first_shareable = first_shareable

        self.last_task = 0
        self.last_task_end = len(self.frame_list) - 1
        self.last_task_start = 0

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
        print('added {i} more frames.')
        self.objset.increase_epoch(self.objset.n_epoch + i)

    def get_start_frame(self, new_task_info, relative_tasks):
        '''
        randomly sample a starting frame to start merging add new frames if needed
        check length of both, then starting first based on first_shareable
        sample from p, if start at the same frame, but new task ends earlier,
        then start later, otherwise, new task can end earlier than existing task
        overall, task order is predetermined such that new task appears or finishes after the existing task
        avoid overlapping response frames
        :param new_task_info:
        :return: index of frame from existing frame_info
        '''
        assert isinstance(new_task_info, TaskInfoCompo)

        # if multiple tasks are shareable, then start from the last task
        # to maintain preexisting order

        first_shareable = self.first_shareable

        shareable_frames = self.frame_list[first_shareable:]

        new_task_len = new_task_info.n_epochs

        # TODO: make task_info only contain tasks?
        new_first_shareable = new_task_info.tasks[0].first_shareable

        if len(shareable_frames) == 0:
            # queue, add frames and start merging
            assert first_shareable == len(self.frame_list)

            self.add_new_frames(new_task_len, relative_tasks)
            self.last_task_end = len(self.frame_list) - 1
            self.last_task = list(relative_tasks)[0]
            self.first_shareable = new_first_shareable + first_shareable
            return first_shareable
        else:
            # add more frames
            if len(shareable_frames) == new_task_len:
                self.add_new_frames(1, relative_tasks)
                first_shareable += 1
            else:
                self.add_new_frames(new_task_len - len(shareable_frames), relative_tasks)

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
                if first_shareable + new_task_len - 1 == self.last_task_end:
                    first_shareable += 1
                    print('response frame overlap')
                # if the sample frames are overlapping, then check if the new task ends before the last task
                # if so, then shift starting frame
                elif first_shareable == self.last_task_start \
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
            self.last_task_start = first_shareable

            self.first_shareable = new_first_shareable + first_shareable
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

        def __init__(self, fi, idx, relative_tasks, description=None, action=None, objs=None):
            assert isinstance(fi, FrameInfo)

            self.fi = fi
            self.idx = idx
            self.relative_tasks = relative_tasks
            self.description = list() if None else description
            self.action = list() if None else action
            self.objs = objs if objs else list()

            self.relative_task_epoch_idx = {}
            for task in self.relative_tasks:
                self.relative_task_epoch_idx[task] = self.idx

        def compatible_merge(self, new_frame):
            assert isinstance(new_frame, FrameInfo.Frame)

            self.relative_tasks = self.relative_tasks | new_frame.relative_tasks
            self.description = self.description + new_frame.description

            self.action = self.action + new_frame.action

            for new_obj in new_frame.objs:
                last_added_obj = self.fi.objset.last_added_obj
                new_added_obj = self.fi.objset.add(new_obj, self.idx, merge_idx=self.idx)
                if last_added_obj == new_added_obj:
                    print('already exists')

            for epoch, obj_list in self.fi.objset.dict.items():
                if epoch == self.idx:
                    self.objs = obj_list.copy()

            # update the dictionary for relative_task_epoch
            temp = self.relative_task_epoch_idx.copy()
            temp.update(new_frame.relative_task_epoch_idx)
            self.relative_task_epoch_idx = temp

        def __str__(self):
            return 'frame: ' + str(self.idx) + ', relative tasks: ' + \
                   ','.join([str(i) for i in self.relative_tasks]) \
                   + ' objects: ' + ','.join([str(o) for o in self.objs])
