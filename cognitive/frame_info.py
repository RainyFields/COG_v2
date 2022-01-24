import stim_generator as sg


class FrameInfo(object):
    def __init__(self, n_epochs, task_question, task_answers, relative_tasks=None, objset=None):
        self.frame_list = []
        self.n_epochs = n_epochs
        for i in range(n_epochs):
            description = []
            if i == self.n_epochs - 1:
                description.append(["ending of task %d" % 0])
                self.frame_list.append(self.Frame(i, description, relative_tasks, task_answers))
            else:
                action = None
                if i == 0:
                    description.append(["start of task %d" % 0])
                    self.frame_list.append(self.Frame(i, description, relative_tasks, task_question))

        if objset:
            for obj in objset:
                if obj.epoch[0] + 1 == self.epoch[1]:
                    self.frame_list[obj.epoch[0]].objs.append(obj)
                else:
                    for epoch in range(obj.epoch[0], obj.epoch[1]):
                        self.frame_list[epoch].objs.append(obj)

    class Frame(object):
        def __init__(self, idx, description, action, relative_tasks):
            self.idx = idx
            self.relative_tasks = relative_tasks
            self.description = description
            self.action = action
            self.objs = []

            self.relative_task_epoch_idx = {}
            for task in self.relative_tasks:
                self.relative_task_epoch_idx[task] = self.idx
