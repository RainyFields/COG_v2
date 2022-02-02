import numpy as np
from cognitive.task_generator import Task
from cognitive.stim_generator import ObjectSet
from cognitive.convert import TaskInfoConvert

class ComboTaskInfo(object):
    def __init__(self, task = Task(), example = {}, objset = ObjectSet(n_epoch = 0), frameinfo = TaskInfoConvert() ):
        """
        used for convinent storage of all task relevant information
        :param task: the task object
        :param example: a dictionary including all necessary information of the task
        :param objset: objset related to the task
        :param frameinfo: frameinfo related to the task
        """
        self.task = task
        self.example = example
        self.objset = objset
        self.frameinfo = frameinfo
