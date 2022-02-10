# from typing import List, Dict, Any
#
# import numpy as np
# import cognitive.task_generator as tg
# import cognitive.stim_generator as sg
# import cognitive.frame_info as fi
# import cognitive.helper as helper
# import cognitive.constants as CONST
#
#
# class TaskInfoConvert(object):
#     """
#     Storage of composition information,
#     including task_frame_info, task_example, task and objset
#     :param task_frame_info: FrameInfo
#     :param task_example: including 'family'(task_family), 'epochs'(epochs),'question','objects' and 'answer'
#     :param task: task family
#     :param objset: objset
#     """
#
#     def __init__(self, frame_info = None, task_example=None, task=None, objset=None):
#         " combining with a second task should be implemented incrementally"
#         if frame_info is None or objset is None:
#             raise ValueError("task information is incomplete")
#         else:
#             # assert isinstance(frame_info, fi.FrameInfo)
#             assert isinstance(objset, sg.ObjectSet)
#             self.frame_info = frame_info
#             self.objset = objset
#             self.task_example = task_example
#             self.task = task
#
#     def __len__(self):
#         # return number of tasks involved
#         return len(self.frame_info)
#
#     @property
#     def n_epochs(self):
#         return len(self.frame_info)
#
#     #### todo: do we still need it?
#     # def index_conv(self, frame_idx, task_idx):
#     #     # return epoch index for given frame index and task index
#     #     return self.frame_info.frame_list[frame_idx]["relative_task_epoch_idx"][task_idx][0]
#
#
#         # reuse visual stimuli with probability reuse
#
#         ########## delete below
#         # if np.random.random() < reuse:
#         #     # reuse past info, and resolve conflict
#         #     for i, curr_frame in enumerate(self.frame_info[start:]):
#         #         if new_task_info.frame_info[i].objs != curr_frame.objs: # subset
#         #             # identify which select operator for this frame: add self.track_op in the GoShape task
#         #             ### todo: move it to the Task Class as a general attribute?
#         #             ### todo: what is the task here? I need to get access to the operator so suppose we have the initialized task
#         #             # current select operator is new_task_info.task.track_op[i]
#         #             # update the associate select operator with curr_frame.objs information, if multiple, choose one
#         #             new_task_info.task.track_op[i].update(curr_frame.objs)
#         #     ####### after this loop, new task with updated info should be compatible with the previous one
#         #
#         #     ### todo: self.compatible_merge()
#         #     return self.compatible_merge(new_task_info, add_stim = False) # compatible merge means merge two task without adding new stims
#         #
#         # # create new frames and merge
#         # else:
#         #     # find the first consecutively shareable frame
#         #     # add more frames or change lastk? add more frames for now
#         #     if start == -1:
#         #         # queue
#         #         extra_f = len(new_task_info.frame_info)
#         #     else:
#         #         extra_f = new_task_info.n_epochs - self.n_epochs - start
#         #
#         #     for i in range(extra_f):
#         #         self.add_new_frame({next_task_idx}, new_task_info)
#         #
#         #     for old, new in zip(self.frame_info[start, len(self.frame_info)], new_task_info.frame_info):
#         #         old.merge(new)
#
#         self.task_info.append(new_task_info.task_info[0])
#         return
#
#     # def inv_convert(self):
#     #     # inverse the frameinfo to task examples
#     #     examples = []
#     #     for i in range(len(self)):
#     #         examples.append[{}]
#     #         examples[i]["family"] = self.task_info[i]["task_family"]
#     #         examples[i]["epochs"] = self.task_info[i]["task_len"]
#     #         examples[i]["question"] = self.task_info[i]["question"]
#     #         examples[i]["answers"] = self.task_info[i]["answers"]
#     #         examples[i]["is_intact"] = self.task_info[i]["is_intact"]
#     #
#     #         inv_frame_index = []  # frame index if involved in task i
#     #         objects_feat = []
#     #         objects = []
#     #         curr_obj = {}
#     #         for j, frame in enumerate(self.frame_info):
#     #             if i in frame["relative_tasks"]:
#     #                 count_i = frame["relative_tasks"].index(i)
#     #                 inv_frame_index.append(j)
#     #                 for obj in self.frame_info[j]["objs"]:
#     #                     for features in ["location", "shape", "color", "is_distractor"]:
#     #                         curr_obj[features] = obj[features]
#     #
#     #                     if curr_obj not in objects_feat:
#     #                         curr_obj["epochs"] = [self.frame_info[j]["relative_task_epoch_idx"][count_i]]
#     #                         objects_feat.append(curr_obj)
#     #                         objects.append(curr_obj)
#     #                     else:
#     #                         obj_idx = objects_feat.index(curr_obj)
#     #                         objects[obj_idx]["epochs"].append(self.frame_info[j]["relative_task_epoch_idx"][count_i])
#     #             examples[i]["objects"] = objects
#     #     return examples
#     #
#     # def inv_convert_objset(self):
#     #     '''
#     #
#     #     :return: list of objsets
#     #     '''
#     #     # convert frame_info to objset
#     #     pass
#     #
#     # def task_update(self):
#     #     pass
#     #     # todo: update task upon request
