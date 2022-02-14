import cognitive.constants as CONST
import numpy as np

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

def merge(compo_task, new_task_info, reuse=None):
    '''

    :param new_task_info: TaskInfoCombo object
    :return: None if no change, and the new task if merge needed change
    '''
    # TODO(mbai): change task instruction here
    # TODO: location conflict, feature (shape, color) conflict

    assert isinstance(new_task_info, fi.TaskInfoCombo)
    if len(new_task_info.task_info) > 1:
        raise NotImplementedError('Currently cannot support adding new composite tasks')

    if reuse is None:
        reuse = 0.5

    # correct task index in new_task_info
    next_task_idx = len(compo_task.task_info)
    for frame in new_task_info.frame_info:
        frame.relative_tasks = [next_task_idx + i for i, task in enumerate(frame.relative_tasks)]
        for i, task in enumerate(frame.relative_tasks):
            frame.relative_task_epoch_idx[next_task_idx + i] = frame.relative_task_epoch_idx[task].pop()
            task = next_task_idx + i

    start = compo_task.frame_info.get_start_frame()

    if start == -1:
        # queue
        extra_f = len(new_task_info.frame_info)
    else:
        extra_f = new_task_info.n_epochs - compo_task.n_epochs - start

    compo_task.frame_info.add_new_frames(extra_f, {next_task_idx}, new_task_info)

    curr_abs_idx = start
    for i, old_frame, new_frame in enumerate(zip(compo_task.frame_info[start:], new_task_info.frame_info)):
        assert isinstance(old_frame, fi.FrameInfo.Frame)
        assert isinstance(new_frame, fi.FrameInfo.Frame)
        # if there are no objects in the frame, then freely merge
        if not old_frame.objs or not new_frame.objs:
            old_frame.compatible_merge(new_frame)  # always update frame descriptions
        else:
            if np.random.random() < reuse:  # use curr frame info from previous task and reinit new task
                attr_expected = dict()
                # random select one obj from old frame objs
                old_obj = np.random.choice(old_frame.objs)
                attr_expected["loc"] = old_obj.loc
                attr_expected["when"] = old_obj.when
                attr_expected["shape"] = old_obj.shape
                attr_expected["color"] = old_obj.color

                new_task_info.task.reinit(i, attr_expected)
                new_task_info.objset = new_task_info.task.generate_objset(n_epoch=new_task_info.task.n_frames,
                                                                          average_memory_span=CONST.AVG_MEM)
                new_task_info.example["question"] = str(new_task_info.task)
                new_task_info.example["objects"] = [o.dump() for o in new_task_info.objset] ## todo: has this been updated?
                targets = new_task_info.task.get_target(compo_task.frame_info.objset)
                new_task_info.example["answers"] = [get_target_value(t) for t in targets]
                old_frame.compatible_merge(new_task_info.frame_info[i])
            else:
                old_frame.compatible_merge(new_frame)

        curr_abs_idx += 1
    return compo_task