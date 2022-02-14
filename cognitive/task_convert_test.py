import numpy as np
import cognitive.task_bank as task_bank
from cognitive.info_generator import FrameInfo
from cognitive.helper import get_target_value
from cognitive.info_generator import TaskInfoCompo

def main():
    task1 = task_bank.GoShapeTemporal(4)
    objset1 = task1.generate_objset(task1.n_frames)
    print(str(task1), str(objset1))
    fi1 = FrameInfo(task=task1, objset=objset1)
    targets1 = task1.get_target(objset1)
    task_example1 = {
        'family': task1,
        'epochs': task1.n_frames,  # saving an epoch explicitly is needed because
        # there might be no objects in the last epoch.
        'question': str(task1),
        'objects': [o.dump() for o in objset1],
        'answers': [get_target_value(t) for t in targets1],
        'first_shareable': task1.first_shareable
    }
    tic1 = TaskInfoCompo(frame_info=fi1, task_example = task_example1, task = task1, objset=objset1)

    task2 = task_bank.GoShapeTemporal(4)
    objset2 = task2.generate_objset(task2.n_frames)
    print(str(task2), str(objset2))
    fi2 = FrameInfo(task=task2, objset=objset2)

    targets2 = task2.get_target(objset2)
    task_example2 = {
        'family': task2,
        'epochs': task2.n_frames,  # saving an epoch explicitly is needed because
        # there might be no objects in the last epoch.
        'question': str(task2),
        'objects': [o.dump() for o in objset2],
        'answers': [get_target_value(t) for t in targets2],
        'first_shareable': task2.first_shareable
    }
    tic2 = TaskInfoCombo(frame_info=fi2, task_example=task_example2, task=task2, objset=objset2)

    for frame in fi2:
        frame.relative_tasks = {1}
    #
    print('first_shareables: ', fi1.first_shareable, fi2.first_shareable)
    fi1.first_shareable = 0

    # test merge function
    compo_task = merge(tic1, tic2, reuse = 1)


    # ti = cv.TaskInfoConvert(task=task2, objset=objset2)
    # start = fi1.get_start_frame(ti,{1})
    # for i, (old, new) in enumerate(zip(fi1[start:], fi2)):
    #     old.compatible_merge(new)
    # for frame in fi1:
    #     print(frame)
    # print(fi1.objset.dict)
    # print(fi1.objset)

if __name__ == '__main__':
    main()
