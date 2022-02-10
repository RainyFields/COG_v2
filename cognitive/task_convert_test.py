import numpy as np
import cognitive.task_bank as task_bank
from cognitive.frame_info import FrameInfo
from cognitive.convert import TaskInfoCombo

def main():
    task1 = task_bank.GoShapeTemporal(4)
    objset1 = task1.generate_objset(task1.n_frames)
    print(task1, objset1)
    fi1 = FrameInfo(task=task1, objset=objset1)

    task2 = task_bank.GoShapeTemporal(4)
    objset2 = task1.generate_objset(task1.n_frames)
    print(task2, objset2)
    fi2 = FrameInfo(task=task2, objset=objset2)
    for frame in fi2:
        frame.relative_tasks = {1}

    print('first_shareables: ', fi1.first_shareable, fi2.first_shareable)
    # fi1.first_shareable = 0
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
