from collections import deque

# import torch
import numpy as np

class Frstack():

    def __init__(self, initial_frame, frame_count=4):
        '''
        Initializes Frstack.

        Args:
            frame: the initial frame to stack
            frame_count: number of frames to stack
        '''
        self.stack = deque(maxlen=frame_count)
        self.frame_count = frame_count
        self.push(initial_frame, True)

    def get_stack(self):
        '''
        Returns stacked frames.

        Returns:
            torch tensor of dimensions (frame_count, frame_height, frame_width)
            (note this is because PyTorch CNN takes input as CHW vs TF's HWC)
        '''
        frames = [frame for frame in self.stack]
        # result = torch.stack(frames)
        result = np.stack(frames)
        return result

    def push(self, frame, new_episode):
        '''
        Pushes new frame onto the stack. If it is a new episode, the frame is
        stacked frame_count number of times

        Args:
            frame: the frame to stack
            new_episode: whether frame is a new_episode
        '''
        if new_episode:
            for i in range(self.frame_count):
                self.stack.append(frame)
        else:
            self.stack.append(frame)
