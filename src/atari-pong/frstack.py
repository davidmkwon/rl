from collections import deque

import torch

class Frstack():

    def __init__(self, frame, frame_count=4):
        self.stack = deque(maxlen=frame_count)
        self.frame_count = frame_count
        self.push(frame, True)

    def get_stack(self):
        frames = [frame for frame in self.stack]
        result = torch.stack(frames, dim=2)
        return result

    def push(self, frame, new_episode):
        if new_episode:
            for i in range(self.frame_count):
                self.stack.append(frame)
        else:
            self.stack.append(frame)
