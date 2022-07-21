import logging
import os
import sys

from queue import Queue
from threading import Thread 

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class PlotterThread():
    '''log tensorboard data in a background thread to save time'''
    def __init__(self, writer):
        self.writer = writer
        self.task_queue = Queue(maxsize=0)
        worker = Thread(target=self.do_work, args=(self.task_queue,))
        worker.setDaemon(True)
        worker.start()

    def do_work(self, q):
        while True:
            content = q.get()
            if content[-1] == 'image':
                self.writer.add_image(*content[:-1])
            elif content[-1] == 'scalar':
                self.writer.add_scalar(*content[:-1])
            else:
                raise ValueError
            q.task_done()

    def add_data(self, name, value, step, data_type='scalar'):
        self.task_queue.put([name, value, step, data_type])

    def __len__(self):
        return self.task_queue.qsize()