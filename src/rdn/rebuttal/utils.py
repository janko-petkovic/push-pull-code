from time import time

class Timer:
    def __init__(self, function):
        self.start = time()
        self.function = function

    def __enter__(self):
        return self.function

    def __exit__(self, type, value, traceback):
        self.stop = time()
        elapsed_time = self.stop - self.start
        print(f'Elapsed time: {elapsed_time:.2f}')
