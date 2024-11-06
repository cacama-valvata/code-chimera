# reference to https://stackoverflow.com/questions/11731136/class-method-decorator-with-self-arguments

import logging
import time
import sys
from functools import wraps


def retry(times, func_name):
    def wrapper_fn(f):
        @wraps(f)
        def new_wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if i < 5:
                        time.sleep(0.1)
                    else:
                        time.sleep(1)
                    error = e
            # here ignore exception
            logging.error('function {} retried {} times, all failed: {}'.format(f.__name__, times, error))
            final_func = getattr(args[0], func_name)
            final_func()

        return new_wrapper

    return wrapper_fn


class A(object):
    def __init__(self):
        self.tt = 0

    @retry(3, 'as_decorator_parameter')
    def aa(self):
        self.tt += 1
        if self.tt < 5:
            logging.info(1)
            raise RuntimeError('get value failed')
        logging.info('success')

    def as_decorator_parameter(self):
        logging.info(self.tt)
        logging.info('in decorator parameter')


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    abj = A()
    abj.aa()

# output
# 2021-10-19 18:31:57,160 - 1
# 2021-10-19 18:31:57,272 - 1
# 2021-10-19 18:31:57,380 - 1
# 2021-10-19 18:31:57,488 - function aa retried 3 times, all failed: get value failed
# 2021-10-19 18:31:57,488 - 3
# 2021-10-19 18:31:57,488 - in decorator parameter
