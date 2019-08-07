import re
import os
import errno


def multi_gsub(subs, string):
    for k, v in subs.items():
        string = re.sub(k, v, string)
    return string


class ChangeDir:
    def __init__(self, dir_name):
        if isinstance(dir_name, str):
            self.dir = dir_name
        else:
            raise ValueError('ChangeDir only accepts string dir_name')
        self.cwd = os.getcwd()

    def __enter__(self):
        try:
            os.mkdir(self.dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise ValueError('Unexpected error from ChangeDir')
        os.chdir(self.dir)
        return self

    def __exit__(self, *args):
        os.chdir(self.cwd)
