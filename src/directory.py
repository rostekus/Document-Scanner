from os import listdir
from os.path import isfile, join

class Directory:
    @classmethod
    def read_directory(self,mypath:str, recursivly= False):
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        return onlyfiles