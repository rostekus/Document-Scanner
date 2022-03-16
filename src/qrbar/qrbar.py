from pyzbar.pyzbar import decode
import sys
import os


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class QrBar:
    @classmethod
    def read(self, img):
        try:
            if decode(img):
                for barcode in decode(img):
                    data = barcode.data.decode("utf-8")
                    return data
        except AssertionError:
            print(AssertionError)
            return None
