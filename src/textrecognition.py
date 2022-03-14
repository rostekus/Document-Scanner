from easyocr import Reader
import cv2
import os
import numpy as np
GPU = -1

class TextRecognition:
    @classmethod
    def cleanup_text(self,text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()
    @classmethod
    def recognition(cls, image):
        img= image.copy()
        
        _,image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)
        reader = Reader(['en' , 'pl'], gpu=GPU)
        results = reader.readtext(image)
        textfile =""
        # loop over the results
        
        for i, (bbox, text, prob) in enumerate(results):
            
            # display the OCR'd text and associated probability
            print("[INFO] {:.4f}: {}".format(prob, text))
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # cleanup the text and draw the box surrounding the text along
            # with the OCR'd text itself
            text = cls.cleanup_text(text)
            if i% 10==0:
                textfile += "\n"
            textfile += f'{text} ' 
        # show the output image
        return textfile

