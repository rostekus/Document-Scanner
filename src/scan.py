import argparse
from tkinter import E
from qrbar import QrBar
from language import LanguageDetect
from directory import Directory
from textrecognition import TextRecognition
from scanner import Scanner, get_contour
from interacter import interactive_get_contour
import os
import cv2
import json


def single_file_handler(path,interactive = False, text = False,
 qr = False, lang = False, contour= None):
    (dirname, filename) = os.path.split(path)
    s = Scanner()
    s.open(path)
    _, new_img = s.crop(box =contour)
    text_img = None
    if text:
        text_img= TextRecognition.recognition(new_img)
        file = os.path.join(dirname,f'{filename}.txt')
        print(file)
        with open(file, 'w') as f:
            f.write(text_img)
    if lang:
        if not text_img:
            text_img = TextRecognition.recognition(new_img) 
        detect = LanguageDetect()
        lang_img = detect.predict(text_img)
        filename = f"{lang_img}_{filename}"
    if qr:
        try:
            qr_code = QrBar.read(cv2.imread(path))
        except:
            print("No qr code")
        else:
            filename = f"'{qr_code}'_{filename}"
    filename = os.path.join(dirname,f'SCANEED__{filename}')
    print(filename)
    cv2.imwrite(filename, s.save(new_img))

def directory_handler(path,text = False,
 qr = False, lang = False):
    output = os.path.join(path, 'output')
    if not os.path.exists(output):
        os.mkdir(output)

    os.makedirs()
    files = Directory.read_directory(path)
    s = Scanner()
    json_file = {}
    for file in files:
        json_file[file] = {}
        filename = os.path.join(path, file)
        s.open(path)
        _, new_img = s.crop()
        text_img = None
        if not text:
            text_img= TextRecognition.recognition(new_img)
            json_file[file]['text'] = text_img
        if not lang:
            if not text_img:
                text_img = TextRecognition.recognition(new_img) 
            detect = LanguageDetect()
            lang_img = detect.predict(text_img)
            json_file[file]['lang'] = lang_img
        if not qr:
            try:
                qr_code = QrBar.read(cv2.imread(path))
            except:
                print('cant find')
            else:
                json_file[file]['qr'] = qr_code
        filename = os.path.join(output, f'SCANNED__{file}')
        cv2.imwrite(filename, s.save(new_img))
        with open('data.json', 'w') as fp:
            json.dump(json_file)



def main():
    ap = argparse.ArgumentParser()
    # group = ap.add_mutually_exclusive_group(required=True)
    # group.add_argument("--images", help="Directory of images to be scanned")
    # group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true',
        help = "Check if Language")
    required = ap.add_argument_group('required named arguments')
    ap.add_argument("-t", action='store_true',
        help = "Check if Language")
    ap.add_argument("-l", action='store_true',
        help = "Check if Language")
    ap.add_argument("-q", action='store_true',
        help = "Check if Language")
    
    args = vars(ap.parse_args())
    # img_dir = args["images"]
    # img_path = args["image"]
    text = args['t']
    lang =  args["l"]
    qr =  args["q"]
    img_dir =False
    img_path = '/Users/rostyslavmosorov/Desktop/projekty/receipt-scanner/src/images/samp.jpg'
    interactive =  args["i"]
    interactive = True
    if interactive and img_dir:
        print('Interective mode only possible with one image')
    if img_dir:
        if not os.path.isdir(img_dir):
            print(f'{img_dir} is not a directory')
        else:
            directory_handler(img_dir,text, qr,lang)

    if interactive:
        s = Scanner()
        try:
            img = cv2.imread(img_path)
        except:
            print('Cant open file {img_path}')
        contour =get_contour(img)
        flags, contour = interactive_get_contour(contour,img)
        print(flags)
        text =flags['text']
        qr =flags['qr'] 
        lang=flags['lang']

        
        single_file_handler(img_path,text = text, qr = qr,lang = lang, contour =contour)
    else:
        single_file_handler(img_path,text = text, qr = qr,lang = lang, contour =contour)


if __name__ == '__main__':
    main()