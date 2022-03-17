from qrbar.qrbar import QrBar
from language.language import LanguageDetect
from directory import Directory
from textrecognition import TextRecognition
from scanner.scanner import Scanner, get_contour
from interacter.interacter import interactive_get_contour

import argparse
import os
import cv2
import json


def single_file_handler(
    path, text=False, qr=False, lang=False, contour=None, dir=None
):
    json_file = {}
    print(path)
    (dirname, filename) = os.path.split(path)
    if dir:
        dirname = dir
    s = Scanner()
    s.open(path)
    try:
        new_img = s.crop(contour)
        if new_img is None:
            raise Exception
        if text:
            text_img = TextRecognition.recognition(new_img)
            file = os.path.join(dirname, f"{filename}.txt")

            with open(file, "w") as f:
                f.write(text_img)
        if lang:
            if not text_img:
                text_img = TextRecognition.recognition(new_img)
            detect = LanguageDetect()
            lang_img = detect.predict(text_img)
            json_file["lang"] = lang_img
            filename = f"{lang_img}_{filename}"
        if qr:
            try:
                qr_code = QrBar.read(cv2.imread(path))
            except:
                print("No qr code")
            else:
                filename = f"'{qr_code}'_{filename}"
                json_file["qr"] = qr_code
        filename = os.path.join(dirname, f"SCANEED__{filename}")
        cv2.imwrite(filename, new_img)
        return json_file
    except:
        print(f"Error for file {path}")
        return None


def directory_handler(path, text=False, qr=False, lang=False):
    output = os.path.join(path, "output")
    if not os.path.exists(output):
        os.mkdir(output)

    files = Directory.read_directory(path)
    json_file = {}
    for file in files:
        try:
            file = os.path.join(path, file)
            img = cv2.imread(file)
            contour = get_contour(img)
            json_file[file] = single_file_handler(
                file, text=text, qr=qr, lang=lang, contour=contour, dir=output
            )
        except:
            print(f"Error for file {file}")
    if any(text, qr, lang):
        with open("data.json", "w") as fp:
            json.dump(json_file, fp)


def main():

    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action="store_true", help="Interactive mode")
    ap.add_argument("-t", action="store_true", help="Text recognition")
    ap.add_argument("-l", action="store_true", help="language classification")
    ap.add_argument("-q", action="store_true", help="Check if image contains QR code")

    args = vars(ap.parse_args())
    img_dir = args["images"]
    img_path = args["image"]
    text = args["t"]
    lang = args["l"]
    qr = args["q"]
    interactive = args["i"]

    if interactive and img_dir:
        print("Interective mode only possible with one image")
    if img_dir:
        if not os.path.isdir(img_dir):
            print(f"{img_dir} is not a directory")
        else:
            directory_handler(img_dir, text, qr, lang)

    try:
        img = cv2.imread(img_path)
        contour = get_contour(img)

    except:
        print("Cant open file {img_path}")
        return
    if interactive:
        flags, contour = interactive_get_contour(contour, img)
        print(flags)
        text = flags["text"]
        qr = flags["qr"]
        lang = flags["lang"]

        single_file_handler(
            img_path, text=text, qr=qr, lang=lang, contour=contour
        )
    else:

        single_file_handler(
            img_path, text=text, qr=qr, lang=lang, contour=contour
        )


if __name__ == "__main__":
    main()
