from pathlib import Path
import pytesseract
from pytesseract import Output, run_and_get_output
from subprocess import check_output

TESSERACT_EXE = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2


class ocr_box_result:

    def __init__(self, img, row):
        self.img = img
        self.hImg, self.wImg, _ = self.img.shape

        b = row.split()
        self.detected_char, self.page = b[0], int(b[5])
        self.left, self.top, self.right, self.bottom = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        self.calc_img_bottom = self.hImg - self.bottom
        self.calc_img_top = self.hImg - self.top

    def extract_box_from_image(self, enlarge_factor=1.0):
        width = (self.right - self.left)
        height = (self.top - self.bottom)
        center_x = (self.left + self.right) / 2
        center_y = (self.top + self.bottom) / 2
        new_width = width * enlarge_factor
        new_height = height * enlarge_factor

        # Hope it's okay, didn't test edge cases yet
        new_left = max(int(center_x - new_width / 2), 0)
        new_right = min(int(center_x + new_width / 2), self.img.shape[1])
        new_top = min(int(center_y + new_height / 2), self.img.shape[0])
        new_bottom = max(int(center_y - new_height / 2), 0)

        char_box_original = self.img[self.bottom:self.top, self.left:self.right].copy()
        char_box_enlarged = self.img[self.new_bottom:self.new_top, self.new_left:self.new_right].copy()
        return char_box_original, char_box_enlarged

def new_char_filename(original_filename, letters_location, description_row):
    path = Path(original_filename)
    b = description_row.split()
    hex_str = b[0].encode("utf-8").hex()
    b[0] = hex_str + 'h_'
    new_filename = '_'.join(b)
    new_filename_full = path.with_stem(new_filename + '__' + path.stem).with_suffix('.png')
    new_filename_full = Path(letters_location).joinpath(new_filename_full.name)
    return new_filename_full.as_posix()


##########
# Source: https://stackoverflow.com/questions/54246492/pytesseract-difference-between-image-to-string-and-image-to-boxes
# Modification
def image_to_boxes_keep_same(
        image,
        lang=None,
        config='',
        nice=0,
        output_type=pytesseract.Output.STRING,
        timeout=0,
):
    """
    Returns string containing recognized characters and their box boundaries
    """
    config = f'{config.strip()} makebox'
    args = [image, 'box', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def perform_ocr_commandline(jpgfile, txt_filename, tesseract_exe=TESSERACT_EXE):
    cmd = f'"{tesseract_exe}" -l heb "{jpgfile}" "{txt_filename}"'
    print(f'Running\n{cmd}\n\n')
    output = check_output(cmd, shell=True).decode()
    print(output)
    print('----------')


def hconcat_resize_max(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


