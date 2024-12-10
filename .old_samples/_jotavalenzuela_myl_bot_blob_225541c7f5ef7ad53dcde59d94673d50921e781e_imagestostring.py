#https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
from PIL import Image
import PIL.Image

from pytesseract import image_to_string
import pytesseract

output = pytesseract.image_to_string(PIL.Image.open('001.png'))
print(output)
