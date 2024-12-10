import struct
import math
import json
import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path
from lxml import etree


dim = (750, 1024)


def hexlongbits2double(s):
    return struct.unpack('d', struct.pack('Q', int(s, 16)))[0] 


def bbox2rect(bbox, t):
    """Convert bbox to a description dictionary."""
    x1 = hexlongbits2double(bbox[0])
    y1 = hexlongbits2double(bbox[1])
    x2 = hexlongbits2double(bbox[2])
    y2 = hexlongbits2double(bbox[3])
    area = (x2 - x1) * (y1 - y2)  # NOTE: origin at this point is still at bottom left, hence y1-y2
    
    return {
        "area":area, 
        "rect": [x1, y1, x2, y2], 
        "type": t
    }


def get_page_bbox(xml):    
    page_bbox = [hexlongbits2double(s) for s in xml.xpath("//Page")[0].get("BBox").split(" ")]
    px0, py1, px1, py0 = page_bbox
    pw = abs(px1 - px0)
    ph = abs(py1 - py0)
    return px0, py1, px1, py0, pw, ph


def recalc_labels(handle, xml, image):
    px0, py1, px1, py0, pw, ph = get_page_bbox(xml)
    label_type = {
        "EmbeddedFormula": "E",
        "IsolatedFormula": "I"
    }

    labels = []
    for el in xml.xpath("//Page/*"):
        _type = label_type[el.tag]
        _box = bbox2rect(el.get("BBox").split(" "), _type)
        rect = _box["rect"]

        x0p, y1m, x1p, y0m = rect

        # https://stackoverflow.com/questions/52824584/do-anyone-know-how-to-extract-image-coordinate-from-marmot-dataset
        x0 = math.floor(image.shape[1]*(x0p - px0)/pw)
        x1 = math.ceil(image.shape[1]*(x1p - px0)/pw)
        y0 = math.ceil(image.shape[0]*(py1 - y0m)/ph)
        y1 = math.floor(image.shape[0]*(py1 - y1m)/ph)

        _box["rect"] = [x0, y1, x1, y0]  # 
        
        _box["identifier"] = handle
        labels.append(_box)
    
    return labels


def labels_to_json():
    """Convert labels to json format."""
    ground_labels = list(Path("../data/ground_truth").glob("*.xml"))

    for _file in tqdm(ground_labels):
        handle = _file.name.replace('.xml', '')

        image_path = f"../data/image/{handle}.tif"
        image = cv2.imread(image_path)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        output_image_path = f"../data/processed/image/{handle}.tif"
        cv2.imwrite(output_image_path, image)

        output = Path(f"../data/processed/labels/{handle}.json")
        labels = recalc_labels(handle, etree.parse(str(_file)), image)
        output.write_text(json.dumps(labels))

        verify_image = np.copy(image)
        for label in labels:
            rect = label["rect"]
            x0, y0, x1, y1 = rect
            cv2.rectangle(verify_image, (int(x0), int(y0)), (int(x1), int(y1)), (205, 0, 0), 1) 
        output_verify_image = f"../data/processed/verify/{handle}.tif"
        cv2.imwrite(output_verify_image, verify_image)


if __name__ == "__main__":
    labels_to_json()
