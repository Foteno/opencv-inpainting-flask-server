import cv2
import numpy as np
import time
from flask import Flask, request
from pathlib import Path
from env import mask_dir
from os.path import abspath

app = Flask(__name__)

out_dir = Path("./temporary")
out_dir.mkdir(parents=True, exist_ok=True)

@app.route('/choose-mask', methods=['GET'])
def choose_mask():

    mask_name = "dilated_" + request.args['mask_file']
    mask_path = mask_dir / mask_name
    image_path = mask_dir / "image.png"

    algorithm = request.args['algorithm']

    img = cv2.imread(abspath(image_path))
    mask = cv2.imread(abspath(mask_path), cv2.IMREAD_GRAYSCALE)
    print("begin")

    if algorithm == "telea":
        start_time = time.perf_counter()
        inpaint = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
        end_time = time.perf_counter()

        latency = (end_time - start_time) 
        print("TELEA latency: %.5f seconds" % latency)
    elif algorithm == "ns":
        start_time = time.perf_counter()
        inpaint = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
        end_time = time.perf_counter()

        latency = (end_time - start_time) 
        print("Navier-Stokes latency: %.5f seconds" % latency)

    img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_path).name}"
    cv2.imwrite(abspath(img_inpainted_p), inpaint)

    with open(abspath(img_inpainted_p), 'rb') as f:
        img_inpainted = f.read()

    return img_inpainted

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5002)