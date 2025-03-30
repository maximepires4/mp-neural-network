from PIL import ImageGrab, Image
from pathlib import Path
import numpy as np

import scipy.ndimage as ndi

class ImageHandler():

    def __init__(self, root, canvas):
        self.root = root
        self.canvas = canvas
        self.image = None
        self.update()
    
    def update(self):
        x=self.root.winfo_rootx()+self.canvas.winfo_x()
        y=self.root.winfo_rooty()+self.canvas.winfo_y()
        x1=x+self.canvas.winfo_width()
        y1=y+self.canvas.winfo_height()
        
        # Converting canvas to image and getting rid and setting mode to grayscale
        base_img = ImageGrab.grab().crop((x,y,x1,y1)).convert('L')

        # Converting image to numpy array to crop it
        base_img_data = np.asarray(base_img)
        non_empty_columns = np.where(base_img_data.max(axis=0)>0)[0]
        non_empty_rows = np.where(base_img_data.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows, default=0), max(non_empty_rows, default=0), min(non_empty_columns, default=0), max(non_empty_columns, default=0))
        base_img_data_cropped = base_img_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
        cropped_img = Image.fromarray(base_img_data_cropped)

        if cropped_img.size[0] <= 1 and cropped_img.size[1] <= 1:
            self.image = Image.new('L', (28, 28))
            return

        # Resizing image to (20, 20) while keeping aspect ratio
        percent = min(20 / float(cropped_img.size[0]), 20 / float(cropped_img.size[1]))
        wsize = int((float(cropped_img.size[0]) * float(percent)))
        hsize = int((float(cropped_img.size[1]) * float(percent)))
        resized_img = cropped_img.resize((wsize, hsize), Image.Resampling.LANCZOS)

        self.new_image = resized_img

        # Finding center of mass of image
        cy, cx = ndi.center_of_mass(resized_img)

        # Creating a (28, 28) image and pasting the old one at the center of the new one
        final_img = Image.new('L', (28, 28))
        Image.Image.paste(final_img, resized_img, (int(final_img.size[0]/2 - round(cx)), int(final_img.size[1]/2 - round(cy))))

        self.image = final_img
    
    def save(self):
        Path("output").mkdir(parents=True, exist_ok=True)
        self.image.save("output/image.jpg", quality=100)