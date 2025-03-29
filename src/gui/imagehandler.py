from PIL import ImageGrab
from pathlib import Path

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
        self.image = ImageGrab.grab().crop((x,y,x1,y1)).convert('L').resize((28, 28))
    
    def save(self):
        Path("output").mkdir(parents=True, exist_ok=True)
        self.image.save("output/image.jpg")