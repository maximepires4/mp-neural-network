from tkinter import *
from tkinter.colorchooser import askcolor

from src.gui.imagehandler import ImageHandler
from src.gui.neuralnethandler import NeuralNetHandler

class Paint(object):

    DEFAULT_PEN_SIZE = 50.0
    DEFAULT_COLOR = 'white'
    DEFAULT_BACKGROUND = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.title('Handwriting recognition')

        self.pen_button = Button(self.root, text='Pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)

        self.erase_all_button = Button(self.root, text='Erase all', command=self.erase_all)
        self.erase_all_button.grid(row=0, column=2)

        self.export_image_button = Button(self.root, text='Export image', command=self.export_image)
        self.export_image_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=self.DEFAULT_PEN_SIZE-10, to=self.DEFAULT_PEN_SIZE+10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)

        #self.c = Canvas(self.root, bg=self.DEFAULT_BACKGROUND, width=600, height=600)
        self.c = Canvas(self.root, bg=self.DEFAULT_BACKGROUND, width=28, height=28)
        self.c.grid(row=1, columnspan=5)
        self.c.scale("all", 0, 0, 10, 10)

        self.textvar = StringVar()
        self.guess = Label(self.root, textvariable=self.textvar, font=("TkDefaultFont", 44))
        self.guess.grid(row=2, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.image_handler = ImageHandler(self.root, self.c)
        self.neuralnet_handler = NeuralNetHandler()

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def export_image(self):
        self.image_handler.save()
    
    def erase_all(self):
        self.c.delete('all')

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.DEFAULT_BACKGROUND if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
        
        self.image_handler.update()
        prediction = self.neuralnet_handler.predict(self.image_handler.image)
        self.textvar.set("Guess: " + str(prediction))

    def reset(self, event):
        self.old_x, self.old_y = None, None
