
import os
import time
import tkinter as tk
from PIL import Image, ImageTk
# create the root window
def slideshows():
    root = tk.Tk()
    # set ULC (x, y) position of root window
    root.geometry("+{}+{}".format(70, 100))
    root.title("a simple Tkinter slide show")

    # delay in seconds (time each slide shows)
    delay = 0.05

    # create a list of image file names
    # (you can add more files as needed)
    # pick image files you have in your working directory or use full path
    # PIL's ImageTk allows .gif  .jpg  .png  .bmp formats
    imageFiles = [ ]
    PATH = '/Users/Edward/PycharmProjects/New_Blocks/deep-blocks'
    dirlist = os.listdir(PATH)
    imageFiles = dirlist
    print(imageFiles)
    # create a list of image objects
    # PIL's ImageTk converts to an image object that Tkinter can handle
    photos =[]
    for f in imageFiles:
        # try:
        #     if e < 10:
        #         if int(str(f)[0]) == e and len(str(f)) == 2:
        #             image1 = Image.open("/Users/Edward/PycharmProjects/New_Blocks/deep-blocks/"+str(f))
        #             tkpi = ImageTk.PhotoImage(image1)
        #             photos.append(tkpi)
        #     else:
        #         if int(str(f)[0:2]) == e and len(str(f)) == 3:
        #             image1 = Image.open("/Users/Edward/PycharmProjects/New_Blocks/deep-blocks/"+str(f))
        #             tkpi = ImageTk.PhotoImage(image1)
        #             photos.append(tkpi)
        # except Exception:
        try:
            image1 = Image.open(PATH+str(f))
            tkpi = ImageTk.PhotoImage(image1)
            photos.append(tkpi)
        except Exception:
            pass

    # use a button to display the slides
    # this way a simple mouse click on the picture-button stops the show
    button = tk.Button(root, command=root.destroy)
    button.pack(padx=5, pady=5)

    for photo in photos:
        button["image"] = photo
        root.update()
        time.sleep(delay)

    # execute the event loop
    root.mainloop()
slideshows()