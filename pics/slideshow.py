import logging
import os
import platform
import sys
from collections import deque
from itertools import cycle
import numpy as np
import scipy
try:
    import tkinter as tk
except ImportError:  # Python 2
    import Tkinter as tk  # $ sudo apt-get install python-tk

from PIL import Image  # $ pip install pillow
from PIL import ImageTk

psutil = None  # avoid "redefinition of unused 'psutil'" warning
try:
    import psutil
except ImportError:
    pass

debug = logging.debug




class Slideshow(object):

    def __init__(self, parent, filenames, slideshow_delay=0.1, history_size=100):
        self.ma = parent.winfo_toplevel()
        self.filenames = cycle(filenames)  # loop forever
        self._files = deque(maxlen=history_size)  # for prev/next files
        self._photo_image = None  # must hold reference to PhotoImage
        self._id = None  # used to cancel pending show_image() callbacks
        self.imglbl = tk.Label(parent)  # it contains current image
        # label occupies all available space
        self.imglbl.pack(fill=tk.BOTH, expand=True)

        # start slideshow on the next tick
        self.imglbl.after(1, self._slideshow, slideshow_delay * 10)

    def _slideshow(self, delay_milliseconds):
        self._files.append(next(self.filenames))
        self.show_image()
        self.imglbl.after(delay_milliseconds, self._slideshow,
                          delay_milliseconds)

    def show_image(self):
        filename = self._files[-1]
        debug("load %r", filename)
        image = Image.open(filename)  # note: let OS manage file cache

        # shrink image inplace to fit in the application window
        w, h = self.ma.winfo_width(), self.ma.winfo_height()
        if image.size[0] > w or image.size[1] > h:
            # note: ImageOps.fit() copies image
            # preserve aspect ratio
            if w < 3 or h < 3:  # too small
                return  # do nothing
            image.thumbnail((w - 2, h - 2), Image.ANTIALIAS)
            debug("resized: win %s >= img %s", (w, h), image.size)

        # note: pasting into an RGBA image that is displayed might be slow
        # create new image instead
        self._photo_image = ImageTk.PhotoImage(image)
        self.imglbl.configure(image=self._photo_image)

        # set application window title
        self.ma.wm_title(filename)

    def _show_image_on_next_tick(self):
        # cancel previous callback schedule a new one
        if self._id is not None:
            self.imglbl.after_cancel(self._id)
        self._id = self.imglbl.after(1, self.show_image)

    def next_image(self, event_unused=None):
        self._files.rotate(-1)
        self._show_image_on_next_tick()

    def prev_image(self, event_unused=None):
        self._files.rotate()
        self._show_image_on_next_tick()

    def fit_image(self, event=None, _last=[None] * 2):
        """Fit image inside application window on resize."""
        if event is not None and event.widget is self.ma and (
                _last[0] != event.width or _last[1] != event.height):
            # size changed; update image
            _last[:] = event.width, event.height
            self._show_image_on_next_tick()


def get_image_files(rootdir):
    for path, dirs, files in os.walk(rootdir):
        dirs.sort()  # traverse directory in sorted order (by name)
        files.sort()  # show images in sorted order
        for filename in files:
            if filename.lower().endswith('.bmp'):
                yield os.path.join(path, filename)


logging.basicConfig(format="%(asctime)-15s %(message)s",
                    datefmt="%F %T",
                    level=logging.DEBUG)

root = tk.Tk()
if logging.getLogger().isEnabledFor(logging.DEBUG) and psutil is not None:

    def report_usage(prev_meminfo=None, p=psutil.Process(os.getpid())):
        # find max memory
        if p.is_running():
            meminfo = p.memory_info()
            if (meminfo != prev_meminfo and
                    (prev_meminfo is None or
                     meminfo.rss > prev_meminfo.rss)):
                prev_meminfo = meminfo
                debug(meminfo)
            root.after(500, report_usage, prev_meminfo)  # report in 0.5s

    report_usage()

# get image filenames
imagedir = sys.argv[1] if len(sys.argv) > 1 else '.'
image_filenames = get_image_files(imagedir)

# configure initial size
if platform.system() == "Windows":
    root.wm_state('zoomed')  # start maximized
else:
    width, height, xoffset, yoffset = 400, 300, 0, 0
    # double-click the title bar to maximize the app
    # or uncomment:

    # # remove title bar
    # root.overrideredirect(True) # <- this makes it hard to kill
    # width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d%+d%+d" % (width, height, xoffset, yoffset))

try:  # start slideshow
    app = Slideshow(root, image_filenames, slideshow_delay=1)
except StopIteration:
    sys.exit("no image files found in %r" % (imagedir, ))

# configure keybindings
root.bind("<Escape>", lambda _: root.destroy())  # exit on Esc
root.bind('<Prior>', app.prev_image)
root.bind('<Up>', app.prev_image)
root.bind('<Left>', app.prev_image)
root.bind('<Next>', app.next_image)
root.bind('<Down>', app.next_image)
root.bind('<Right>', app.next_image)

root.bind("<Configure>", app.fit_image)  # fit image on resize
root.focus_set()
root.mainloop()


