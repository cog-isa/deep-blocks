import numpy as np
import scipy

def draw(state,time,e):
    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = np.zeros((900, 900, 3), dtype=np.uint8)
    #img = scipy.misc.imsave("pic.bmp", data)  # Create a PIL image
    #img.show()
    state = state[0][:900]
    for i in range(900):
        if state[i] == 1:
            m = i//30
            n = i%30
            data[30*m:30*m+30, 30*n:30*n+30] = [254, 0, 0]  # Makes the middle pixel red
    #data[512, 513] = [0, 0, 255]  # Makes the next pixel blue
    scipy.misc.imsave("pics/"+str(e)+str(time)+"pic.bmp", data)  # Create a PIL image