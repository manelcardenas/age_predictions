

def center_crop(img, new_shape=(160, 192, 160)):
    startx = img.shape[0]//2 - new_shape[0]//2
    starty = img.shape[1]//2 - new_shape[1]//2
    startz = img.shape[2]//2 - new_shape[2]//2
    return img[startx:startx+new_shape[0], starty:starty+new_shape[1], startz:startz+new_shape[2]]



