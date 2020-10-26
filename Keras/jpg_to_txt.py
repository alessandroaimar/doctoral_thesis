import tensorflow as tf
import matplotlib.pyplot as plt

image_filename = r"D:\DL\datasets\ILSVRC2012\train\\n01440764_11335.JPEG"
txt_filename =  r"D:\DL\datasets\ILSVRC2012\txt\\n01440764_11335.txt"

image = plt.imread(image_filename)
image = tf.image.resize(image, (224,224)).numpy()

with open(txt_filename, "w", newline ="") as f_ptr:


    for row_idx in range(224):
        for col_idx in range(224):
            for ch_idx in range(3):
                pixel = str(int(image[row_idx,col_idx, ch_idx]))
                f_ptr.write(pixel)
                f_ptr.write("\n")


    f_ptr.close()




