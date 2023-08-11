import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from PIL import Image
from matplotlib.path import Path
import os

class SelectiveWhitener:
    def __init__(self, image_path):
        self.image_path = image_path
        self.polygons = []
        self.image_name = os.path.basename(image_path)

    def onselect(self, verts):
        self.polygons.append(verts)
        print('Polygon vertices:', verts)

    def show_image(self):
        Image.MAX_IMAGE_PIXELS = 1000000000
        self.image = Image.open(self.image_path)
        self.image_array = np.array(self.image)

        fig, ax = plt.subplots()
        ax.imshow(self.image_array)  # 'img' should be a NumPy array containing your image

        self.ps = PolygonSelector(ax, self.onselect)

        plt.show()

    def whiten_polygons(self):
        for polygon in self.polygons:
            path = Path(polygon)
            for i in range(self.image_array.shape[1]):  # x coordinate
                for j in range(self.image_array.shape[0]):  # y coordinate
                    if path.contains_point((i, j)):
                        self.image_array[j, i] = [255, 255, 255]  # make the pixel white

        whitened_image = Image.fromarray(self.image_array)
        whitened_image.save('Palm/small_images_new/' + self.image_name)  # Save in specified directory

whitener = SelectiveWhitener('Palm/small_images_new/small_image_19.jpg') # small_images_new small_images
whitener.show_image()  # After closing the image window, the polygons are saved
whitener.whiten_polygons()  # Apply whitening
