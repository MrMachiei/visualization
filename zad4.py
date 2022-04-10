import skimage.io as io
from pylab import *
from skimage import feature, morphology, filters, util
from matplotlib import pyplot as plt
from matplotlib import gridspec as grd

def read(data):
    return io.imread(data, as_gray=True)

if __name__ == '__main__':
    pliki = ["samoloty/samolot02.jpg", "samoloty/samolot05.jpg", "samoloty/samolot12.jpg",
          "samoloty/samolot03.jpg", "samoloty/samolot11.jpg", "samoloty/samolot10.jpg"]
    obrazy = [read(i) for i in pliki]

    fig = plt.figure(figsize=(10, 10))

    grid = grd.GridSpec(3, 2)
    grid.update(wspace=0.01, hspace=0.02)

    for i, v  in enumerate(obrazy):
        ox = subplot(grid[i])
        ox.axis('off')
        image = v
        image = morphology.erosion(image, morphology.square(4))
        image = morphology.dilation(image, morphology.square(3))
        image = filters.rank.median(util.img_as_ubyte(image), ones([8, 8], dtype=uint8))
        image = feature.canny(image=image, sigma=4)
        ox.imshow(image, cmap='gray')
    fig.savefig('samoloty.pdf')
