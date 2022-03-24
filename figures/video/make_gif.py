import imageio
from PIL import Image
import numpy as np
import glob
import natsort

input_fname = '/home/haitian/CM-GAN-Inpainting/figures/video/frames/*.png'
output_fname = '/home/haitian/CM-GAN-Inpainting/figures/video/output.gif'
frames = []
filenames = glob.glob(input_fname)
filenames = natsort.natsorted(filenames)

images = []
for filename in filenames[0:]:
    image = Image.open(filename)
    image = np.array(image)
    image = image[180:870,0:960,:]
    images.append(image)
imageio.mimsave(output_fname, images)

def make_gif(output_file, folder=input_folder):
    imgs = []
    for i in range(52):
        img = Image.open('{}{}_iter_{}.png'.format(folder, img_list[img_id], i))
        img = np.array(img)
        imgs.append(img)
    imageio.mimsave(output_file, imgs)

def img_parse(input_file, output_file):
    img = Image.open(input_file)
    img.save(output_file)

for img_id in range(14):
    output_name = '/home/yaping/projects/NeRD/PaperID_411_supplementary_materials/gif/{}'.format(img_list[img_id])
    output_file = output_name + '.gif'
    imgs = []
    for i in range(52):
        img = Image.open('{}{}_iter_{}.png'.format(input_folder, img_list[img_id], i))
        h,w = img.size
        img = img.resize((int(h/1.5),int(w/1.5)))
        img = np.array(img)
        imgs.append(img)
    imageio.mimsave(output_file, imgs)

# if __name__ == '__main__':
#     make_gif()
    # img_parse(input_folder + 'deconstructed.jpg', './' + output_name + '_decon.jpg')
    # img_parse(input_folder + '/output/output_100.jpg', './' + output_name + '_final.jpg')
    # img_parse(input_folder + 'bunny_512.jpg', './' + output_name + '_truth.jpg')


