from PIL import Image
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def do_nothing(path):
    return path


class ShuffleTiles(object):

    def __init__(self, size=(2, 2)):
        self.size = size

    def __call__(self, pic):
        size = self.size
        img = np.asarray(pic)
        tile_size = (int(img.shape[0] / size[0]), int(img.shape[1] / size[1]))
        tiled_imgs = []
        for i in range(size[0]):
            for ii in range(size[1]):
                from_y = i * tile_size[0]
                to_y = (i + 1) * tile_size[0]
                from_x = ii * tile_size[1]
                to_x = (ii + 1) * tile_size[1]
                tiled_imgs.append(img[from_y: to_y, from_x: to_x, :])
        shuffled_img = np.zeros((tile_size[0] * size[0], tile_size[1] * size[1], 3), dtype=np.uint8)
        tile_index = np.random.permutation(size[0] * size[1])
        count = 0
        for i in range(size[0]):
            for ii in range(size[1]):
                from_y = i * tile_size[0]
                to_y = (i + 1) * tile_size[0]
                from_x = ii * tile_size[1]
                to_x = (ii + 1) * tile_size[1]
                shuffled_img[from_y: to_y, from_x: to_x, :] = tiled_imgs[tile_index[count]]
                count += 1
        return shuffled_img


class InterpolateToGrey(object):

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, pic):
        alpha = self.alpha
        img = np.asarray(pic)
        grey = (1.0 - alpha) * np.mean(img, -1)
        interpolated_img = np.empty(img.shape)
        for i in range(len('rgb')):
            interpolated_img[:, :, i] = alpha * img[:, :, i] + grey
        return np.array(interpolated_img, dtype=np.uint8)


class HighpassFilter(object):

    def __init__(self, radius=0.5):
        self.radius = radius
        self.filter_matrix = None

    @staticmethod
    def _create_filter(shape, radius):
        height, width, _ = shape
        height_center, width_center = height / 2, width / 2
        height_r, width_r = height_center * radius, width_center * radius
        filter_matrix = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                if (i - height_center) * (i - height_center) + (j - width_center) * (j - width_center) < height_r * width_r:
                    filter_matrix[i, j] = 0
        return filter_matrix

    def __call__(self, pic):
        img = np.asarray(pic)
        img = img[:224, :224, :]
        height, width, channel = img.shape
        height_center, width_center = height / 2, width / 2
        if self.filter_matrix is None or not self.filter_matrix.shape == img.shape[:2]:
            self.filter_matrix = self._create_filter(img.shape, self.radius)
        highpassed_img = np.zeros(img.shape, dtype=np.uint8)
        for i in range(len('rgb')):
            fft = np.fft.fft2(img[:, :, i])
            fft = np.fft.fftshift(fft)
            fft = fft * self.filter_matrix
            fft = np.fft.fftshift(fft)
           # fft = fft * self.filter_matrix
            highpassed_img[:, :, i] = np.uint8(np.fft.ifft2(fft).real)
        return highpassed_img


class LowpassFilter(object):

    def __init__(self, radius=0.5):
        self.radius = radius
        self.filter_matrix = None

    @staticmethod
    def _create_filter(shape, radius):
        height, width, _ = shape
        height_center, width_center = height / 2, width / 2
        height_r, width_r = height_center * radius, width_center * radius
        filter_matrix = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                if (i - height_center) * (i - height_center) + (j - width_center) * (j - width_center) < height_r * width_r:
                    filter_matrix[i, j] = 1
        return filter_matrix

    def __call__(self, pic):
        img = np.asarray(pic)
        img = img[:356, :356, :]
        height, width, channel = img.shape
        height_center, width_center = height / 2, width / 2
        if self.filter_matrix is None or not self.filter_matrix.shape == img.shape[:2]:
            self.filter_matrix = self._create_filter(img.shape, self.radius)
        highpassed_img = np.zeros(img.shape, dtype=np.uint8)
        for i in range(len('rgb')):
            fft = np.fft.fft2(img[:, :, i])
            fft = np.fft.fftshift(fft)
            fft = fft * self.filter_matrix
            fft = np.fft.fftshift(fft)
            # fft = fft * self.filter_matrix
            highpassed_img[:, :, i] = np.uint8(np.fft.ifft2(fft).real)
        return highpassed_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import utility
    from utility import make_dir
    from torchvision import datasets
    for path in utility.find_files('/home/nutszebra/Downloads/examples/lowpass'):
        img = default_loader(path)
        plt.clf()
        plt.imshow(LowpassFilter(0.01)(img))
        save_path = '{}_lowpass.jpg'.format(path[:-4])
        plt.savefig(save_path)
    for path in utility.find_files('/home/nutszebra/Downloads/examples/shuffle'):
        img = default_loader(path)
        plt.clf()
        plt.imshow(ShuffleTiles((32, 32))(img))
        save_path = '{}_shuffle.jpg'.format(path[:-4])
        plt.savefig(save_path)
    # img = default_loader('/home/nutszebra/Downloads/m_ILSVRC/Data/CLS-LOC/train/n02123394/n02123394_12.JPEG')
    # dataset = datasets.CIFAR10('../data_cifar10', train=True)
    # save_path = '/home/nutszebra/Downloads/custom_transformers'
    # make_dir('{}/lowpass'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.0)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_0.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.01)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_01.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.02)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_02.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.03)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_03.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.04)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_04.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.05)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_05.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.06)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_06.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.07)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_07.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.08)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_08.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.09)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_09.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.1)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_1.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.2)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_2.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.3)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_3.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.4)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_4.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.5)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_5.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.6)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_6.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.7)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_7.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.8)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_8.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.9)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/0_9.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(1.0)(img))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/1_0.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.0)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_0.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.05)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_05.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.1)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_1.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.2)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_2.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.3)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_3.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.4)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_4.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.5)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_5.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.6)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_6.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.7)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_7.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.8)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_8.png'.format(save_path))
    # plt.clf()
    # plt.imshow(LowpassFilter(0.9)(dataset[0][0]))
    # plt.axis('off')
    # plt.savefig('{}/lowpass/cifar_0_9.png'.format(save_path))
    # make_dir('{}/tile'.format(save_path))
    # plt.imshow(ShuffleTiles((1, 1))(img))
    # plt.axis('off')
    # plt.savefig('{}/tile/1.png'.format(save_path))
    # plt.clf()
    # plt.imshow(ShuffleTiles((2, 2))(img))
    # plt.axis('off')
    # plt.savefig('{}/tile/2.png'.format(save_path))
    # plt.clf()
    # plt.imshow(ShuffleTiles((3, 3))(img))
    # plt.axis('off')
    # plt.savefig('{}/tile/3.png'.format(save_path))
    # plt.clf()
    # plt.imshow(ShuffleTiles((4, 4))(img))
    # plt.axis('off')
    # plt.savefig('{}/tile/4.png'.format(save_path))
    # plt.clf()
    # plt.imshow(ShuffleTiles((8, 8))(img))
    # plt.axis('off')
    # plt.savefig('{}/tile/8.png'.format(save_path))
    # plt.clf()
    # plt.imshow(ShuffleTiles((16, 16))(img))
    # plt.axis('off')
    # plt.savefig('{}/tile/16.png'.format(save_path))
    # plt.clf()
    # make_dir('{}/grey'.format(save_path))
    # plt.imshow(InterpolateToGrey(0.0)(img))
    # plt.axis('off')
    # plt.savefig('{}/grey/0_0.png'.format(save_path))
    # plt.clf()
    # plt.imshow(InterpolateToGrey(0.25)(img))
    # plt.axis('off')
    # plt.savefig('{}/grey/0_25.png'.format(save_path))
    # plt.clf()
    # plt.imshow(InterpolateToGrey(0.5)(img))
    # plt.axis('off')
    # plt.savefig('{}/grey/0_5.png'.format(save_path))
    # plt.clf()
    # plt.imshow(InterpolateToGrey(0.75)(img))
    # plt.axis('off')
    # plt.savefig('{}/grey/0_75.png'.format(save_path))
    # plt.clf()
    # plt.imshow(InterpolateToGrey(1.0)(img))
    # plt.axis('off')
    # plt.savefig('{}/grey/1_0.png'.format(save_path))
