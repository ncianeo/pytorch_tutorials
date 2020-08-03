import cv2 # library for image input/output
import numpy as np # numerical library
from torch.utils.data import Dataset # predefined dataset class
from os import listdir
from os.path import join


BSD_PATH = "/workspace/BSR/BSDS500/data/images/"


class SISR_Dataset(Dataset):
    """Single Image Super Resolution Dataset"""
    def __init__(self, mode='train', size=256, ratio=2):
        """
        Args:
            mode (string): 'train' or 'valid' or 'test'
            size (int): image size
            ratio (int): super resolution ratio=2, 4, 8
        """
        self.mode = mode
        root = join(BSD_PATH, mode)
        self.y_paths = [join(root, f) for f in listdir(root) if f.endswith('.jpg')]
        self.size = size
        self.ratio = ratio

    def __len__(self):
        return len(self.y_paths)

    def __getitem__(self, index):
        # read image and type casting to avoid value wrapping
        y = cv2.imread(self.y_paths[index]).astype(np.float32)
        size = self.size
        h, w = y.shape[:2]
        # random cropping & rotation
        # 1. randomly choose the center
        center_row = np.random.randint(size//2, max(size//2+1, h-size//2))
        center_col = np.random.randint(size//2, max(size//2+1, w-size//2))
        center = (center_col, center_row)
        # 2. generate rotation matrix
        angle = np.random.randint(-180, 180)
        R = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 3. apply rotation transform
        y = cv2.warpAffine(y, R, (w, h))
        # 4. apply cropping
        if h >= size:
            row_slice = slice(center_row-size//2, center_row+size//2)
        else:
            row_slice = slice(0, h)
        if w >= size:
            col_slice = slice(center_col-size//2, center_col+size//2)
        else:
            col_slice = slice(0, w)
        y = y[row_slice, col_slice, ...]
        # if patch size is smaller than size, resize the image
        if y.shape[0] < size or y.shape[1] < size:
            y = cv2.resize(y, (size, size), cv2.INTER_CUBIC)
        
        # generate small input image
        ratio = self.ratio
        x = cv2.resize(y, (size//ratio, size//ratio), cv2.INTER_CUBIC)

        # change axis from (h, w, c) to (c, h, w) & normalize
        x = 2*(x.transpose(2, 0, 1)/255 - 0.5)
        y = 2*(y.transpose(2, 0, 1)/255 - 0.5)
        return (x, y)


def tensor_to_image_array(tensor):
    if type(tensor) == np.ndarray:
        arr = tensor
    else:
        arr = tensor.squeeze().cpu().numpy()
    arr = arr.transpose(1, 2, 0)
    return ((arr+1)/2*255).astype(np.uint8)


if __name__ == '__main__':
    dataset = SISR_Dataset()
    x, y = dataset[0]
    print(x.shape)
    print(y.shape)
    x_img = tensor_to_image_array(x)
    y_img = tensor_to_image_array(y)
    cv2.imshow('x', x_img)
    cv2.imshow('y', y_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
