import os
import cv2
import glob
import imageio
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset
from torchvision import transforms


class BowelDataset(Dataset):

    def __init__(self, paths, img_size=448, mode='train'):
        super(BowelDataset, self).__init__()
        
        self.img_size = img_size
        self.data = []
        self.mode = mode

        for _path in paths:
            cases = os.listdir(_path)
            print (f"  > {_path}: {len(cases)} cases")

            for _case in cases:
                img = glob.glob(os.path.join(_path, _case, "*.dcm"))
                if len(img) != 1: continue         
                
                label = None
                if self.mode == 'train':
                    label = img[0].replace(".dcm", "_bowel.png")
                    if not os.path.exists(label): continue

                self.data.append({'img': img[0], 'mask': label})

    
    def __getitem__(self, index):
        img = self.load_image(index)
        img = transforms.ToTensor()(img)

        if self.mode == 'test':
            return img

        mask = self.load_mask(index)
        mask = transforms.ToTensor()(mask)
        
        return img, mask


    def load_image(self, index):
        img = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(self.data[index]['img'])))
        
        den_min = np.min(img)
        den_max = np.percentile(img, 99)

        img = np.array(img, dtype='float32')
        img = img - den_min
        img = img / (den_max - den_min)
        img[img < 0.] = 0.
        img[img > 1.] = 1.

        if img.shape[0] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LINEAR)
        
        return img


    def load_mask(self, index):
        if not self.data[index]['mask']: return None

        mask = np.squeeze(imageio.imread(self.data[index]['mask'])).astype('float32')
        mask[mask > 0.] = 1.

        if mask.shape[0] != self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), cv2.INTER_NEAREST)
        return mask


    def __len__(self):
        return len(self.data)


    def save_results(self, pred_mask):
        TH = 0.5
        VERSION = "_v2"
        for i, _data in enumerate(self.data):
            print (_data['img'])
            filename = _data['img']

            prob = np.copy(pred_mask[i])        
            ### resize ->
            img = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(self.data[i]['img'])))
            if img.shape[0] != self.img_size:
                prob = cv2.resize(prob, img.shape, cv2.INTER_LINEAR) 
            ### <- resize
            sitk.WriteImage(sitk.GetImageFromArray(prob), filename.replace(".dcm", f"_prob_bowel{VERSION}.nii.gz"))

            mask = np.copy(prob)
            mask[mask < TH] = 0.
            mask[mask > 0.] = 1.
            mask =  np.array(mask, dtype='uint8')
            sitk.WriteImage(sitk.GetImageFromArray(mask), filename.replace(".dcm", f"_pred_bowel{VERSION}.nii.gz"))

            mask = np.array(mask*255, dtype='uint8')
            imageio.imwrite(filename.replace(".dcm", f"_pred_bowel{VERSION}.png"), mask)


class AverageValue(object):
    """
        Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count