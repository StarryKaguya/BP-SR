from torch.utils import data as data
from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.data_util import paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import img2tensor, scandir
from os import path as osp
import cv2
import random
import torch

@DATASET_REGISTRY.register()
class NtireImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(NtireImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.diff_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_diff']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = self.paired_paths_from_folder([self.lq_folder, self.gt_folder, self.diff_folder], ['lq', 'gt','diff'], self.filename_tmpl)

    def paired_paths_from_folder(self, folders, keys, filename_tmpl):
        """Generate paired paths from folders.

        Args:
            folders (list[str]): A list of folder path. The order of list should
                be [input_folder, gt_folder].
            keys (list[str]): A list of keys identifying folders. The order should
                be in consistent with folders, e.g., ['lq', 'gt'].
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Usually the filename_tmpl is
                for files in the input folder.

        Returns:
            list[str]: Returned path list.
        """
        assert len(folders) == 3, ('The len of folders should be 3 with [input_folder, gt_folder]. '
                                f'But got {len(folders)}')
        assert len(keys) == 3, f'The len of keys should be 3 with [input_key, gt_key]. But got {len(keys)}'
        input_folder, gt_folder, diff_folder = folders
        input_key, gt_key, diff_key = keys
        input_paths = list(scandir(input_folder))
        diff_paths = list(scandir(diff_folder))
        gt_paths = list(scandir(gt_folder))
        assert len(input_paths) == len(gt_paths), (f'{input_key} and {diff_key} and {gt_key} datasets have different number of images: '
                                                f'{len(input_paths)}, {len(gt_paths)}.')
        paths = []
        for gt_path in gt_paths:
            basename, ext = osp.splitext(osp.basename(gt_path))
            input_name = f'{filename_tmpl.format(basename)}{ext}'
            input_path = osp.join(input_folder, input_name)
            diff_path = osp.join(diff_folder, input_name)
            assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
            assert input_name in diff_paths, f'{input_name} is not in {diff_key}_paths.'
            gt_path = osp.join(gt_folder, gt_path)
            paths.append(dict([(f'{input_key}_path', input_path),(f'{diff_key}_path', diff_path), (f'{gt_key}_path', gt_path)]))
        return paths
    

    def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
        """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

        We use vertical flip and transpose for rotation implementation.
        All the images in the list use the same augmentation.

        Args:
            imgs (list[ndarray] | ndarray): Images to be augmented. If the input
                is an ndarray, it will be transformed to a list.
            hflip (bool): Horizontal flip. Default: True.
            rotation (bool): Ratotation. Default: True.
            flows (list[ndarray]: Flows to be augmented. If the input is an
                ndarray, it will be transformed to a list.
                Dimension is (h, w, 2). Default: None.
            return_status (bool): Return the status of flip and rotation.
                Default: False.

        Returns:
            list[ndarray] | ndarray: Augmented images and flows. If returned
                results only have one element, just return ndarray.

        """
        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        def _augment_flow(flow):
            if hflip:  # horizontal
                cv2.flip(flow, 1, flow)
                flow[:, :, 0] *= -1
            if vflip:  # vertical
                cv2.flip(flow, 0, flow)
                flow[:, :, 1] *= -1
            if rot90:
                flow = flow.transpose(1, 0, 2)
                flow = flow[:, :, [1, 0]]
            return flow

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        if flows is not None:
            if not isinstance(flows, list):
                flows = [flows]
            flows = [_augment_flow(flow) for flow in flows]
            if len(flows) == 1:
                flows = flows[0]
            return imgs, flows
        else:
            if return_status:
                return imgs, (hflip, vflip, rot90)
            else:
                return imgs

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        diff_path = self.paths[index]['diff_path']
        img_bytes = self.file_client.get(diff_path, 'diff')
        img_diff = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, img_diff = self.paired_random_crop(img_gt, img_lq, img_diff, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_diff = augment([img_gt, img_lq, img_diff], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
            img_diff = bgr2ycbcr(img_diff, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_diff = img2tensor([img_gt, img_lq, img_diff], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_diff, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'diff': img_diff, 'gt': img_gt, 'lq_path': lq_path, 'diff_path': diff_path, 'gt_path': gt_path}

    def paired_random_crop(self, img_gts, img_lqs, img_diffs, gt_patch_size, scale, gt_path=None):
        """
        Paired random crop. Support Numpy array and Tensor inputs.

        It crops lists of lq, diff and gt images with corresponding locations.

        Args:
            img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images.
            img_lqs (list[ndarray] | ndarray): LQ images.
            img_diffs (list[ndarray] | ndarray): Diff images.
            gt_patch_size (int): GT patch size.
            scale (int): Scale factor.
            gt_path (str): Path to ground-truth. Default: None.

        Returns:
            list[ndarray] | ndarray: GT images, LQ images and Diff images. If returned results
                only have one element, just return ndarray.
        """
        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(img_diffs, list):
            img_diffs = [img_diffs]

        # determine input type: Numpy array or Tensor
        input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

        if input_type == 'Tensor':
            h_lq, w_lq = img_lqs[0].size()[-2:]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            h_lq, w_lq = img_lqs[0].shape[0:2]
            h_gt, w_gt = img_gts[0].shape[0:2]

        lq_patch_size = gt_patch_size // scale

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ' \
                             f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ' \
                             f'({lq_patch_size}, {lq_patch_size}). ' \
                             f'Please remove {gt_path}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # crop lq patch and diff patch
        if input_type == 'Tensor':
            img_lqs = [v[:, :, top:top+lq_patch_size, left:left+lq_patch_size] for v in img_lqs]
            img_diffs = [v[:, :, top:top+lq_patch_size, left:left+lq_patch_size] for v in img_diffs]
        else:
            img_lqs = [v[top:top+lq_patch_size, left:left+lq_patch_size, ...] for v in img_lqs]
            img_diffs = [v[top:top+lq_patch_size, left:left+lq_patch_size, ...] for v in img_diffs]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        if input_type == 'Tensor':
            img_gts = [v[:, :, top_gt:top_gt+gt_patch_size, left_gt:left_gt+gt_patch_size] for v in img_gts]
        else:
            img_gts = [v[top_gt:top_gt+gt_patch_size, left_gt:left_gt+gt_patch_size, ...] for v in img_gts]

        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(img_diffs) == 1:
            img_diffs = img_diffs[0]

        return img_gts, img_lqs, img_diffs
    def __len__(self):
        return len(self.paths)