import os, cv2, glob, random, numpy as np, pyclipper, torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset

class DetDataset(Dataset):
    def __init__(self, image_dir, gt_dir, img_size=(640, 640), shrink_ratio=0.4, augment=True):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.shrink_ratio = shrink_ratio
        self.augment = augment
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        print(f'DetDataset: {len(self.image_paths)} images from {image_dir}')

    def __len__(self): return len(self.image_paths)

    def load_gt(self, gt_path):
        polys, tags = [], []
        if not os.path.exists(gt_path): return np.array([]), np.array([])
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                try:
                    coords = [float(x) for x in parts[:8]]
                    poly = np.array(coords).reshape((4, 2))
                    text = ','.join(parts[8:])
                    tags.append(text == '###')
                    polys.append(poly)
                except: continue
        return np.array(polys) if polys else np.array([]), np.array(tags) if tags else np.array([])

    def shrink_polygon(self, poly):
        polygon = Polygon(poly)
        if polygon.area == 0 or polygon.length == 0: return poly
        d = polygon.area * (1 - self.shrink_ratio ** 2) / polygon.length
        clipper = pyclipper.PyclipperOffset()
        clipper.AddPath(poly.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrunk = clipper.Execute(-d)
        return np.array(shrunk[0]).reshape(-1, 2) if shrunk else poly

    def augment_image(self, img, polys):
        h, w = img.shape[:2]
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            if len(polys) > 0:
                for i in range(len(polys)):
                    pts = np.hstack([polys[i], np.ones((len(polys[i]), 1))])
                    polys[i] = pts @ M.T
        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            if len(polys) > 0:
                for i in range(len(polys)):
                    polys[i][:, 0] = w - polys[i][:, 0]
        return img, polys

    def gen_maps(self, shape, polys, tags):
        h, w = shape
        prob = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i, poly in enumerate(polys):
            poly_int = np.round(poly).astype(np.int32)
            if tags[i]:
                cv2.fillPoly(mask, [poly_int], 0.0)
                continue
            shrunk = self.shrink_polygon(poly)
            cv2.fillPoly(prob, [np.round(shrunk).astype(np.int32)], 1.0)
        return prob, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        name = os.path.basename(img_path).split('.')[0]
        gt_path = os.path.join(self.gt_dir, f'gt_{name}.txt')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(self.gt_dir, f'{name}.txt')

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        polys, tags = self.load_gt(gt_path)

        # Augmentation
        if self.augment and len(polys) > 0:
            img, polys = self.augment_image(img, polys)

        # Resize
        sx, sy = self.img_size[0] / w, self.img_size[1] / h
        img = cv2.resize(img, self.img_size)
        if len(polys) > 0:
            polys[:, :, 0] *= sx
            polys[:, :, 1] *= sy

        prob, mask = self.gen_maps(self.img_size, polys, tags)
        thresh = np.zeros_like(prob)
        thresh_mask = mask.copy()

        # Normalize
        img = img.astype(np.float32) / 255.0
        img[:,:,0] = (img[:,:,0] - 0.485) / 0.229
        img[:,:,1] = (img[:,:,1] - 0.456) / 0.224
        img[:,:,2] = (img[:,:,2] - 0.406) / 0.225

        return {
            'image': torch.from_numpy(img).permute(2,0,1),
            'prob_map': torch.from_numpy(prob).unsqueeze(0),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'thresh_map': torch.from_numpy(thresh).unsqueeze(0),
            'thresh_mask': torch.from_numpy(thresh_mask).unsqueeze(0),
        }
