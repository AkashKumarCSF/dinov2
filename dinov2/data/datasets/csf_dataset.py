import os

from PIL import Image
from torch.utils.data import Dataset


class CSFDataset(Dataset):

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
    ):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        #print("Root CSF dataset... ", self.root)

        valid_ext = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        )

        for dirpath, _, filenames in os.walk(root):

            for f in filenames:

                if f.lower().endswith(valid_ext):

                    self.samples.append(
                        os.path.join(dirpath, f)
                    )

        self.samples.sort()

        print(f"Found {len(self.samples)} images.")


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        path = self.samples[index]
        #print("Loading image... ", path)
        image = Image.open(path).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        target = ()

        if self.target_transform is not None:

            target = self.target_transform(target)

        return image, target