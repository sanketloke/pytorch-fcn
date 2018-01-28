import random
from PIL import Image
import math

def randomHorizontalFlip( img1,img2):
    if random.random() < 0.5:
        return ( img1.transpose(Image.FLIP_LEFT_RIGHT) , img2.transpose(Image.FLIP_LEFT_RIGHT) )
    return (img1,img2)


def randomSizedCrop(img1,img2,size,interpolation=Image.BILINEAR):
        for attempt in range(10):
            area = img1.size[0] * img1.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img1.size[0] and h <= img1.size[1]:
                x1 = random.randint(0, img1.size[0] - w)
                y1 = random.randint(0, img1.size[1] - h)
                img1 = img1.crop((x1, y1, x1 + w, y1 + h))
                img2 =  img2.crop((x1, y1, x1 + w, y1 + h))
                assert(img1.size == (w, h))
                assert(img2.size == (w, h))
                return ( img1.resize((size, size), interpolation),
                             img2.resize((size, size), interpolation) )
