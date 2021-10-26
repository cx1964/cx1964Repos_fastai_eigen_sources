# file: test_voorbeeld 
# functie: voorbeeld om te kijken of de fastai python envirnment goed werkt
#
# Deze code voert het volgende uit:
# A dataset called the Oxford-IIIT Pet Dataset that contains 7,349 images of cats and dogs from 37 breeds will be downloaded
# from the fast.ai datasets collection to the GPU server you are using, and will then be extracted.
#
# A pretrained model that has already been trained on 1.3 million images using a competition-winning model will be downloaded
# from the internet.
#
# The pretrained model will be fine-tuned using the latest advances in transfer learning to create a model that is specially
# customized for recognizing dogs and cats.


import fastbook
from fastai.vision.all import *

# Bepaal een path naar de plaatjes van Oxford-IIIT Pet Dataset
path = untar_data(URLs.PETS)/'images'
print('path ', path)

'''
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
'''

img = PILImage.create(image_cat())
img.to_thumb(192)
print('Wat gebeurt er?')

'''
img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
'''

