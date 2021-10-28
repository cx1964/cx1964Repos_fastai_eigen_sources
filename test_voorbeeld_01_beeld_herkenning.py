# file: test_voorbeeld_01_beeld_herkenning.py
# functie: Voorbeeld om te kijken of de fastai python envirnment goed werkt
#          Dit voorbeld is overgenomen uit jupyter notebook 01_intro.ipynb
#          dat hoort bij het boek "Deep Learning for Coders with fastai and PyTorch"
#          Zie hoofdstuk 1 eerste code voorbeeld.       
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

# Deze package bevat de functies en classes die voor het creeren van computer vision models.
from fastai.vision.all import *

from torch._C import BoolStorageBase

# Bepaal een path naar de plaatjes van Oxford-IIIT Pet Dataset
# Dit commando download ook de dataset met plaatjes.
path = untar_data(URLs.PETS)/'images'
print('path ', path)

### Label functie ###
# Deze functie labelt de plaatjes als katten obv een filename rule
# die gegeven is door de creator van de dataset.
# Alle plaatjes van katten in de dataset hebben een filenaam die
# bestaan uit kleine letters.
# Bij computer vision datasets is het gebruikelijk dat de datset dusdanig is
# gestructureerd dat label van de image onderdeel uitmaakt van de filenaam of
# het path. Onderstaande functie lablet de plaatjes.
# De inhoud van deze functie wordt bepaald welke regels zijn gedefinieerd om
# de label van een plaatje te bepalen.
def is_cat(x): 
    return x[0].isupper()

# Onderstaande expressie geeft fastai aan wat voor type dataset we verwerken
# en hoe het is gestructureerd. ImageDataLoaders geeft aan dat we plaatjes gaan verwerken. 
dls = ImageDataLoaders.from_name_func(
         path
        ,bs=8 # batch size toegevoegd om programma niet te laten crashen
        ,fnames=get_image_files(path)
        ,valid_pct=0.2 # Deze parameter geeft aan dat fastai 20% van de data niet mag gebruiken om te trainen
                       # Deze 20% van de data wordt de validation set genoemd. 
        ,seed=42
        ,label_func=is_cat # deze functie bepaalt of foto een kat is
        ,item_tfms=Resize(224)
)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

img = PILImage.create('/home/claude/Desktop/sources/fast.ai/cx1964Repos_fastai_eigen_sources/test_foto_kat.jpg')
#img.show() # dit werkt niet?
img.to_thumb(128,128) # dit werkt niet

is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
