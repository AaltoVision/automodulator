import torch
from PIL import Image
import numpy as np
import torchvision.utils

# Use as:
# z_fused = model.zbuilder().use(zA,[0,1]).use(zB,[2,3]).use(zA,[3,-1])
# or:
# z_fused = model.zbuilder().high(zA).mid(zB).lo(zA)
# imgAB = model.decode(z_fused)

repo = 'AaltoVision/automodulator'

#datatype = 'celebahq256'
#datatype = 'ffhq256'
datatype = 'ffhq512'
#datatype = 'lsuncars256'
#datatype = 'lsunbedrooms256'

model = torch.hub.load(repo, datatype, pretrained=True, source='github', force_reload = True)
model.eval(useLN=False)

# If you do not need the Low-memory footprint approach, just run model.generate(N) once for the N you need.
for k in range(4):
    omg = model.generate(1)
    torchvision.utils.save_image(omg[0]/ 2 + 0.5, fp=f'{datatype}_rand_{k}.png')
    del omg

if datatype in ['celebahq256', 'ffhq256', 'ffhq512']:
    simg = {}
    simg[0] = 'fig/source-0.png'
    simg[1] = 'fig/source-1.png'

    # Load 2 images to torch
    imgs = torch.stack([model.tf()(Image.open(simg[0])),
                        model.tf()(Image.open(simg[1]))])

    z = model.encode(imgs)

    # Reconstruct
    for i,img in enumerate(simg):
        omg = model.decode( model.zbuilder().hi(z[i]).mid(z[i]).lo(z[i]))
        torchvision.utils.save_image(omg[0], normalize=True, fp=f'{datatype}_reco_{i}.png')

    # Create typical mixture images
    omg = model.decode( model.zbuilder().hi(z[0]).mid(z[1]).lo(z[1]))
    torchvision.utils.save_image(omg[0], normalize=True, fp=f'{datatype}_mixB_011.png')

    omg = model.decode( model.zbuilder().hi(z[1]).mid(z[0]).lo(z[1]) )
    torchvision.utils.save_image(omg[0], normalize=True, fp=f'{datatype}_mixB_101.png')

    omg = model.decode( model.zbuilder().hi(z[1]).mid(z[1]).lo(z[0]) )
    torchvision.utils.save_image(omg[0], normalize=True, fp=f'{datatype}_mixB_110.png')

    omg = model.decode( model.zbuilder().hi(z[0]).mid(z[1]).lo(z[0]) )
    torchvision.utils.save_image(omg[0], normalize=True, fp=f'{datatype}_mixB_010.png')

    omg = model.decode( model.zbuilder().hi(z[1]).mid(z[0]).lo(z[0]) )
    torchvision.utils.save_image(omg[0], normalize=True, fp=f'{datatype}_mixB_100.png')

    # Mix two latents at every possible cut-off point (you could also interleave, not shown here)

    omgs = torch.stack([model.decode( model.zbuilder().use(z[0], [0,i]).use(z[1], [i,-1]) )
                        for i in range(0,9)])

    for j in range(9):
        torchvision.utils.save_image(omgs[j], normalize=True, fp=f'mix_{j}.png')
