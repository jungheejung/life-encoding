import os
from shutil import copyfile

data_dir= '/idata/DBIC/cara/life/ridge/'
dest_dir = '/idata/DBIC/cara/weights_for_sam'
for model in ['avg-slh-CV', 'avg-ana-CV']:
    for sub in os.listdir(os.path.join(data_dir, model)):
        if not os.path.exists(os.path.join(dest_dir, model)):
            os.makedirs(os.path.join(dest_dir, model))
        if 'sub' in sub:
            print(sub)
            for h in ['lh', 'rh']:
                if not os.path.exists(os.path.join(dest_dir, model, h)):
                    os.makedirs(os.path.join(dest_dir, model, h))
                copyfile(os.path.join(data_dir, model, sub, h, 'weights.npy'), os.path.join(dest_dir, model, h, 'weights.{0}.npy'.format(sub)))
