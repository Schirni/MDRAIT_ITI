import argparse
import os

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sunpy.visualization.colormaps import cm

from iti.callback import SaveCallback, PlotBAB, PlotABA
from iti.data.dataset import SDODataset, StorageDataset, FSIDataset, ITIDataModule, AIADataset, SDODataset2, EUIDataset
from iti.data.editor import RandomPatchEditor, BrightestPixelPatchEditor
from iti.iti import ITIModule


parser = argparse.ArgumentParser(description='Train EUI/FSI-To-AIA translation')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

args = parser.parse_args()

with open(args.config, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

base_dir = config['base_dir']
os.makedirs(base_dir, exist_ok=True)

# Init Dataset
data_config = config['data']
fsi_path = data_config['A_path']
fsi_converted_path = data_config['converted_A_path']
sdo_path = data_config['B_path']
sdo_converted_path = data_config['converted_B_path']

test_months = [11, 12]
train_months = list(range(2, 10))

sdo_dataset = SDODataset2(sdo_path, months=train_months)
sdo_dataset = StorageDataset(sdo_dataset,
                             sdo_converted_path,
                             ext_editors=[RandomPatchEditor((256, 256))])

fsi_dataset = EUIDataset(fsi_path, months=train_months)
fsi_dataset = StorageDataset(fsi_dataset,
                             fsi_converted_path,
                             ext_editors=[RandomPatchEditor((128, 128))])

sdo_valid = StorageDataset(SDODataset2(sdo_path, months=test_months, limit=200), sdo_converted_path,
                       ext_editors=[RandomPatchEditor((256, 256))])
fsi_valid = StorageDataset(EUIDataset(fsi_path, months=test_months, limit=200), fsi_converted_path,
                          ext_editors=[RandomPatchEditor((128, 128))])

data_module = ITIDataModule(fsi_dataset, sdo_dataset, fsi_valid, sdo_valid, **config['data'])

plot_settings_A = [
    {"cmap": cm.sdoaia171, "title": "FSI 174", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "FSI 304", 'vmin': -1, 'vmax': 1},
]
plot_settings_B = [
    {"cmap": cm.sdoaia171, "title": "SDO 171", 'vmin': -1, 'vmax': 1},
    {"cmap": cm.sdoaia304, "title": "SDO 304", 'vmin': -1, 'vmax': 1},
]

# setup logging
logging_config = config['logging']
wandb_id = logging_config['wandb_id'] if 'wandb_id' in logging_config else None
log_model = logging_config['log_model'] if 'wandb_log_model' in logging_config else False
wandb_logger = WandbLogger(project=logging_config['wandb_project'], name=logging_config['wandb_name'], offline=False,
                           entity=logging_config['wandb_entity'], id=wandb_id, dir=config['base_dir'], log_model=log_model)
wandb_logger.experiment.config.update(config, allow_val_change=True)

# Start traiu
module = ITIModule(**config['model'])

# setup sava callbacks
checkpoint_callback = ModelCheckpoint(dirpath=base_dir, save_last=True, every_n_epochs=1)
save_callback = SaveCallback(base_dir)

# setup plot callbacks
prediction_dir = os.path.join(base_dir, 'predictions')
os.makedirs(prediction_dir, exist_ok=True)
plot_callbacks = []
plot_callbacks += [PlotBAB(sdo_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]
plot_callbacks += [PlotABA(fsi_valid.sample(4), module, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)]

n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=int(config['training']['epochs']),
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=-1,
                  callbacks=[checkpoint_callback, save_callback, *plot_callbacks], )

trainer.fit(module, data_module, ckpt_path='last')