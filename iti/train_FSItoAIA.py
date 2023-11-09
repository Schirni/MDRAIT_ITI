from data.Dataset import FSIDataset, AIADataset, StorageDataset
from iti.trainer_iti import Trainer
from iti.Model import DiscriminatorMode
from data.Editor import RandomPatchEditor
import argparse
import logging
import wandb


wandb.init(project="FSItoAIA_304")


parser = argparse.ArgumentParser(description='Homogenize EUI/FSI data to AIA data')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--hq_path', type=str, help='path to the AIA data.')
parser.add_argument('--lq_path', type=str, help='path to the FSI data.')
parser.add_argument('--hq_converted_path', type=str, help='path to store the converted AIA data.')
parser.add_argument('--lq_converted_path', type=str, help='path to store the converted FSI data.')

args = parser.parse_args()
base_dir = args.base_dir
low_path = args.lq_path
low_converted_path = args.lq_converted_path
high_path = args.hq_path
high_converted_path = args.hq_converted_path

log_iteration = 1000
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

#base_dir = '/gpfs/data/fs71254/schirni/Training/Training2'
#low_path = '/gpfs/data/fs71254/schirni/Level1_Files/Level1_Files_GBand'
#high_path = '/gpfs/data/fs71254/schirni/Level2_Files/Level2_Files_GBand'
#low_converted_path = '/gpfs/data/fs71254/schirni/Lvl1GBand_converter'
#high_converted_path = '/gpfs/data/fs71254/schirni/Lvl2Gband_converter'

trainer = Trainer(input_dim_a=1, input_dim_b=1, norm='in_rs_aff', discriminator_mode=DiscriminatorMode.SINGLE,
                  lambda_diversity=0, upsampling=1)
trainer.cuda()

FSI_train = FSIDataset(low_path, wavelength=304)
FSI_train_storage = StorageDataset(FSI_train, low_converted_path,
                                          ext_editors=[RandomPatchEditor((128, 128))])

AIA_train = AIADataset(high_path, wavelength=304)
AIA_train_storage = StorageDataset(AIA_train, high_converted_path,
                                           ext_editors=[RandomPatchEditor((256, 256))])

FSI_valid = FSIDataset(low_path, wavelength=304, limit=20)
FSI_valid_storage = StorageDataset(FSI_valid, low_converted_path,
                                          ext_editors=[RandomPatchEditor((128, 128))])
AIA_valid = AIADataset(high_path, wavelength=304, limit=20)
AIA_valid_storage = StorageDataset(AIA_valid, high_converted_path,
                                           ext_editors=[RandomPatchEditor((256, 256))])

plot_settings_A = {"cmap": "sdoaia304", "title": "EUI/FSI", 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "sdoaia304", "title": "SDO/AIA", 'vmin': -1, 'vmax': 1}

trainer.startBasicTraining(base_dir, FSI_train_storage, AIA_train_storage, FSI_valid_storage, AIA_valid_storage, num_workers=1, plot_settings_A=plot_settings_A, plot_settings_B=plot_settings_B)


#python3 train_FSItoAIA.py --base_dir '/home/christophschirninger/ITI/FSItoAIA_304' --lq_path '/mnt/disks/data/FSI/eui-fsi304-image' --hq_path '/mnt/disks/data/SDO/304' --lq_converted_path '/home/christophschirninger/ITI/EUI_storage' --hq_converted_path '/home/christophschirninger/ITI/AIA_storage'