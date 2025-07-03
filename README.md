Preparation of the Data


Ensure that the data is in the following folder format:
/
F02
F03
F04
F05
M01
M04
M05
M07
M08
M09
M10
M11
M12
M14
M16
speaker wordlist.xls
Note that the speaker_wordlist.xls file provided with the UASpeech dataset is included.


This file structure organization is the only pre-processing we need.


Installation of the necessary dependencies


The codebase includes a requirements.txt file for ease of dependency installation and proper
package versioning. We use Python version 12.2. However, first, we must install Pytorch
(torch) in a specific way to ensure that it is functional. The Pytorch version used is 2.5.1
with CUDA12.1. Adjust the CUDA version (replace cu121 with the appropriate CUDA version)
according to the GPU used.



pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

Then we run:
pip install -r requirements.txt



Running the Experiment
After installing the required packages, all that is left is to run the experiment script, implemented
in main.py. We run the script using the following format:


python main.py --data_dir /path/to/data --output_dir  /path/to/results/dir --device cuda or cpu --truncation truncation_value  --epochs number_of_epochs


where
	data_dir is the directory containing the data as defined,
	output_dir is the path to the output directory (i.e., the directory the results are to be
	stored in),
	device is the device where the training will occur (cuda for gpu or cpu, default cuda),
	truncation is the number of seconds to truncate every speech sample to (default 12),
	epochs is the number of epochs to train (default 20).



Output
The output is stored in the output directory specified in output_dir. In this directory, the
confusion matrices from each epoch are saved, along with the best model and the model from
the final epoch, for checkpointing purposes. Additionally, there are csv files named wrong_pred
ictions_epoch_epoch_number.csv which contain, for each misclassified sample, the true and
predicted classes. Also, there is a file called training_summary.txt, which has the following
format:


Best validation accuracy: 52.69%
Test speakers: [’M04’, ’M16’, ’M11’, ’F05’]
Excluded speakers: [’M01’, ’M09’, ’M10’]
Training samples: 3266
Validation samples: 1820
Epoch of best model 3
Training accuracy of best model 0.5296999387630129


ROC Curves For the production of ROC curves and additional metrics, another script, roc.py, is included. We run the script in the following way:

python roc.py --folder1 path/to/output_folder --folder2 path/to/second/output/folder --output_dir /path/to/output/dir


There is an optional option for a second output folder due to the large space required to save
the experiment models, which may span multiple disks. In the folder output_dir, the average
confusion matrix and average ROC curve are saved.

