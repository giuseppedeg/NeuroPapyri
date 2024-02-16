print("This might take several minutes...\n")

print("Downloading ALPUB_V2 dataset...")
import utils.dataset.download_alpub_v2
# Data Download at: https://data.cs.mtsu.edu/al-pub/

print("\nPreparing ALPUB_V2 dataset for Character Identification...")
import utils.dataset.generate_dataset_alpubv2

print("Downloading ICDAR dataset...")
import utils.dataset.download_ICDAR
# Data Download at: https://lme.tf.fau.de/competitions/2023-competition-on-detection-and-recognition-of-greek-letters-on-papyri/

print("\nPreparing ICDAR dataset for Character Identification...")
import utils.dataset.generate_dataset_icdar_ci

print("\nPreparing ICDAR dataset for Document Retrieval...")
import utils.dataset.generate_dataset_icdar_dr
# You can edit the file to define the number of model head editing the "letters" list