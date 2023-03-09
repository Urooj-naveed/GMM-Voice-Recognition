import os
import pickle
import warnings
import numpy as np
from sklearn import mixture
from FeaturesExtractor import FeaturesExtractor
#from Trainer import fit
#from Trainer2 import fit
#from Trainer3 import fit
#from Trainer4 import fit
from Trainer5 import fit
from MyGMM import MyGMM
from datetime import datetime

warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def process(self):
        # datetime object containing current date and time
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        # collect voice features
        female_voice_features = self.collect_features(females)
        male_voice_features   = self.collect_features(males)
        #Option 1--------------------------------------------------------------------------------------
        # generate gaussian mixture models
        #females_gmm = MyGMM(n_components=2, tol=1e-4, max_iter=200, covariance_type='diag', n_init=1)
        #males_gmm = MyGMM(n_components=2, tol=1e-4, max_iter=200, covariance_type='diag', n_init=1)
        # fit features to models
        #females_gmm.fit(female_voice_features)
        #males_gmm.fit(male_voice_features)
        #----------------------------------------------------------------------------------------------
        #Option 2--------------------------------------------------------------------------------------
        #print('Step 1 Initialize the Model females_gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type=diag, n_init = 3)')
        #females_gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        #print('Step 2 Initialize the Model males_gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type=diag, n_init = 3)')
        #males_gmm   = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        #now = datetime.now()
        #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #print("Step 3 females_gmm = fit(females_gmm, female_voice_features) - Start date and time =", dt_string)
        #females_gmm = fit(females_gmm, female_voice_features)
        #now = datetime.now()
        #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #print("Step 3 females_gmm = fit(females_gmm, female_voice_features) - End date and time =", dt_string)
        #now = datetime.now()
        #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #print("Step 4 males_gmm = fit(males_gmm, male_voice_features) - Start date and time =", dt_string)
        #males_gmm = fit(males_gmm, male_voice_features)
        #now = datetime.now()
        #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #print("Step 4 males_gmm = fit(males_gmm, male_voice_features) - End date and time =", dt_string)
        #----------------------------------------------------------------------------------------------
        #Option 3--------------------------------------------------------------------------------------
        print('Step 1 Initialize the Model females_gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type=diag, n_init = 3)')
        females_gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        print('Step 2 Initialize the Model males_gmm   = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type=diag, n_init = 3)')
        males_gmm   = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Step 3 females_gmm = fit(females_gmm, female_voice_features) - Start date and time =", dt_string)
        females_gmm = fit(female_voice_features, 16)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Step 3 females_gmm = fit(females_gmm, female_voice_features) - End date and time =", dt_string)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Step 4 males_gmm = fit(males_gmm, male_voice_features) - Start date and time =", dt_string)
        males_gmm = fit(male_voice_features, 16)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Step 4 males_gmm = fit(males_gmm, male_voice_features) - End date and time =", dt_string)
        #----------------------------------------------------------------------------------------------
        # save models
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")

    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        return females, males

    def collect_features(self, files):
        """
    	Collect voice features from various speakers of the same gender.

    	Args:
    	    files (list) : List of voice file paths.

    	Returns:
    	    (array) : Extracted features matrix.
    	"""
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector    = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))
        return features

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.

            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))


if __name__== "__main__":
    models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    models_trainer.process()
