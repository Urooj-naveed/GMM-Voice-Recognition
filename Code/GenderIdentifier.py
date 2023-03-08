import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
#from Trainer import score
#from Trainer3 import score
from Trainer4 import score

warnings.filterwarnings("ignore")


class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        accurate_male_list = []
        accurate_female_list = []
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            winner = self.identify_gender(vector)
            expected_gender = file.split("/")[1][:1]

            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

            if winner[0] != expected_gender: 
                self.error += 1
            else:
                if expected_gender.lower() == 'm':
                    accurate_male_list.append(os.path.basename(file))
                else:
                    accurate_female_list.append(os.path.basename(file))
            print("----------------------------------------------------")

        for x in accurate_male_list:
            amo_message = "*** Matched the following male audio  = " + str(x) + " ***"
            print(amo_message)
        
        for y in accurate_female_list:
            afo_message = "*** Matched the following female audio = " + str(y) + " ***"
            print(afo_message)

        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accurateelements_msg = "*** number of accurate elements = " + str(float(self.total_sample - self.error)) + " ***"
        print(accurateelements_msg)
        wrongelements_msg = "*** number of wrong elements = " + str(float(self.error)) + " ***"
        print(wrongelements_msg)
        totalelements_msg = "*** total number of elements = " + str(float(self.total_sample)) + " ***"
        print(totalelements_msg)
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)

    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        files   = females + males
        return files

    def identify_gender(self, vector):
        # female hypothesis scoring
        #is_female_scores         = np.array(self.females_gmm.score(vector))
        is_female_scores         = np.array(score(vector, self.females_gmm.means_, self.females_gmm.covariances_, self.females_gmm.weights_))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        #is_male_scores         = np.array(self.males_gmm.score(vector))
        is_male_scores         = np.array(score(vector, self.males_gmm.means_, self.males_gmm.covariances_, self.males_gmm.weights_))
        is_male_log_likelihood = is_male_scores.sum()

        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood,3))))

        if is_male_log_likelihood > is_female_log_likelihood: winner = "male"
        else                                                : winner = "female"
        return winner


if __name__== "__main__":
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.gmm", "males.gmm")
    gender_identifier.process()
