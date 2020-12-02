"""
------------------------------------
------------------------------------
Computer Vision - Purdue University - Homework 11
------------------------------------
Face Recognition & Object Detection
------------------------------------
Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Dec 2, 2020

Reference : https://engineering.purdue.edu/RVL/ECE661_2018/Homeworks/HW10/2BestSolutions/2.pdf

------------------------------------
[TO RUN CODE]: python3 ObjectDetection.py
------------------------------------
------------------------------------
"""
import os
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm


class ViolaJonesOD:
    """
    Main class to perform Object Detection using the Viola Jones Algorithm
    """
    def __init__(self, folder_locations):
        """
        Initialise the object with the folder locations of the data sets
        :param folder_locations:
        """
        self.folders = folder_locations

    def scheduler(self):
        """
        This function runs all the required functions to get the Object Detection network
        running
        :return:
        """
        tester = self.GetFeatures(data_path=[self.folders[0], self.folders[1]],type='train.obj')
        tester.scheduler()
        tester = self.GetFeatures(data_path=[self.folders[2], self.folders[3]],type='test.obj')
        tester.scheduler()
        obj = self.getClassifier()
        trainadaboost = self.AdaBoostTrain(obj=obj,feature_path='train.obj')
        trainadaboost.scheduler()
        testadaboost = self.AdaBoostTest(obj=obj, feature_path='test.obj', model_path='classifier.obj')
        testadaboost.scheduler()

    class getClassifier:
        """
        This is the class to get the classifier for the cascaded AdaBoost approach
        """

        def get_weight(self, samplesP, samplesN):
            """
            Get weights for the given sample set
            :param samplesP: Positives
            :param samplesN: Negatives
            :return: weights
            """
            return np.concatenate((np.ones((1,samplesP))*0.5/samplesP, np.ones((1,samplesN))*0.5/samplesN), axis=1)

        def get_labels(self, samplesP, samplesN):
            """
            Prepare and get the labels
            :param samplesP: Positives
            :param samplesN: Negatives
            :return: Labels
            """
            return np.concatenate((np.ones((1,samplesP)), np.zeros((1, samplesN))) , axis=1)

        def sort_WL(self, W, L, vector):
            """
            Sort the weights and labels
            :param W: Updated weights
            :param L: Labels
            :param vector: Concatenated positive and negative sample vector
            :return:Sorted values
            """
            sortW = np.tile(W, (len(vector),1))
            sortL = np.tile(L, (len(vector),1))
            return sortW, sortL

        def getSum(self, W, ps, sL, sW):
            """
            Get the sum for the sorted values
            :param W: weights
            :param ps: Positive samples
            :param sL: Sorted Labels
            :param sW: Sorted Weights
            """
            tnW = np.sum(W[:, ps:])
            tpW = np.sum(W[:,:ps])
            spW = np.cumsum(sW * sL, axis=1)
            snW = np.cumsum(sW, axis=1) - spW
            return spW, snW, tnW, tpW

        def get_error_terms(self, vector, spW, snW, tnW, tpW):
            """
            Get the required error terms.
            :return: Index of minimum error, minimum error value, error vector
            """
            error = np.zeros((vector.shape[0], vector.shape[1], 2))
            error[:, :, 1] = snW + tpW - spW
            error[:, :, 0] = spW + tnW - snW
            index = np.unravel_index(np.argmin(error), error.shape)
            errro_min = error[index]
            return index, errro_min, error

        class StrongC:
            """
            Strong classifier object
            """
            index = list()
            weak_number = 0
            classifierT = list()

        def getWeakC(self, vector, positive_samples, negative_samples):
            """
            Get the wek classifier which can extract low level image features
            :param vector: Concatenated positive and negative samples vector
            :param positive_samples: Positive samples
            :param negative_samples: Negative samples
            :return: Classifier object
            """
            cl = list()
            cl_T = list()
            alpha_value = list()
            W = self.get_weight(samplesP=positive_samples, samplesN=negative_samples)
            L = self.get_labels(samplesP=positive_samples, samplesN=negative_samples)
            obj = self.StrongC()
            for number in range(25):
                W = W/np.sum(W)
                sortW, sortL = self.sort_WL(W,L, vector)
                index = np.argsort(vector, axis=1)
                row_value = np.arange(len(vector)).reshape((-1,1))
                sortL = sortL[row_value, index]
                sortW = sortW[row_value, index]
                spW, snW, tnW, tpW = self.getSum(W=W, ps=positive_samples, sL = sortL, sW=sortW)
                index_e, errro_min, error = self.get_error_terms(vector=vector, spW=spW, snW = snW, tnW=tnW, tpW=tpW)
                f_value = index_e[0]
                index_S = index[f_value,:]
                pt = np.zeros((vector.shape[1],1))
                p_matrix = np.zeros((vector.shape[1],1))
                if index_e[2] == 0:
                    value_p = -1
                    pt[index_e[1]+1:] = 1
                else:
                    value_p = 1
                    pt[:index_e[1]+1] = 1
                p_matrix[index_S]= pt
                sortV = vector[f_value,:]
                sortV = sortV[index_S]
                if index_e[1]==0:
                    angle = sortV[0]-0.01
                elif index_e[1]==-1:
                    angle = sortV[-1]+0.01
                else:
                    angle = np.mean(sortV[index_e[1]-1:index_e[1]+1])
                beta_value = errro_min/(1-errro_min)
                alpha_value.append(np.log(1/beta_value))
                cl.append(p_matrix.transpose())
                cl_T.append([f_value, angle, value_p, np.log(1/beta_value)])
                W = W*(beta_value**(1-np.abs(L-p_matrix.transpose())))
                s_value = np.dot(np.asarray(cl).transpose(),np.asarray(alpha_value))
                angle_updated = np.min(s_value[:positive_samples])
                prediction_s = np.zeros(s_value.shape)
                prediction_s[s_value>=angle_updated]=1
                if (np.sum(prediction_s[positive_samples:])/negative_samples<0.5):
                    break
            index_new = list()
            index_new.extend(np.arange(positive_samples))
            wrong_negative_index = [positive_samples+x for x in range(negative_samples) if prediction_s[positive_samples+x]==1]
            index_new.extend(wrong_negative_index)
            obj.index = np.asarray(index_new)
            obj.weak_number = number+1
            obj.classifierT = cl_T
            return obj

    class AdaBoostTest:
        """
        Class to run AdaBoost Testing
        """
        def __init__(self, obj, feature_path, model_path):
            """
            Initialise the AdaBoost Tester object
            :param obj: Classifier object
            :param feature_path: Features object from training
            :param model_path: classifier object from training
            """
            self.parameter_dict = dict()
            self.object = obj
            self.feature_path = feature_path
            file = open(feature_path, 'rb')
            file_value = pickle.load(file)
            self.positive = file_value[0]
            self.negative = file_value[1]
            self.sampleP = self.positive
            self.sampleN = self.negative
            file.close()
            file = open(model_path, 'rb')
            self.model = pickle.load(file)
            file.close()

        def scheduler(self):
            """
            This function runs all the required functions in order
            to get the task done
            """
            self.process_data()
            self.commence_testing()

        def get_predicted_angle(self, angle):
            return 0.5*np.sum(angle)

        def get_vector(self):
            """
            Get the required vector to compute weights and parameters.
            :return: Vector
            """
            return np.concatenate((self.sampleP, self.sampleN), axis = 1)

        def get_f_value(self, classifier_T):
            return classifier_T[:,0].astype(int)

        def get_weight_pred(self, value, wpred ):
            weight_T = (value[1] * value[0])[:, None] - value[1][:, None] * value[2]
            wpred[weight_T >= 0] = 1
            return wpred

        def update_fp_fn(self, ftp_one, ftp_two, number_wrongP, number_rightP):
            ftp_one.append(number_wrongP / self.parameter_dict['Positives_WHL'])
            ftp_two.append(
                (self.parameter_dict['Negatives_WHL'] - number_rightP) / self.parameter_dict['Negatives_WHL'])
            return ftp_one,ftp_two

        def get_sample_lengths(self, spred):
            return [x for x in range(self.parameter_dict['Positives']) if spred[x]==1], [x for x in range(self.parameter_dict['Negatives']) if spred[x+self.parameter_dict['Positives']]==1]

        def get_angle_params(self, classifier_T):
            return classifier_T[:,1], classifier_T[:,2], classifier_T[:,3]

        def process_data(self):
            """
            Process the data before commencing testing into
            dictionary entries.
            """
            self.parameter_dict['Negatives'] = self.negative.shape[1]
            self.parameter_dict['Positives'] = self.positive.shape[1]
            self.parameter_dict['Positives_WHL'] = self.positive.shape[1]
            self.parameter_dict['Negatives_WHL'] = self.negative.shape[1]
            print('Positive Samples:')
            print(self.parameter_dict['Positives'])
            print('Negative Samples:')
            print(self.parameter_dict['Negatives'])

        def get_classifier_T(self, com_value):
            return np.asarray(self.model[com_value].classifierT)

        def commence_testing(self):
            """
            This function runs all the required items to execute the AdaBoost testing process.
            """
            number_wrongP, number_rightP = 0,0
            ftp_one, ftp_two = [], []
            for com in range(len(self.model)):
                print('Samples positive: ' +str(self.parameter_dict['Positives_WHL']))
                print('Samples negative: ' +str(self.parameter_dict['Negatives_WHL']))
                vector = self.get_vector()
                classifier_T = self.get_classifier_T(com_value=com)
                f_value = self.get_f_value(classifier_T=classifier_T)
                param_1, param_2, param_3 = self.get_angle_params(classifier_T=classifier_T)
                angle_predicted = self.get_predicted_angle(angle=param_3)
                wpred = np.zeros((len(classifier_T),vector.shape[1]))
                tempF = vector[f_value,:]
                wpred = self.get_weight_pred(value=(param_1,param_2,tempF), wpred=wpred)
                spred = np.zeros((vector.shape[1],1))
                tempS = np.dot(wpred.transpose(), param_3)
                spred[tempS>=angle_predicted]=1
                pcIdX, neIdX = self.get_sample_lengths(spred=spred)
                print('Right samples + '+ str(pcIdX))
                print('Error samples - '+str(neIdX))
                number_wrongP = number_wrongP + (self.parameter_dict['Positives']-len(pcIdX))
                number_rightP =number_rightP + (self.parameter_dict['Negatives']-len(neIdX))
                ftp_one, ftp_two = self.update_fp_fn(ftp_one=ftp_one, ftp_two=ftp_two, number_wrongP=number_wrongP, number_rightP=number_rightP)
                self.sampleP = self.sampleP[:, pcIdX]
                self.sampleN = self.sampleN[:, neIdX]
                self.parameter_dict['Positives'] = len(pcIdX)
                self.parameter_dict['Negatives'] = len(neIdX)
            list = [self.parameter_dict, ftp_one, ftp_two]
            file = open('AdaBoost_Testing_Results.obj', 'wb')
            pickle.dump(list, file)
            print('AdaBoost Testing complete')

    class AdaBoostTrain:
        """
        This class is for the Training of the AdaBoost Object Detection network
        """
        def __init__(self, obj, feature_path):
            """
            Initialise object
            :param obj: Classifier object
            :param feature_path: Path to the features extracted from the training set
            """
            self.parameter_dict = dict()
            self.object = obj
            self.feature_path = feature_path
            file = open(feature_path, 'rb')
            file_value = pickle.load(file)
            self.positive = file_value[0]
            self.negative = file_value[1]
            self.classifier = list()
            file.close()

        def scheduler(self):
            """
            This function runs all the required function in the correct order.
            """
            self.process_data()
            self.commence_training()

        def process_data(self):
            """
            Process and organise the data before training
            """
            self.parameter_dict['Positives'] = self.positive.shape[1]
            self.parameter_dict['Negatives'] = self.negative.shape[1]
            self.parameter_dict['Negatives_WHL'] = self.parameter_dict['Negatives']
            print('Initial Positive Samples:')
            print(self.parameter_dict['Positives'])
            print('Initial Negative Samples:')
            print(self.parameter_dict['Negatives'])

        def get_vector(self):
            """
            Get the required vector to compute weights and parameters.
            :return: Vector
            """
            return np.concatenate((self.positive, self.negative), axis = 1)

        def commence_training(self):
            """
            This function takes care of the AdaBoost Training process
            :return: Save the trained feature model
            """
            vector = self.get_vector()
            classifier_list = list()
            for com in range(8):
                wc = self.object.getWeakC(vector=vector, positive_samples=self.parameter_dict['Positives'],negative_samples=self.parameter_dict['Negatives'])
                classifier_list.append(wc)
                if (len(wc.index)==self.parameter_dict['Positives']):
                    break
                negatives = len(wc.index)-self.parameter_dict['Positives']
                self.parameter_dict['Negatives'] = negatives
                vector = vector[:,wc.index]
                val_to_apnd = self.parameter_dict['Negatives']/self.parameter_dict['Negatives_WHL']
                self.classifier.append(val_to_apnd)
            db = open('classifier.obj','wb')
            pickle.dump(classifier_list, db)
            print('Model saved. Training complete')

    class GetFeatures:
        """
        This class is to extract features from the testing and training directories
        """
        def __init__(self, data_path, type):
            """
            Initialise the object to extract the features
            :param data_path: Path to the directory
            :param type: Testing or Training to store them accordingly
            """
            self.filename = type
            self.feature_list = list()
            self.image_path = data_path
            self.positive_path = os.listdir(self.image_path[0])
            self.positive_path.sort()
            self.negative_path = os.listdir(self.image_path[1])
            self.negative_path.sort()
            self.reference_image_positive = cv.imread(self.image_path[0] + self.positive_path[0])
            self.reference_image_negative = cv.imread(self.image_path[1] + self.negative_path[0])
            self.image_vector_dict = list()
            self.image_vector_dict.append(np.zeros(
                (self.reference_image_positive.shape[0], self.reference_image_positive.shape[1], len(self.positive_path))))
            self.image_vector_dict.append(np.zeros(
                (self.reference_image_negative.shape[0], self.reference_image_negative.shape[1], len(self.negative_path))))
            self.paths = [self.positive_path, self.negative_path]
            self.ref_images = [self.reference_image_positive, self.reference_image_negative]
            self.get_images_ready()

        def scheduler(self):
            """
            This function runs all the required functions in order
            :return:
            """
            self.extract_features()

        def get_images_ready(self):
            """
            Organize the images
            """
            for index, path in enumerate(tqdm(self.paths, desc='Image Load')):
                for value in range(len(path)):
                    ref_img = cv.imread(self.image_path[index]+path[index])
                    self.image_vector_dict[index][:,:, index] = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)

        def set_filter_size(self, value):
            """
            Get filter size
            :param value: Value for the shape
            :return: Filter size
            """
            return (value+2)*2

        def get_cumulative_sum(self, image):
            value = np.cumsum(image, axis=1)
            return np.cumsum(image, axis=0)

        def get_sum_of_box(self, points, integral):
            """
            Box sum function
            :param points: Corner points
            :param integral: Integral image
            :return: Sum of the box
            """
            left_top = integral[np.int(points[0][0])][np.int(points[0][1])]
            right_top = integral[np.int(points[1][0])][np.int(points[1][1])]
            right_bottom = integral[np.int(points[2][0])][np.int(points[2][1])]
            left_bottom = integral[np.int(points[3][0])][np.int(points[3][1])]
            return left_bottom-right_top-right_bottom+left_top

        def get_integral_image(self):
            """
            Get the integral image
            :return: Integral image list
            """
            temp = list()
            for index in range(2):
                integral = self.get_cumulative_sum(image=self.image_vector_dict[index])
                integral = np.concatenate((np.zeros((self.ref_images[index].shape[0],1,len(self.paths[index]))),integral), axis=1)
                integral = np.concatenate((np.zeros((1,self.ref_images[index].shape[1]+1,len(self.paths[index]))),integral), axis=0)
                temp.append(integral)
            return temp

        def get_points(self, value, value_two, mask, type):
            """
            Get the required points for the box
            :param value: First value (row)
            :param value_two: Second value (Column)
            :param mask: Filter or mask size
            :param type: Type of points
            :return: Corner points
            """
            if type ==1:
                points = list()
                points.append([value, value_two])
                points.append([value, value_two + mask / 2])
                points.append([value + 1, value_two])
                points.append([value + 1, value_two + mask / 2])
                return points
            elif type ==2:
                points = list()
                points.append([value, value_two + mask / 2])
                points.append([value, value_two + mask])
                points.append([value+1, value_two + mask / 2])
                points.append([value+1, value_two + mask])
                return points
            elif type ==3:
                points = list()
                points.append([value, value_two])
                points.append([value, value_two + 2])
                points.append([value+mask/2, value_two])
                points.append([value+mask/2, value_two + 2])
                return points
            elif type ==4:
                points = list()
                points.append([value+mask/2, value_two])
                points.append([value+mask/2, value_two + 2])
                points.append([value+mask, value_two])
                points.append([value+mask, value_two + 2])
                return points

        def get_diff(self, tuple):
            return (tuple[1] - tuple[0]).reshape((1, -1))

        def add_feature(self, feature):
            feature = np.asarray(feature).reshape((len(feature),-1))
            self.feature_list.append(feature)

        def save_features(self, feature_list):
            assert len(feature_list) == 2
            db = open(self.filename, 'wb')
            pickle.dump(feature_list, db)
            print('feature list saved')

        def extract_features(self):
            integral_list = self.get_integral_image()
            for index in tqdm(range(2), desc='Feature Extraction'):
                temp_features = list()
                shape_one = self.ref_images[index].shape[1]
                shape_zero = self.ref_images[index].shape[0]
                for value_n in range(np.int(shape_one / 2)):
                    mask = self.set_filter_size(value=value_n)
                    criteria = [np.int(shape_one / 2), shape_zero, shape_one + 1 - mask, np.int(shape_zero / 2),
                                shape_zero + 1 - mask, shape_one + 1 - 2]
                    for value in range(criteria[1]):
                        for value_two in range(criteria[2]):
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=1)
                            first_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=2)
                            second_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            store_value = self.get_diff(tuple=(first_SB, second_SB))
                            temp_features.append(store_value)
                for value_n in range(criteria[3]):
                    mask = self.set_filter_size(value=value_n)
                    for value in range(criteria[4]):
                        for value_two in range(criteria[5]):
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=3)
                            first_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=4)
                            second_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            store_value = self.get_diff(tuple=(first_SB, second_SB))
                            temp_features.append(store_value)
                self.add_feature(feature=temp_features)
            self.save_features(feature_list=self.feature_list)


if __name__ == "__main__":
    """
    Code starts here
    """
    tester = ViolaJonesOD(['ECE661_2020_hw11_DB2/train/positive/','ECE661_2020_hw11_DB2/train/negative/','ECE661_2020_hw11_DB2/test/positive/','ECE661_2020_hw11_DB2/test/negative/'])
    tester.scheduler()
