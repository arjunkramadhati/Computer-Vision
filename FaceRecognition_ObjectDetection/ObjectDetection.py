"""
------------------------------------
------------------------------------
Computer Vision - Purdue University - Homework 11
------------------------------------
Face Recognition & Object Detection
------------------------------------
Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Dec 2, 2020
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
from scipy import spatial
from scipy.linalg import null_space
from scipy import optimize
import matplotlib.pyplot as plt


class ViolaJonesOD:
    def __init__(self, folder_locations):
        self.folders = folder_locations

    def scheduler(self):
        tester = self.GetFeatures(data_path=[self.folders[0], self.folders[1]],type='train.obj')
        tester.scheduler()
        tester = self.GetFeatures(data_path=[self.folders[2], self.folders[3]],type='test.obj')
        tester.scheduler()
        obj = self.getClassifier()
        trainadaboost = self.AdaBoostTrain(obj=obj,feature_path='train.obj')
        trainadaboost.scheduler()

    class getClassifier:
        class StrongC:
            index = list()
            weak_number = 0
            classifierT = list()

        def get_weight(self, samplesP, samplesN):
            return np.concatenate((np.ones((1,samplesP))*0.5/samplesP, np.ones((1,samplesN))*0.5/samplesN), axis=1)

        def get_labels(self, samplesP, samplesN):
            return np.concatenate((np.ones((1,samplesP)), np.zeros((1, samplesN))) , axis=1)

        def sort_WL(self, W, L, vector):
            sortW = np.tile(W, (len(vector),1))
            sortL = np.tile(L, (len(vector),1))
            return sortW, sortL

        def getSum(self, W, ps, ns, sL, sW):
            spW = np.cumsum(sW * sL, axis=1)
            snW = np.cumsum(sW, axis=1) - spW
            tnW = np.sum(W[:, ps:])
            tpW = np.sum(W[:,:ps])
            return spW, snW, tnW, tpW

        def get_error_terms(self, vector, spW, snW, tnW, tpW):
            error = np.zeros((vector.shape[0], vector.shape[1], 2))
            error[:, :, 1] = snW + tpW - spW
            error[:, :, 0] = spW + tnW - snW
            index = np.unravel_index(np.argmin(error), error.shape)
            errro_min = error[index]
            return index, errro_min, error

        def getWeakC(self, vector, positive_samples, negative_samples):
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
                spW, snW, tnW, tpW = self.getSum(W=W, ps=positive_samples, ns=negative_samples, sL = sortL, sW=sortW)
                index_e, errro_min, error = self.get_error_terms(vector=vector, spW=spW, snW = snW, tnW=tnW, tpW=tpW)
                f_value = index_e[0]
                index_S = index[f_value,:]
                pt = np.zeros((vector.shape[1],1))
                p_matrix = np.zeros((vector.shape[1],1))
                if index_e[2]==0:
                    p = -1
                    pt[index_e[1]+1:]=1
                else:
                    p = 1
                    pt[:index_e[1]+1]=1
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
                cl_T.append([f_value, angle, p, np.log(1/beta_value)])
                W = W*(beta_value**(1-np.abs(L-p_matrix.transpose())))
                s_value = np.dot(np.asarray(cl).transpose(),np.asarray(alpha_value))
                angle_updated = np.min(s_value[:positive_samples])
                prediction_s = np.zeros(s_value.shape)
                prediction_s[s_value>=angle_updated]=1
                print(np.sum(prediction_s[positive_samples:])/negative_samples)
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
        def __init__(self, obj, feature_path, model_path):
            self.parameter_dict = dict()
            self.object = obj
            self.feature_path = feature_path
            file = open(feature_path, 'rb')
            file_value = pickle.load(file)
            self.positive = file_value[0]
            self.negative = file_value[1]
            self.classifier = list()
            file.close()
            file = open(model_path, 'rb')
            model = pickle.load(file)
            file.close()

        def scheduler(self):
            pass

        def process_data(self):
            self.parameter_dict['Positives'] = self.positive.shape[1]
            self.parameter_dict['Negatives'] = self.negative.shape[1]
            self.parameter_dict['Negatives_WHL'] = self.parameter_dict['Negatives']
            print('Positive Samples:')
            print(self.parameter_dict['Positives'])
            print('Negative Samples:')
            print(self.parameter_dict['Negatives'])

    class AdaBoostTrain:
        def __init__(self, obj, feature_path):
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
            self.process_data()
            self.commence_training()

        def process_data(self):
            self.parameter_dict['Positives'] = self.positive.shape[1]
            self.parameter_dict['Negatives'] = self.negative.shape[1]
            self.parameter_dict['Negatives_WHL'] = self.parameter_dict['Negatives']
            print('Positive Samples:')
            print(self.parameter_dict['Positives'])
            print('Negative Samples:')
            print(self.parameter_dict['Negatives'])

        def get_vector(self):
            return np.concatenate((self.positive, self.negative), axis = 1)

        def commence_training(self):
            vector = self.get_vector()
            classifier_list = list()

            for com in range(10):
                print('Phase: '+ str(com))
                wc = self.object.getWeakC(vector=vector, positive_samples=self.parameter_dict['Positives'],negative_samples=self.parameter_dict['Negatives'])
                classifier_list.append(wc)
                if (len(wc.index)==self.parameter_dict['Positives']):
                    break
                self.parameter_dict['Negatives'] = len(wc.index)-self.parameter_dict['Positives']
                print('Negative samples: '+ str(self.parameter_dict['Negatives']))
                temp_vector = vector[:,wc.index]
                vector = temp_vector
                self.classifier.append(self.parameter_dict['Negatives']/self.parameter_dict['Negatives_WHL'])
            db = open('classifier.obj','wb')
            pickle.dump(self.classifier, db)
            print('Model saved. Training complete')


    class GetFeatures:
        def __init__(self, data_path, type):
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

        def get_images_ready(self):
            for index, path in enumerate(tqdm(self.paths, desc='Image Load')):
                for value in range(len(path)):
                    ref_img = cv.imread(self.image_path[index]+path[index])
                    self.image_vector_dict[index][:,:, index] = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)

        def scheduler(self):
            self.extract_features()

        def set_filter_size(self, value):
            return (value+1)*2

        def get_sum_of_box(self, points, integral):
            left_top = integral[np.int(points[0][0])][np.int(points[0][1])]
            right_top = integral[np.int(points[1][0])][np.int(points[1][1])]
            right_bottom = integral[np.int(points[2][0])][np.int(points[2][1])]
            left_bottom = integral[np.int(points[3][0])][np.int(points[3][1])]
            return left_bottom-right_top-right_bottom+left_top

        def get_integral_image(self):
            temp = list()
            for index in range(2):
                integral = np.cumsum(self.image_vector_dict[index], axis=1)
                integral = np.cumsum(integral, axis=0)
                integral = np.concatenate((np.zeros((self.ref_images[index].shape[0],1,len(self.paths[index]))),integral), axis=1)
                integral = np.concatenate((np.zeros((1,self.ref_images[index].shape[1]+1,len(self.paths[index]))),integral), axis=0)
                temp.append(integral)
            return temp

        def get_points(self, value, value_two, mask, type):
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
                for value_n in range(np.int(shape_one/2)):
                    mask = self.set_filter_size(value=value_n)
                    for value in range(shape_zero):
                        for value_two in range(shape_one+1-mask):
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=1)
                            first_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=2)
                            second_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            store_value = (second_SB-first_SB).reshape((1,-1))
                            temp_features.append(store_value)
                for value_n in range(np.int(shape_zero / 2)):
                    mask = self.set_filter_size(value=value_n)
                    for value in range(shape_zero + 1 - mask):
                        for value_two in range(shape_one + 1 - 2):
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=3)
                            first_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            points = self.get_points(value=value, value_two=value_two, mask=mask, type=4)
                            second_SB = self.get_sum_of_box(points=points, integral=integral_list[index])
                            store_value = (second_SB - first_SB).reshape((1, -1))
                            temp_features.append(store_value)
                self.add_feature(feature=temp_features)
            self.save_features(feature_list=self.feature_list)



if __name__ == "__main__":
    """
    Code starts here

    """
    tester = ViolaJonesOD(['ECE661_2020_hw11_DB2/train/positive/','ECE661_2020_hw11_DB2/train/negative/','ECE661_2020_hw11_DB2/test/positive/','ECE661_2020_hw11_DB2/test/negative/'])
    tester.scheduler()
    # tester = ViolaJonesOD.GetFeatures(['ECE661_2020_hw11_DB2/test/positive/','ECE661_2020_hw11_DB2/test/negative/'])
    # tester.scheduler()
    # tester = ViolaJonesOD.AdaBoostTest(feature_path='test_positive.obj')
    # tester.scheduler()
