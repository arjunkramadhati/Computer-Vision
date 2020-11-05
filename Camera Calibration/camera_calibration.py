"""
Computer Vision - Purdue University - Homework 9

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Nov 2, 2020
-----------------------------------

Reference : (Ref: https://engineering.purdue.edu/RVL/ECE661_2018/Homeworks/HW8/2BestSolutions/1.pdf)

-----------------------------------
[TO RUN CODE]: python3 camera_calibration.py
"""
import re
import glob
import os
import pickle
from tqdm import tqdm
import cv2 as cv
from PIL import Image,ImageFont,ImageDraw
from scipy.optimize import least_squares
import numpy as np
from pylab import *
import copy


class Calibrate:
    def __init__(self, image_path):
        """
        Initialization code
        :param image_path: Path to the images
        """
        print("Initializing Calibration process...")
        self.image_path = glob.glob(image_path)
        print("Loading image from path " + image_path)
        self.color_images_dict = dict()
        self.gray_images_dict = dict()
        self.lines_dict = dict()
        self.corner_size = (8,10)
        self.corner_list = []
        self.corner_list_filtered = []
        self.homographies = []
        self.cost_variable = []
        self.calibration_performance_raw = dict()
        self.calibration_performance_refined = dict()
        self.parameter_dict = dict()
        self.reference_image = Image.open('Files/Dataset1/Pic_11.jpg')
        self.draw = ImageDraw.Draw(self.reference_image)
        self.image_list_g = []
        self.image_list_c = []
        for image_index in range(len(os.listdir('Files/Dataset1/'))):
            imagepath = 'Files/Dataset1/Pic_'+str(image_index+1)+'.jpg'
            image = np.asarray(Image.open(imagepath))
            self.image_list_c.append(image)
            self.image_list_g.append(cv.cvtColor(image,cv.COLOR_BGR2GRAY))
        print(len(self.image_list_g))
        for index, element in enumerate(tqdm(self.image_path, ascii=True, desc='Image loading')):
            image = cv.imread(element)
            self.color_images_dict[index] = image
            self.gray_images_dict[index] = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print("Initialization complete")
        print("-------------------------------------")
        print("-------------------------------------")

    def calibrate_camera(self, run_config = 'Run'):
        """
        This function sequences all the required functions needed to calibrate the camera
        :return:
        """
        if run_config == 'Run':
            #Extract lines
            self.extract_lines()
            #Extract corners from lines
            self.extract_corners()
            #
            self.estimate_corner_homography()
            self.compute_parameter_w()
            self.estimate_extrinsic()
            self.estimate_raw_H()
            self.reproject_and_save()
            self.refine_calibration()
            self.reproject_and_save(Htype='Refined')
            self.save_results()
        elif run_config == 'Analyse':
            self.analyse_results()

    def save_results(self):
        print("Saving Results...")
        pickle.dump(self.parameter_dict,open("calibration_parameters.p","wb"))
        pickle.dump(self.calibration_performance_raw, open("performance_raw.p","wb"))
        pickle.dump(self.calibration_performance_refined,open("performance_refined.p","wb"))
        print("Results saved")

    def analyse_results(self):
        """
        Analyse the results of the calibration
        :return:
        """
        parameters = pickle.load(open("calibration_parameters.p","rb"))
        raw_performance = pickle.load(open("performance_raw.p","rb"))
        refined_performance = pickle.load(open("performance_refined.p","rb"))
        while True:
            print(parameters.keys())
            print("Enter Selection")
            selection = input('Selection: ')
            if selection == '0':
                break
            elif selection == '9':
                pass
            else:
                if selection =='R' or 'refinedRT':
                    print('Need index: ')
                    index = input()
                    print(parameters[selection][int(index)])
                else:
                    print(parameters[selection])
            print('Continue?')
            exit = input('0 for quit, 1 for continue')
            if exit == 0:
                break
            elif exit == 1:
                continue

        while True:
            print('Enter comparison image number')
            i = input()
            mean_raw = raw_performance[int(i)][0]
            var_raw =raw_performance[int(i)][1]
            mean_refined = refined_performance[int(i)][0]
            var_refined = refined_performance[int(i)][1]
            print("Image " + str(int(i)+1))
            print("Mean before LM : "+ str(mean_raw) + "||||| Mean after LM : " + str(mean_refined))
            print("Var before LM : " + str(var_raw) + "||||| Variance after LM : " + str(var_refined))
            print("Continue?")
            exit = input('0 for quit, 1 for continue')
            if exit == 0:
                quit()
            elif exit == 1:
                continue

    def get_line(self, rho, theta):
        """
        Get the coordinates of the end points of the lines
        :param rho: rho value
        :param theta: Thetha value
        :return: coordinates
        """
        proportion_one = np.cos(theta)
        proportion_two = np.sin(theta)
        centerX = rho*proportion_one
        centerY = rho*proportion_two
        p1 = int(centerX + 1000*(-proportion_two))
        p2 = int(centerY + 1000*(proportion_one))
        p3 = int(centerX - 1000*(-proportion_two))
        p4 = int(centerY - 1000*(proportion_one))
        return (p1,p2),(p3,p4)

    def extract_lines(self, cutoff = 50, output_path='Files/calibration_output/edges_lines/'):
        """
        Extract the Hough lines
        :param cutoff: Threshold
        :param output_path: Saving the output
        :return:
        """
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Edge & Line extraction'):
            color = (self.image_list_c[key].copy())/2
            edges = cv.Canny(cv.GaussianBlur(self.image_list_g[key],(5,5),0),2500, 4000, apertureSize=5)
            color[edges!=0] = (255,0,255)
            cv.imwrite('Files/calibration_output/edges/'+str(key)+'.jpg', color)
            hline = cv.HoughLines(edges,1, np.pi/180, cutoff)
            for line in hline:
                for rho, theta in line:
                    point_one, point_two = self.get_line(rho, theta)
                    cv.line(color, point_one, point_two, (0, 255, 0), 3)
            self.lines_dict[key] = hline
            cv.imwrite(output_path+str(key)+'.jpg', color)
        print("Line extraction complete...")
        print("-------------------------------------")

    def filter_lines(self, houghlines):
        final_hough_list = np.asarray(houghlines).copy()
        final_hlist_hesse = []
        final_vlist_hesse = []
        for index in range(len(final_hough_list)):
            individual_final_list = np.array(final_hough_list[index]).tolist()
            individual_final_list_h = individual_final_list[0:10]
            individual_final_list_h.sort(key=lambda item:item[0][0])
            final_hlist_hesse.append(individual_final_list_h)
            individual_final_list_v = individual_final_list[10:]
            individual_final_list_v.sort(key=lambda item:abs(item[0][0]))
            final_vlist_hesse.append(individual_final_list_v)
        return final_hlist_hesse, final_vlist_hesse

    def filter_list(self, hlist, vlist, distance_cutoffH = 100,distance_cutoffV = 100):
        """
        Filter the list of the hough line selections.
        :param hlist: Horizontal hough lines
        :param vlist: Vertical hough lines
        :param distance_cutoffH: Cutoff criteria
        :param distance_cutoffV: Cutoff criteria
        :return: Filtered list of horizontal and vertical lines
        """
        filtered_hlist = []
        filtered_vlist = []
        while(len(filtered_hlist)<10):
            distance_cutoffH -=0.05
            filtered_hlist = []
            for index in range(len(hlist)):
                selectedline = hlist[index][0][0]
                reject = 0
                for line in filtered_hlist:
                    if abs(abs(line[0][0])- abs(selectedline))<distance_cutoffH:
                        reject = 1
                if reject ==0:
                    filtered_hlist.append(hlist[index])

        while(len(filtered_vlist)<8):
            distance_cutoffV -=0.05
            filtered_vlist=[]
            for index in range(len(vlist)):
                selectedline = vlist[index][0][0]
                reject = 0
                for line in filtered_vlist:
                    if abs(abs(line[0][0]) -abs(selectedline)) < distance_cutoffV:
                        reject = 1
                if reject == 0:
                    filtered_vlist.append(vlist[index])

        return filtered_hlist, filtered_vlist

    def draw_filtered_lines(self, linelist, path='Files/calibration_output/final_lines/'):
        """
        Draw the final selected Hough Lines.
        :param linelist: List of all detected hough lines
        :param path: Path to save
        :return:
        """
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Drawing filtered lines and saving'):
            lines=linelist[key]
            image = copy.deepcopy(self.image_list_c[key])
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                point_one, point_two = self.get_line(rho, theta)
                cv.line(image, point_one, point_two, (0, 255, 0), 3)
            cv.imwrite(path+str(key)+'.jpg', image)

    def extract_corners(self):
        """
        Corner extraction algorithm. I referred the implementation from the link provided below.
        https://stackoverflow.com/a/383527/5087436
        :return:
        """
        linelist = []
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Line filtering'):
            horizontal_line_list = []
            vertical_line_list = []
            color = self.image_list_c[key].copy()
            lines = self.lines_dict[key]
            for line in lines:
                theta = line[0][1]
                if np.pi/4<theta<(np.pi*3)/4:
                    horizontal_line_list.append(line)
                else:
                    vertical_line_list.append(line)
            assert(len(horizontal_line_list)+len(vertical_line_list) == len(lines))
            horizontal_line_list, vertical_line_list = self.filter_list(horizontal_line_list, vertical_line_list)
            assert(len(horizontal_line_list) == 10)
            assert(len(vertical_line_list) == 8)
            linelist.append(horizontal_line_list+vertical_line_list)
        self.draw_filtered_lines(linelist)
        final_horizontal_lines, final_vertical_lines = self.filter_lines(linelist)
        corners = []
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Corner extraction'):
            individual_corners = []
            for index_vertical in range(len(final_vertical_lines[key])):
                for index_horizontal in range(len(final_horizontal_lines[key])):
                    rho_horizontal, theta_horizontal = final_horizontal_lines[key][index_horizontal][0]
                    rho_vertical, theta_vertical = final_vertical_lines[key][index_vertical][0]
                    A = np.array([
                        [np.cos(theta_vertical), np.sin(theta_vertical)],
                        [np.cos(theta_horizontal), np.sin(theta_horizontal)]
                    ])
                    B = np.array([[rho_vertical],[rho_horizontal]])
                    cornerX, cornerY = np.linalg.solve(A,B)
                    cornerX, cornerY = int(np.round(cornerX)), int(np.round(cornerY))
                    individual_corners.append([[cornerX,cornerY]])
            corners.append(individual_corners)
        corners_filtered = np.array(np.asarray(corners).copy()).tolist()
        self.enumerate_draw_corners(corners_filtered)
        self.corner_list = corners
        self.corner_list_filtered = corners_filtered

    def enumerate_draw_corners(self, corners, path ='Files/calibration_output/enumerated_corners/'):
        """
        Draw the numbers for each corner using the same numbering pattern.
        :param corners: List of corners
        :param path: Path to save the images.
        :return:
        """
        corners =np.array(np.asarray(corners).copy()).tolist()
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Enumerate coners'):
            image_path='Files/Dataset1/Pic_'+str(key+1)+'.jpg'
            image =Image.open(image_path)
            recreate_img = ImageDraw.Draw(image)
            for corner_index in range(len(corners[key])):
                recreate_img.text((corners[key][corner_index][0][0],corners[key][corner_index][0][1]),str(corner_index),(255,0,0))
            image.save(path+str(key)+'.jpg')

    def estimate_extrinsic(self):
        """
        Estimates the extrinsic parameters
        :return: Stores in the dictionary
        """
        omega = self.parameter_dict['omega']
        centerX = ((omega[0][1]*omega[0][2])-(omega[0][0]*omega[1][2]))/((omega[0][0]*omega[1][1])-(omega[0][1]*omega[0][1]))
        lambdavalue = omega[2][2] - (((omega[0][2]*omega[0][2])+centerX*((omega[0][1]*omega[0][2])-(omega[0][0]*omega[1][2])))/omega[0][0])
        a_x,a_y = abs(np.sqrt(lambdavalue/omega[0][0])),abs(np.sqrt((lambdavalue*omega[0][0])/abs((omega[0][0]*omega[1][1])-(omega[0][1]*omega[0][1]))))
        svalue = -1*((omega[0][1]*a_x*a_x*a_y)/(lambdavalue))
        centerY = ((svalue*centerX)/a_y)-((omega[0][2]*a_x*a_x)/lambdavalue)
        K = np.zeros((3,3))
        K[0][0] = a_x
        K[0][1] = svalue
        K[0][2] = centerX
        K[1][0] = 0.0
        K[1][1] = a_y
        K[1][2] = centerY
        K[2][0] = 0.0
        K[2][1] = 0.0
        K[2][2] = 1.0
        self.parameter_dict['K'] = K
        self.parameter_dict['a_x'] = a_x
        self.parameter_dict['a_y'] = a_y
        self.parameter_dict['svalue'] = svalue
        self.parameter_dict['centerX'] = centerX
        self.parameter_dict['centerY'] = centerY

        matrixR =[]
        for key in tqdm(range(len(self.homographies)), ascii=True, desc='Extrinsic estimation'):
            evalue = 1/np.linalg.norm(np.matmul(np.linalg.pinv(K), self.homographies[key][: , 0]))
            firstR = evalue*np.matmul(np.linalg.pinv(K), self.homographies[key][: , 0])
            secondR = evalue*np.matmul(np.linalg.pinv(K) ,self.homographies[key][: ,1])
            thirdR = np.cross(firstR, secondR)
            matrixZ = self.condition_rotation_matrix([firstR,secondR,thirdR])
            firstR, secondR, thirdR = matrixZ[:,0],matrixZ[:,1], matrixZ[:,2]
            tvalue = evalue*np.matmul(np.linalg.pinv(K) ,self.homographies[key][:,2])
            rotationmatrix = np.zeros((3,4))
            rotationmatrix[:,0] = firstR
            rotationmatrix[:,1] = secondR
            rotationmatrix[:,2] = thirdR
            rotationmatrix[:,3] = tvalue
            matrixR.append(rotationmatrix)
        self.parameter_dict['R'] = matrixR

    def condition_rotation_matrix(self, rvalues):
        """
        Condition the rotation matrix
        :param rvalues: R matrix required to condition
        :return: Conditioned matrix
        """
        matrixQ = np.zeros((3,3))
        matrixQ[:,0] = rvalues[0]
        matrixQ[:,1] = rvalues[1]
        matrixQ[:,2] = rvalues[2]
        uvalue, dvalue, vvalueT = np.linalg.svd(matrixQ)
        matrixZ = np.matmul(uvalue,vvalueT)
        return matrixZ

    def get_omega_matrix(self, matrixb):
        matrix_omega = np.zeros((3, 3))
        matrix_omega[0][0] = matrixb[0]
        matrix_omega[0][1] = matrixb[1]
        matrix_omega[0][2] = matrixb[3]
        matrix_omega[1][0] = matrixb[1]
        matrix_omega[1][1] = matrixb[2]
        matrix_omega[1][2] = matrixb[4]
        matrix_omega[2][0] = matrixb[3]
        matrix_omega[2][1] = matrixb[4]
        matrix_omega[2][2] = matrixb[5]
        return matrix_omega

    def compute_parameter_w(self):
        """
        Computes the omega parameter of the camera.
        :return: Stores the omega value in the dictionary
        """
        matrixV = np.zeros((2*len(self.homographies),6))
        for key in tqdm(range(len(self.homographies)), ascii=True, desc='Omega estimation'):
            homography = np.transpose(self.homographies[key])
            templist = []
            for item in [(0,1),(0,0),(1,1)]:
                vmatrix = np.zeros((1, 6))
                vmatrix[0][0] = homography[item[0]][0] * homography[item[1]][0]
                vmatrix[0][1] = (homography[item[0]][0] * homography[item[1]][1])+(homography[item[0]][1] * homography[item[1]][0])
                vmatrix[0][2] = homography[item[0]][1] * homography[item[1]][1]
                vmatrix[0][3] = (homography[item[0]][2] * homography[item[1]][0])+(homography[item[0]][0] * homography[item[1]][2])
                vmatrix[0][4] = (homography[item[0]][2] * homography[item[1]][1]) + (
                            homography[item[0]][1] * homography[item[1]][2])
                vmatrix[0][5] = homography[item[0]][2] * homography[item[1]][2]
                templist.append(vmatrix)
            first_vmatrix = templist[0][0]
            second_vmatrix = (templist[1] - templist[2])[0]
            matrixV[2*key] = first_vmatrix
            matrixV[2*key+1] = second_vmatrix
        umatrix, dmatrix, vmatrixT = np.linalg.svd(matrixV)
        matrixB = np.transpose(vmatrixT)[:,-1]
        omega = self.get_omega_matrix(matrixB)
        self.parameter_dict['omega'] = omega
        self.parameter_dict['matrixV'] = matrixV

    def get_refined_omega(self,x,y,z):
        omegamatrix_x = np.zeros((3, 3))
        omegamatrix_x[0][0] = 0.0
        omegamatrix_x[1][1] = 0.0
        omegamatrix_x[2][2] = 0.0
        omegamatrix_x[0][1] = -1*z
        omegamatrix_x[0][2] = y
        omegamatrix_x[1][0] = z
        omegamatrix_x[1][2] = -1*x
        omegamatrix_x[2][0] = -1*y
        omegamatrix_x[2][1] = x
        return omegamatrix_x

    def set_temp_matrix(self, parameter, point):
        temp_estimate = np.matmul(parameter, np.asarray(point))
        return temp_estimate/temp_estimate[2]

    def set_gamma(self, temp_estimate, center):
        return np.sqrt(np.square(temp_estimate[0]-center[0])+np.square(temp_estimate[1]-center[1]))

    def calibration_cost(self, point):
        """
        Cost function for LM refinement
        :return: Cost vector
        """
        resid = []
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='LM Refine'):
            omega_x = point[6*key+5]
            omega_y = point[6*key+1+5]
            omega_z = point[6*key+2+5]
            first_t = point[6*key+3+5]
            second_t = point[6*key+4+5]
            third_t = point[6*key+5+5]
            a_x = point[0]
            svalue = point[1]
            centerX = point[2]
            a_y = point[3]
            centerY = point[4]
            omegamatrix = np.zeros((3,1))
            omegamatrix[0] = omega_x
            omegamatrix[1] = omega_y
            omegamatrix[2] = omega_z
            phivalue = np.linalg.norm(omegamatrix)
            omegamatrix_x = self.get_refined_omega(omega_x,omega_y,omega_z)
            matrixT = np.zeros((3))
            matrixT[0]=first_t
            matrixT[1]=second_t
            matrixT[2]=third_t
            matrix_first_R = np.zeros((3,3))
            second_R = (np.sin(phivalue) / phivalue) * omegamatrix_x
            third_R = ((1-np.cos(phivalue))/(phivalue*phivalue))*np.matmul(omegamatrix_x,omegamatrix_x)
            matrix_first_R[0][0] = 1.0
            matrix_first_R[1][1] = 1.0
            matrix_first_R[2][2] = 1.0
            final_R = matrix_first_R+second_R+third_R
            matrixK = np.zeros((3,3))
            matrixK[0][0] = a_x
            matrixK[2][2] = 1.0
            matrixK[0][1] = svalue
            matrixK[0][2] = centerX
            matrixK[1][1] = a_y
            matrixK[1][2] = centerY
            rotation_matrix = np.zeros((3,3))
            rotation_matrix[:,0]=final_R[:,0]
            rotation_matrix[:,1]=final_R[:,1]
            rotation_matrix[:,2]=matrixT
            for imagecorner_index in range(len(self.corner_list_filtered[key])):
                point_estimate = []
                x_est = (imagecorner_index/10)*2.5
                y_est = (imagecorner_index%10)*2.5
                point_coordinate = np.array(np.asarray(self.corner_list_filtered[key][imagecorner_index][0]).copy()).tolist()
                point_coordinate.append(1.0)
                point_estimate.append(x_est)
                point_estimate.append(y_est)
                point_estimate.append(1.0)
                camera_parameter = np.matmul(matrixK,rotation_matrix)
                temp_estimate = self.set_temp_matrix(camera_parameter, point_estimate)
                gammavalue = self.set_gamma(temp_estimate, (centerX,centerY))
                final_estimate = (np.asarray(point_coordinate)-temp_estimate)
                resid.append(final_estimate[0])
                resid.append(final_estimate[1])
                resid.append(final_estimate[2])
        return resid

    def refine_calibration(self):
        """
        Refine the calibration parameters of the camera.
        :return:
        """
        matrixR = list(np.asarray(self.parameter_dict['R']))
        self.cost_variable.append(self.parameter_dict['a_x'])
        self.cost_variable.append(self.parameter_dict['svalue'])
        self.cost_variable.append(self.parameter_dict['centerX'])
        self.cost_variable.append(self.parameter_dict['a_y'])
        self.cost_variable.append(self.parameter_dict['centerY'])
        for homography_index in range(len(self.homographies)):
            trace_value =(np.trace(matrixR[homography_index][:,0:3])-1)/2
            if trace_value>1.0:
                trace_value=1.0
            phivalue = np.arccos(trace_value)
            if phivalue==0:
                phivalue=1
            self.cost_variable.append((matrixR[homography_index][2][1]-matrixR[homography_index][1][2])*(phivalue/(2*np.sin(phivalue))))
            self.cost_variable.append((matrixR[homography_index][0][2] - matrixR[homography_index][2][0]) * (
                        phivalue / (2 * np.sin(phivalue))))
            self.cost_variable.append((matrixR[homography_index][1][0] - matrixR[homography_index][0][1]) * (
                        phivalue / (2 * np.sin(phivalue))))
            self.cost_variable.append(matrixR[homography_index][0][3])
            self.cost_variable.append(matrixR[homography_index][1][3])
            self.cost_variable.append(matrixR[homography_index][2][3])
        optimised_R = least_squares(self.calibration_cost, self.cost_variable, method='lm',max_nfev=800)
        self.estimate_refined_H(optimised_R)

    def reproject_and_save(self, Htype = 'Raw'):
        """
        Reproject and save the images.
        :param Htype:Type of homography used. Raw is for the normal homography and refined is for the
        optimised homography value.
        :return:
        """
        if Htype == 'Raw':
            for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Reprojection Raw'):
                if key == 10:
                    pass
                else:
                    homography = np.matmul(self.parameter_dict['rawH'][10], np.linalg.pinv(self.parameter_dict['rawH'][key-1]))
                    projection = []
                    self.reference_image = Image.open('Files/Dataset1/Pic_11.jpg')
                    self.draw = ImageDraw.Draw(self.reference_image)
                    for index in range(len(self.corner_list[10])):
                        coordinates = list(np.asarray(self.corner_list[key-1][index][0]).copy())
                        coordinates.append(1.0)
                        projectedpoint = np.matmul(homography, np.asarray(coordinates))
                        projectedpoint = projectedpoint/projectedpoint[2]
                        projection.append(projectedpoint[:-1])
                    distance = []
                    for corner_index in range(len(self.corner_list[10])):
                        self.draw.text(( self.corner_list[10][corner_index][0][0] , self.corner_list[10][corner_index][0][1]) ,"*" ,(255,0,0))
                        self.draw.text(( list(projection[corner_index]) [ 0 ] , list( projection[corner_index])[ 1 ] ) , " *" , ( 255 , 255 , 0 ))
                        distance.append(np.linalg.norm(np.asarray(self.corner_list[10][corner_index][0])-projection[corner_index]))
                    self.reference_image.save('Files/calibration_output/reprojection_raw/'+str(key)+'.jpg')
                    self.calibration_performance_raw[key] = (np.mean(distance),np.var(distance))
        elif Htype == 'Refined':
            for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Reprojection Refined'):
                if key == 10:
                    pass
                else:
                    homography = np.matmul(self.parameter_dict['refined_homography'][10], np.linalg.pinv(self.parameter_dict['rawH'][key-1]))
                    projection = []
                    self.reference_image = Image.open('Files/Dataset1/Pic_11.jpg')
                    self.draw = ImageDraw.Draw(self.reference_image)
                    for index in range(len(self.corner_list[10])):
                        coordinates = list(np.asarray(self.corner_list[key-1][index][0]).copy())
                        coordinates.append(1.0)
                        projectedpoint = np.matmul(homography, np.asarray(coordinates))
                        projectedpoint = projectedpoint/projectedpoint[2]
                        projection.append(projectedpoint[:-1])
                    distance = []
                    for corner_index in range(len(self.corner_list[10])):
                        self.draw.text(( self.corner_list[10][corner_index][0][0] , self.corner_list[10][corner_index][0][1]) ,"*" ,(255,0,0))
                        self.draw.text(( list(projection[corner_index])[0] , list( projection[corner_index])[ 1 ] ) , " *" , ( 255 , 255 , 0 ))
                        distance.append(np.linalg.norm(np.asarray(self.corner_list[10][corner_index][0])-projection[corner_index]))
                    self.reference_image.save('Files/calibration_output/reprojection_refined/'+str(key)+'.jpg')
                    self.calibration_performance_refined[key] = (np.mean(distance),np.var(distance))

    def estimate_refined_H(self, refined_R):
        """
        Estimate the refined homography needed to reproject the points
        :param refined_R: Refined R matrix we got by running the LM least squares algorithm
        :return:
        """
        homography = []
        templist = []
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Refined Homography'):
            matrixK = np.zeros((3,3))
            matrixK[0][0] = refined_R.x[0]
            matrixK[0][1] = refined_R.x[1]
            matrixK[0][2] = refined_R.x[2]
            matrixK[1][1] = refined_R.x[3]
            matrixK[1][2] = refined_R.x[4]
            matrixK[2][2] = 1.0
            matrixW = np.zeros((3,1))
            matrixW_x = np.zeros((3,3))
            matrixR = np.zeros((3,3))
            matrixT = np.zeros((3))
            firstr = np.zeros((3,3))
            rotationmatrix = np.zeros((3,3))
            omega_x,omega_y,omega_z =refined_R.x[5+6*key],refined_R.x[5+1+6*key], refined_R.x[5+2+6*key]
            matrixW[0] = omega_x
            matrixW[1] = omega_y
            matrixW[2] = omega_z
            phivalue = np.linalg.norm(matrixW)
            matrixW_x[0][1] = -1*omega_z
            matrixW_x[0][2] = omega_y
            matrixW_x[1][0] = omega_z
            matrixW_x[1][2] = -1*omega_x
            matrixW_x[2][0] = -1*omega_y
            matrixW_x[2][1] = omega_x
            firstr[0][0] = 1.0
            firstr[1][1] =1.0
            firstr[2][2] = 1.0
            secondr = (np.sin(phivalue)/phivalue)*matrixW_x
            thirdr = ((1-np.cos(phivalue))/(phivalue*phivalue))*np.matmul(matrixW_x,matrixW_x)
            matrixR = firstr+secondr+thirdr
            matrixT[0] = refined_R.x[5+3+6*key]
            matrixT[1] = refined_R.x[5+4+6*key]
            matrixT[2] = refined_R.x[5+5+6*key]
            rotationmatrix[:,0]=matrixR[:,0]
            rotationmatrix[:,1] = matrixR[:,1]
            rotationmatrix[:,2] = matrixT
            homography.append(np.matmul(matrixK,rotationmatrix))
            templist.append(rotationmatrix)
            self.parameter_dict['refinedK'] = matrixK
        self.parameter_dict['refinedRT'] = templist
        self.parameter_dict['refined_homography'] = homography

    def estimate_raw_H(self):
        """
        Estimate the homography needed for the reprojection of the points.
        :return:
        """
        matrixR = np.asarray(self.parameter_dict['R'])
        raw_homographies = []
        K = self.parameter_dict['K']
        for key in tqdm(range(len(self.image_list_g)), ascii=True, desc='Raw matrix estimation'):
            raw_homographies.append(np.matmul(K,matrixR[key][:,[0,1,3]]))
        self.parameter_dict['rawH'] = raw_homographies

    def estimate_corner_homography(self):
        """
        Estimate the homography which maps the corners and their world coordinates
        :return:
        """
        H = []
        for key in tqdm(range(len(self.corner_list)), ascii=True, desc='Homography estimation'):
            matrixA = np.zeros((2*len(self.corner_list[key]), 9))
            for corner_index in range(len(self.corner_list[key])):
                matrixA[2 * corner_index + 0][0] = (corner_index/10)*2.5
                matrixA[2 * corner_index + 0][1] = (corner_index%10)*2.5
                matrixA[2 * corner_index + 0][2] = 1.0
                matrixA[2 * corner_index + 0][3] = 0.0
                matrixA[2 * corner_index + 0][4] = 0.0
                matrixA[2 * corner_index + 0][5] = 0.0
                matrixA[2 * corner_index + 0][6] = -1*((corner_index/10)*2.5)*self.corner_list[key][corner_index][0][0]
                matrixA[2 * corner_index + 0][7] = -1*((corner_index%10)*2.5)*self.corner_list[key][corner_index][0][0]
                matrixA[2 * corner_index + 0][8] = -1*self.corner_list[key][corner_index][0][0]
                matrixA[2 * corner_index + 1][0] = 0.0
                matrixA[2 * corner_index + 1][1] = 0.0
                matrixA[2 * corner_index + 1][2] = 0.0
                matrixA[2 * corner_index + 1][3] = (corner_index/10)*2.5
                matrixA[2 * corner_index + 1][4] = (corner_index%10)*2.5
                matrixA[2 * corner_index + 1][5] = 1.0
                matrixA[2 * corner_index + 1][6] = -1*((corner_index/10)*2.5)*self.corner_list[key][corner_index][0][1]
                matrixA[2 * corner_index + 1][7] = -1 * ((corner_index % 10) * 2.5) * self.corner_list[key][corner_index][0][1]
                matrixA[2 * corner_index + 1][8] = -1*self.corner_list[key][corner_index][0][1]
            homography = np.zeros((3,3))
            umatrix, dmatrix, vmatrixT = np.linalg.svd(matrixA)
            H_matrix = np.transpose(vmatrixT)[:,-1]
            if H_matrix[8] == 0 or H_matrix[8] ==NaN:
                print("True divide conflict. Ignoring value...")
            else:
                H_matrix = H_matrix/H_matrix[8]
            homography[0][0] = H_matrix[0]
            homography[0][1] = H_matrix[1]
            homography[0][2] = H_matrix[2]
            homography[1][0] = H_matrix[3]
            homography[1][1] = H_matrix[4]
            homography[1][2] = H_matrix[5]
            homography[2][0] = H_matrix[6]
            homography[2][1] = H_matrix[7]
            homography[2][2] = 1.0
            H.append(homography)
        self.homographies = H


if __name__ == "__main__":
    """
    Program starts here.
    """
    tester = Calibrate('./Files/Dataset1/*')
    tester.calibrate_camera()




