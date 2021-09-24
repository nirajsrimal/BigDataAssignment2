from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_MobileNetV1
from PIL import Image
from tog.attacks import *
import os
import time
import random
import xml.etree.ElementTree as ET

K.clear_session()
images_per_file = 5
images_input_file = '100Input.txt' #'images_class_input.txt'
total_samples = 100
weights = 'model_weights/YOLOv3_MobileNetV1.h5'  # TODO: Change this path to the victim model's weights
detector = YOLOv3_MobileNetV1(weights=weights)

def getOriginalClassesfromXMLfile(xmlfile): #Get the Original Classes for the image using the XML File
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    classes = []
    for item in root.findall('./object'):
        news = {}
        for child in item:
  
            if child.tag == 'name':
                classes.append(child.text)
    classes.sort()
    return classes

def getImages():	#Generate File containing image samples (use images_per_file to select n images per class)
	dirpath = './VOCdevkit/VOC2012/ImageSets/Main/';
	imgpath = './VOCdevkit/VOC2012/JPEGImages';
	xmlpath = './VOCdevkit/VOC2012/Annotations/'
	filenames = os.listdir(dirpath)
	dataFile = open("100_Samples.txt", "w")
	for fname in filenames:
		if("_divergence.txt" in fname):
			original_class = fname.split('_')[0]
			filename = dirpath+fname
			myfile = open(filename, "r");
			head = []
			for aline in myfile:
				if(" 1" in aline):
					image_file = aline.split()[0]
					head.append(image_file) 
			#new_list = random.sample(head, 5)
			#original_class = fname.rsplit( "_", 1 )[ 0 ] #Creating a randomized list of the samples
			count = 0
			for i in head:    	
				fpath = imgpath +'/'+ i +'.jpg'
				xmlfile = xmlpath + i + '.xml'
				if(len(getOriginalClassesfromXMLfile(xmlfile)) == 1):
					dataFile.write(original_class+' '+fpath+"\n")
					count = count + 1
				if(count == images_per_file):
					break
	dataFile.close()

def run_model(fpath): #Run the model with no attack
	input_img = Image.open(fpath)
	x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
	time_start = time.time()
	detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
	dur = (time.time() - time_start)
	visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes)})
	result_class = []
	for box in detections_query:
		result_class.append(detector.classes[(int)(box[0])])
	result_class.sort()
	return dur, result_class


def run_model_with_attack(fpath):	#Run the model with an attack
	input_img = Image.open(fpath)
	x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
	time_start = time.time()
	detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
	dur = (time.time() - time_start)
	#visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes)})
	#print(detector.classes[(int)(detections_query[0][0])],"\n")
	
	eps = 8 / 255
	eps_iter = 2 / 255
	n_iter = 10
	time_start = time.time()
	# Generation of the adversarial example
	x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
	detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
	visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-vanishing': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes)})

	dur += (time.time() - time_start)	
	result_class = []
	for box in detections_adv_mislabeling_ml:
		result_class.append(detector.classes[(int)(box[0])])
	result_class.sort()
	return dur, result_class

def run_model_onfile():
	with open(images_input_file) as myfile:
		total_time = 0
		pos = 0 
		j = 1
		head = [next(myfile) for x in range(total_samples)]
		for i in head:  	
			img_src = i.split()[1]
			xml_file = './VOCdevkit/VOC2012/Annotations/' + ((img_src.split('/')[-1]).split('.')[0]) + '.xml'
			#print(xml_file)
			original_class = getOriginalClassesfromXMLfile(xml_file)
			original_class.sort()
			model_ret = run_model_with_attack(img_src)
			result_class = model_ret[1]
			time_taken =  model_ret[0]
			#print(original_class, " = ", result_class, "?\n")
			print("Original class : ", original_class)
			print("Prediction class : ", result_class)
			if(original_class == result_class):
				pos = pos + 1;
				print("Sample ",j," : Predicted Successfully, Time taken: ", time_taken)
			else:
				print("Sample ",j," : Predicted Incorrectly, Time taken: ", time_taken)
			j = j+1
			total_time += time_taken
			accuracy = pos / total_samples
		print("\nAccuracy : ", pos, "% , Time taken per Sample: ", total_time/total_samples)


def run_model_onImagesFile_Attack():
	imgpath = './Images/';
	filenames = os.listdir(imgpath)
	total_time = 0
	pos = 0 
	for fname in filenames:
		img_name= fname.split('.')[0]
		print(fname,"\n")
		model_ret = run_model_with_attack(imgpath+fname)
		original_class = img_name[:-1]
		result_class = model_ret[1]
		time_taken =  model_ret[0]
		if(original_class == result_class):
			pos = pos + 1;
		total_time += time_taken
		accuracy = pos / total_samples
	print("\nModel Accuracy : ", pos/total_samples * 100, " , Time taken per Sample: ", total_time/total_samples)
	print("\nAttack Accuracy :", (total_samples - pos)/total_samples * 100, "%");


def run_model_on_image():
	fpath = imagename
	#run_model(fpath)
	run_model_with_attack(fpath)

#imagename = './VOCdevkit/VOC2012/JPEGImages/2008_001203.jpg' #./VOCdevkit/VOC2012/JPEGImages/2008_000703.jpg' #1307 #1691 instead of 1203
#run_model_on_image() Run the test on a specific image
#run_model_onImagesFile_Attack() Run the attack on the Images folder
#getImages() Creates a file with the Images from the Dataset equally from all 20 classes
#run_model_onfile()	Runs the model on samples present in the input_images_file