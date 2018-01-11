import argparse
import numpy
import skimage
import skimage.io
import skimage.transform
import csv
import os
import os.path
import sys
sys.path.insert(0, '~/Application/Caffe/caffe-master/python')
import caffe
import time

def main():
	# parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument('-iPath', type = str, default = 'None', help = 'input directory containing testing images')
	parser.add_argument('-oPath', type = str, default = './outputs/', help = 'output directory containing csv and html files of defect scores')
	parser.add_argument('-holisticDeployPath', type = str, default = 'None', help = 'holistic model deploy prototxt file')
	parser.add_argument('-holisticWeightsPath', type = str, default = 'None', help = 'holistic model pre-trained weights caffemodel or caffemodel.h5 file')
	parser.add_argument('-patchDeployPath', type = str, default = 'None', help = 'patch model deploy prototxt file')
	parser.add_argument('-patchWeightsPath', type = str, default = 'None', help = 'patch model pre-trained weights caffemodel or caffemodel.h5 file')
	parser.add_argument('-gpu', type = int, default = 0, help = 'assign a single gpu index')
	args = parser.parse_args()
	iPath = args.iPath
	if not os.path.isdir(iPath):
		print 'Input directory does not exist!'
		return
	holisticDeployPath = args.holisticDeployPath
	if not os.path.isfile(holisticDeployPath):
		print 'Holistic model deploy file does not exist!'
		return
	holisticWeightsPath = args.holisticWeightsPath
	if not os.path.isfile(holisticWeightsPath):
		print 'Holistic model pre-trained weights file does not exist!'
		return
	patchDeployPath = args.patchDeployPath
	if not os.path.isfile(patchDeployPath):
		print 'Patch model deploy file does not exist!'
		return
	patchWeightsPath = args.patchWeightsPath
	if not os.path.isfile(patchWeightsPath):
		print 'Patch model pre-trained weights file does not exist!'
		return
	oPath = args.oPath
	if not os.path.isdir(oPath):
		os.makedirs(oPath)
		os.chmod(oPath, 0o777)
	gpu = args.gpu

	# global variables
	defect_list = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur', 'Bad Composition']
	defect_layer = ['softmax_badExposure', 'softmax_badWhiteBalance', 'softmax_badSaturation', 'softmax_noise', 'softmax_haze', 'softmax_undesiredBlur', 'softmax_badComposition']
	batch_size = 1
	new_side_length = 224
	peak = numpy.arange(0.0, 1.1, 0.1)
	peak = numpy.reshape(peak, (1,-1))
	peak_saturation = numpy.arange(-1.0, 1.1, 0.1)
	peak_saturation = numpy.reshape(peak_saturation, (1,-1))

	num_images_per_row_vis = 8
	col_width = 224

	# model setup
	caffe.set_device(gpu)
	caffe.set_mode_gpu()
	net_holistic = caffe.Net(holisticDeployPath, holisticWeightsPath, caffe.TEST)
	net_holistic.blobs['data'].reshape(batch_size, # batch size
                          	  3, # 3-channel (BGR) images
                          	  new_side_length, new_side_length) # image size is 224x224
	net_patch = caffe.Net(patchDeployPath, patchWeightsPath, caffe.TEST)
	net_patch.blobs['data'].reshape(batch_size*9, # batch size
                          	  3, # 3-channel (BGR) images
                          	  new_side_length, new_side_length) # image size is 224x224
	mu = numpy.array([97.3598806324274, 104.61048961074, 109.466934369976]) # BGR mean over warped images
	transformer = caffe.io.Transformer({'data': net_holistic.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # move image channels to outermost dimension
	transformer.set_mean('data', mu) # BGR
	transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR

	start_time = time.time()
	# test the defect scores by looping over all the images in the input directory
	path_list = []
	batch_holistic = numpy.zeros((batch_size, 3, new_side_length, new_side_length))
	count_holistic = 0
	defect_score_holistic = {}
	for layer in defect_layer:
		defect_score_holistic[layer] = []
	batch_patch = numpy.zeros((batch_size*9, 3, new_side_length, new_side_length))
	count_patch = 0
	defect_score_patch = {}
	for layer in defect_layer[:-1]:
		defect_score_patch[layer] = []
	count = 0
	for name in os.listdir(iPath):
		if name[0] != '.' and name[-4:] == '.jpg' or name[-4:] == '.png':
			count += 1
			print '%d/%d images finished' % (count, len(os.listdir(iPath)))
			if iPath[-1] == '/':
				path = iPath + name
			else:
				path = iPath + '/' + name
			path_list.append(path)
			image = caffe.io.load_image(path)
			if len(image.shape) == 1:
				image = numpy.dstack((image, image, image))

			# holistic testing
			warped_image = skimage.transform.resize(image, (new_side_length, new_side_length), mode = 'reflect')
			transformed_image = transformer.preprocess('data', warped_image)
			batch_holistic[count_holistic,:,:,:] = transformed_image
			count_holistic += 1
			if count_holistic == batch_size:
				net_holistic.blobs['data'].data[...] = batch_holistic
				output = net_holistic.forward()
				for layer in defect_layer:
					prob = numpy.array(numpy.squeeze(output[layer]))
					if layer == 'softmax_badSaturation':
						my_peak = numpy.array(peak_saturation)
					else:
						my_peak = numpy.array(peak)
					my_peak = numpy.tile(my_peak, (batch_size,1))
					score = numpy.sum(prob*my_peak, 1)
					defect_score_holistic[layer] += list(score)
				batch_holistic = numpy.zeros((batch_size, 3, new_side_length, new_side_length))
				count_holistic = 0

			# patch testing
			height = image.shape[0]
			width = image.shape[1]
			if height < new_side_length or width < new_side_length:
				if height < width:
					new_height = new_side_length
					new_width = int(round(float(width) / (float(height)/float(new_height))))
				else:
					new_width = new_side_length
					new_height = int(round(float(height) / (float(width)/float(new_width))))
				warped_image = skimage.transform.resize(image, (new_height, new_width), mode = 'reflect')
			else:
				new_height = height
				new_width = width
				warped_image = numpy.array(image)
			cropped_image = warped_image[:new_side_length, :new_side_length, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[:new_side_length, new_width/2-new_side_length/2:new_width/2+new_side_length/2, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[:new_side_length, -new_side_length:, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[new_height/2-new_side_length/2:new_height/2+new_side_length/2, :new_side_length, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[new_height/2-new_side_length/2:new_height/2+new_side_length/2, new_width/2-new_side_length/2:new_width/2+new_side_length/2, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[new_height/2-new_side_length/2:new_height/2+new_side_length/2, -new_side_length:, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[-new_side_length:, :new_side_length, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[-new_side_length:, new_width/2-new_side_length/2:new_width/2+new_side_length/2, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			cropped_image = warped_image[-new_side_length:, -new_side_length:, :]
			transformed_image = transformer.preprocess('data', cropped_image)
			batch_patch[count_patch,:,:,:] = transformed_image
			count_patch += 1
			if count_patch == batch_size*9:
				net_patch.blobs['data'].data[...] = batch_patch
				output = net_patch.forward()
				for layer in defect_layer[:-1]:
					prob = numpy.array(numpy.squeeze(output[layer]))
					if layer == 'softmax_badSaturation':
						my_peak = numpy.array(peak_saturation)
					else:
						my_peak = numpy.array(peak)
					my_peak = numpy.tile(my_peak, (batch_size*9,1))
					score = numpy.sum(prob*my_peak, 1)
					for i in range(batch_size):
						defect_score_patch[layer].append(numpy.mean(score[i*9:(i+1)*9]))
				batch_patch = numpy.zeros((batch_size*9, 3, new_side_length, new_side_length))
				count_patch = 0

	if count_holistic > 0:
		net_holistic.blobs['data'].data[...] = batch_holistic
		output = net_holistic.forward()
		for layer in defect_layer:
			prob = numpy.array(numpy.squeeze(output[layer]))[:count_holistic,:]
			if layer == 'softmax_badSaturation':
				my_peak = numpy.array(peak_saturation)[:count_holistic,:]
			else:
				my_peak = numpy.array(peak)[:count_holistic,:]
			score = numpy.sum(prob*my_peak, 1)
			defect_score_holistic[layer] += list(score)
	if count_patch > 0:
		net_patch.blobs['data'].data[...] = batch_patch
		output = net_patch.forward()
		for layer in defect_layer[:-1]:
			prob = numpy.array(numpy.squeeze(output[layer]))[:count_patch,:]
			if layer == 'softmax_badSaturation':
				my_peak = numpy.array(peak_saturation)[:count_patch,:]
			else:
				my_peak = numpy.array(peak)[:count_patch,:]
			score = numpy.sum(prob*my_peak, 1)
			for i in range(count_patch/9):
				defect_score_patch[layer].append(numpy.mean(score[i*9:(i+1)*9]))
	
	end_time = time.time()
	print end_time-start_time


	# write defect score csv files
	if oPath[-1] != '/':
		oPath += '/'
	file = open('%sdefect_scores_holistic.csv' % oPath, 'w+')
	csvWriter = csv.writer(file)
	row = ['path'] + defect_list
	csvWriter.writerow(row)
	for i in range(len(path_list)):
		path = os.path.relpath(path_list[i], oPath)
		row = [path]
		for layer in defect_layer:
			score = defect_score_holistic[layer][i]
			row.append(score)
		csvWriter.writerow(row)
	file.close()

	file = open('%sdefect_scores_patch.csv' % oPath, 'w+')
	csvWriter = csv.writer(file)
	row = ['path'] + defect_list[:-1]
	csvWriter.writerow(row)
	for i in range(len(path_list)):
		path = os.path.relpath(path_list[i], oPath)
		row = [path]
		for layer in defect_layer[:-1]:
			score = defect_score_patch[layer][i]
			row.append(score)
		csvWriter.writerow(row)
	file.close()

	score_dict = {}
	for defect_type in defect_list:
		score_dict[defect_type] = {}
	file = open('%sdefect_scores_combined.csv' % oPath, 'w+')
	csvWriter = csv.writer(file)
	row = ['path'] + defect_list
	csvWriter.writerow(row)
	for i in range(len(path_list)):
		path = os.path.relpath(path_list[i], oPath)
		row = [path]
		for (j, layer) in enumerate(defect_layer[:-1]):
			score = (defect_score_holistic[layer][i] + defect_score_patch[layer][i]) / 2.0
			row.append(score)
			defect_type = defect_list[j]
			score_dict[defect_type][path] = score
		score = defect_score_holistic['softmax_badComposition'][i]
		row.append(score)
		defect_type = defect_list[-1]
		score_dict[defect_type][path] = score
		csvWriter.writerow(row)
	file.close()

	# for each defect, visualize the testing images in the descent order of the corresponding scores, and write to an html file
	for defect_type in defect_list:
		scores = score_dict[defect_type]
		scores_sorted = sorted(scores.items(), key = lambda x: x[1], reverse = True)

		html = open('%sdefect_scores_combined_%s.html' % (oPath, defect_type), 'w+')
		message = """<html><body>
			<table border="1" style="width:100%">"""
		html.write(message)
		count = 0
		for (path, score) in scores_sorted:
			if count == 0:
				message = """<tr>
					<td width="%d%%"><div align="center">
					<img width="%d" src=%s>
					<figcaption><p>%s score = %s</p></figcaption>
					</div></td>""" % (100/num_images_per_row_vis, col_width, path, defect_type, str(score))
				count += 1
			elif count < num_images_per_row_vis - 1:
				message = """<td width="%d%%"><div align="center">
					<img width="%d" src=%s>
					<figcaption><p>%s score = %s</p></figcaption>
					</div></td>""" % (100/num_images_per_row_vis, col_width, path, defect_type, str(score))
				count += 1
			else:
				message = """<td width="%d%%"><div align="center">
					<img width="%d" src=%s>
					<figcaption><p>%s score = %s</p></figcaption>
					</div></td>
					</tr>""" % (100/num_images_per_row_vis, col_width, path, defect_type, str(score))
				count = 0
			html.write(message)
		message = """</table>
			</body></html>"""
		html.write(message)		
		html.close()

if __name__ == '__main__':
    main()