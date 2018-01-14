import argparse
import numpy
import numpy.random
import scipy
import scipy.stats
import csv
import os
import os.path

def main():
	# parse command line
	parser = argparse.ArgumentParser()
	parser.add_argument('-gtPath', type = str, default = 'None', help = 'csv file of the ground truth defect severity scores')
	parser.add_argument('-predPath', type = str, default = 'None', help = 'csv file of the predicted defect severity scores')
	parser.add_argument('-oPath', type = str, default = './outputs/', help = 'output directory containing the evaluation csv file')
	args = parser.parse_args()
	gtPath = args.gtPath
	if not os.path.isfile(gtPath):
		print 'csv file of the ground truth defect severity scores does not exist!'
		return
	predPath = args.predPath
	if not os.path.isfile(predPath):
		print 'csv file of the predicted defect severity scores does not exist!'
		return
	oPath = args.oPath
	if not os.path.isdir(oPath):
		os.makedirs(oPath)
		os.chmod(oPath, 0o777)

	# global variables
	defect_list = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur', 'Bad Composition']
	hist_bin = numpy.arange(-0.05, 1.1, 0.1)
	hist_bin_saturation = numpy.array([-1.05, -0.85, -0.65, -0.45, -0.25, -0.05, 0.05, 0.25, 0.45, 0.65, 0.85, 1.05])
	num_images_per_row_vis = 3
	thres_gt = 0.25

	# read gt
	name_list = []
	gt_file = open(gtPath, 'r')
	csvReader = csv.reader(gt_file)
	gt_dict = {}
	name_list = []
	for defect_type in defect_list:
		gt_dict[defect_type] = []
	idx_dict = {}
	count = 0
	for row in csvReader:
		if count == 0:
			item_list = row
			for defect_type in defect_list:
				idx = item_list.index(defect_type)
				idx_dict[defect_type] = idx
		else:
			for defect_type in defect_list:
				idx = idx_dict[defect_type]
				level = float(row[idx])
				gt_dict[defect_type].append(level)
		count += 1
	gt_file.close()

	# read prediction
	pred_file = open(predPath, 'r')
	csvReader = csv.reader(pred_file)
	pred_dict = {}
	for defect_type in defect_list:
		pred_dict[defect_type] = []
	idx_dict = {}
	count = 0
	for row in csvReader:
		if count == 0:
			item_list = row
			for defect_type in defect_list:
				idx = item_list.index(defect_type)
				idx_dict[defect_type] = idx
		else:
			for defect_type in defect_list:
				idx = idx_dict[defect_type]
				level = float(row[idx])
				pred_dict[defect_type].append(level)
		count += 1
	pred_file.close()
	
	# compute and write cross-class ranking correlation
	if oPath[-1] != '/':
		oPath += '/'
	evaluation_file = open('%sevaluation.csv' % oPath, 'w+')
	csvWriter = csv.writer(evaluation_file)
	row = ['', 'Overall'] + defect_list
	csvWriter.writerow(row)
	cross_class_ranking_correlation_row = ['cross_class_ranking_correlation']
	for (j, defect_type) in enumerate(defect_list):
		gt = numpy.array(gt_dict[defect_type])
		pred = numpy.array(pred_dict[defect_type])
		num_samples = len(gt) * 10
		if defect_type == 'Bad Saturation':
			my_bin = hist_bin_saturation
		else:
			my_bin = hist_bin
		hist = numpy.histogram(gt, bins = my_bin)[0]
		idx_samples_array = numpy.zeros((num_samples, len(hist)))
		idx_samples_array = idx_samples_array.astype('uint32')
		for i in range(len(hist)):
			idx = numpy.where(scipy.logical_and(gt>=my_bin[i], gt<=my_bin[i+1]))[0]
			idx_samples = numpy.random.choice(idx, num_samples)
			idx_samples_array[:,i] = idx_samples
		correlation_samples_list = []
		for i in range(num_samples):
			print 'Sampling: %s: %d/%d' % (defect_type, i, num_samples)
			idx_samples = idx_samples_array[i,:]
			gt_samples = gt[idx_samples]
			pred_samples = pred[idx_samples]
			[correlation_samples, p_value] = scipy.stats.spearmanr(gt_samples, pred_samples)
			correlation_samples_list.append(correlation_samples)
		cross_class_ranking_correlation = numpy.mean(numpy.array(correlation_samples_list))
		cross_class_ranking_correlation_row.append(cross_class_ranking_correlation)
	cross_class_ranking_correlation_row.insert(1, numpy.mean(numpy.array(cross_class_ranking_correlation_row[1:])))
	csvWriter.writerow(cross_class_ranking_correlation_row)
	evaluation_file.close()

if __name__ == '__main__':
    main()