#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:00:13 2019

@author: vincent roy
"""

import pandas as pd
import os
import yaml

company_dict = yaml.load(open('companies.yaml', 'r'))
print(company_dict)


def concat_by_companies(company_dict):
	try:
		os.mkdir('TripAdvisors_data/')
	except FileExistsError:
		pass
	try:
		os.mkdir('TripAdvisors_data/data_by_companies/')
	except FileExistsError:
		pass
	company_list = list(company_dict.items())

	for company in company_list:
		page = 1
		data_list = []
		next_page_exist = True
		while next_page_exist:
		 	if os.path.exists('results/{}/page_{}.csv'.format(company[0], page)):
		 		data_list.append(pd.read_csv('results/{}/page_{}.csv'.format(company[0], page), sep = '|'))
		 		page += 1
		 	else:
		 		next_page_exist = False
		 		try: 
			 		data_df = pd.concat(data_list, ignore_index = True)
			 		data_df['company'] = company[0]
			 		data_df.to_csv('TripAdvisors_data/data_by_companies/TripAdvisors_{}.csv'.format(company[0]), sep = '|', index = False)
			 		print('{} data exported'.format(company[0]))
			 	except:
			 		pass
		 		page = 1
		 		data_list = []

def concat_to_one_file(company_dict):
	company_list = list(company_dict.items())
	data_list = []
	for company in company_list:
		try:
			company_df = pd.read_csv('TripAdvisors_data/data_by_companies/TripAdvisors_{}.csv'.format(company[0]), sep = '|')
			data_list.append(company_df)
		except:
			pass
	data_df = pd.concat(data_list, ignore_index = True)
	data_df.to_csv('TripAdvisors_data/TripAdvisors_data.csv', sep = '|', index = False)
	print('Data exported in one file')



concat_by_companies(company_dict)
concat_to_one_file(company_dict)
