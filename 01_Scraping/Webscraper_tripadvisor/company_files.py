import os 
import yaml



company_dict = yaml.load(open('companies.yaml', 'r'))
company_list = list(company_dict.items())


for nb in range(len(company_list) - 7):
	if nb % 8 == 0:
		if not os.path.exists('scrapers/scrap_{}/'.format(nb//8)):
			os.mkdir('scrapers/scrap_{}/'.format(nb//8))

		if not os.path.exists('scrapers/scrap_{}/company.yaml'.format(nb//8)):
			file = open('scrapers/scrap_{}/company.yaml'.format(nb//8),'w')
			config = dict({
				company_list[nb][0]: company_list[nb][1],
				company_list[nb + 1][0]: company_list[nb + 1][1],
				company_list[nb + 2][0]: company_list[nb + 2][1],
				company_list[nb + 3][0]: company_list[nb + 3][1],
				company_list[nb + 4][0]: company_list[nb + 4][1],
				company_list[nb + 5][0]: company_list[nb + 5][1],
				company_list[nb + 6][0]: company_list[nb + 6][1],
				company_list[nb + 7][0]: company_list[nb + 7][1]})
			file.write(yaml.dump(config))
			file.close()