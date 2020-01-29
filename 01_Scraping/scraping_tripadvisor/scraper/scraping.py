#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:00:13 2019

@author: vincent roy
"""


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, NoSuchWindowException, StaleElementReferenceException, TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import yaml
os.getcwd()
from urllib.request import urlopen as uReq
from time import sleep

#launch url
def open_driver():
    try:
        driver.close()
    except (NameError, NoSuchWindowException):
        pass
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument("--no-startup-window")
    driver = webdriver.Chrome('/usr/bin/chromedriver', options = options)
    driver.implicitly_wait(8)
    return driver

def close_driver(driver):
    try:
        driver.close()
    except (NameError, NoSuchWindowException):
        pass

def click_on_plus_button(driver):
    buttons = driver.find_elements_by_class_name("location-review-review-list-parts-ExpandableReview__cta--2mR2g")
    for k in range(len(buttons)):
        sleep(0.1)
        try:
            buttons_2 = driver.find_elements_by_class_name("location-review-review-list-parts-ExpandableReview__cta--2mR2g")
            buttons_2[k].click()
        except (IndexError, ElementNotInteractableException, StaleElementReferenceException):
            pass

def click_on_next_button(driver):
    sleep(0.5)
    button = driver.find_element_by_link_text("Next")
    button.click()


def get_grades(review_bloc):
    grade_dict = {}
    grades_list = review_bloc.findAll('div',attrs={"location-review-review-list-parts-AdditionalRatings__rating--1_G5W"})
    for grade in grades_list:
        grade_dict[grade.text] = list(grade.find('span').find('span').attrs.values())[0][1].split('_')[-1][0]
    return grade_dict

def get_reviews(driver):
    reviews_dict = {
        "username": {},
        "entry_date": {},
        "user_hometown": {},
        "user_contributions_number": {},
        "user_useful_vote_number": {},
        "flight_cities": {},
        "flight_type": {},
        "passenger_class": {},
        "general_grade": {},
        "title": {},
        "review_text": {},
        "travel_date": {},
        "grades": {},
        "helpful_review_vote_number": {}
    }
    try :
        #Selenium hands the page source to Beautiful Soup
        soup = BeautifulSoup(driver.page_source)

        review_blocs = soup.findAll('div',attrs={"class":"location-review-review-list-parts-SingleReview__mainCol--1hApa"})
        members_infos = soup.findAll('div',attrs={"class":"social-member-event-MemberEventOnObjectBlock__event_wrap--1YkeG"})[:len(review_blocs)]
        for reviewer in range(len(review_blocs)):
            try:
                reviews_dict['username'][reviewer] = members_infos[reviewer].find('a',attrs={"class":"ui_header_link social-member-event-MemberEventOnObjectBlock__member--35-jC"}).text
            except:
                reviews_dict['username'][reviewer] = None
                
            try:
                member_text = members_infos[reviewer].find('div', attrs = {"class":"social-member-event-MemberEventOnObjectBlock__event_type--3njyv"}).find('span')
                reviews_dict['entry_date'][reviewer] = member_text.text.split(member_text.find('a').text)[1].split('review')[1].strip()
            except:
                reviews_dict['entry_date'][reviewer] = None

            try:
                reviews_dict['user_hometown'][reviewer] = members_infos[reviewer].find('span', attrs = {"class": "default social-member-common-MemberHometown__hometown--3kM9S small"}).text
            except:
                reviews_dict['user_hometown'][reviewer] = None

            user_contributions = members_infos[reviewer].findAll('span', attrs = {"class": "social-member-MemberHeaderStats__bold--3z3qh"})
            try:
                 reviews_dict['user_contributions_number'][reviewer] = user_contributions[0].text
            except:
                reviews_dict['user_contributions_number'][reviewer] = None
            try:
                 reviews_dict['user_useful_vote_number'][reviewer] = user_contributions[1].text
            except:
                reviews_dict['user_useful_vote_number'][reviewer] = None


            location_review = review_blocs[reviewer].findAll('div',attrs={"location-review-review-list-parts-RatingLine__labelBtn--e58BL"})
            try:
                reviews_dict['flight_cities'][reviewer] = location_review[0].text
            except:
                reviews_dict['flight_cities'][reviewer] = None
            try:
                 reviews_dict['flight_type'][reviewer] = location_review[1].text
            except:
                reviews_dict['flight_type'][reviewer] = None
            try:
                 reviews_dict['passenger_class'][reviewer] = location_review[2].text
            except:
                reviews_dict['passenger_class'][reviewer]= None

            try:
                reviews_dict['general_grade'][reviewer] = list(review_blocs[reviewer]
                                                .find('div',attrs={"class":"location-review-review-list-parts-RatingLine__bubbles--GcJvM"})
                                                .find('span')
                                                .attrs
                                                .values())[0][1].split('_')[-1][0]
            except:
                reviews_dict['general_grade'][reviewer] = None

            try:
                reviews_dict['title'][reviewer] = review_blocs[reviewer].find('a',attrs={"class":"location-review-review-list-parts-ReviewTitle__reviewTitleText--2tFRT"}).text
            except NoSuchElementException:
                 reviews_dict['title'][reviewer] = None

            try:
                 reviews_dict['review_text'][reviewer] = review_blocs[reviewer].find('q',attrs={"class":"location-review-review-list-parts-ExpandableReview__reviewText--gOmRC"}).text.replace('\n',' ')
            except:
                reviews_dict['review_text'][reviewer] = None

            try:
                reviews_dict['travel_date'][reviewer] = review_blocs[reviewer].find('span',attrs={"location-review-review-list-parts-EventDate__event_date--1epHa"}).text.split(':')[-1].strip()
            except:
                reviews_dict['travel_date'][reviewer] = None

            try:
                reviews_dict['grades'][reviewer] = get_grades(review_blocs[reviewer])
            except:
                reviews_dict['grades'][reviewer] = None
            try:
                reviews_dict['helpful_review_vote_number'][reviewer] = review_blocs[reviewer].find('span', attrs = {"class":"social-statistics-bar-SocialStatisticsBar__counts--3Zm4V social-statistics-bar-SocialStatisticsBar__item--2IlT7"}).text.split(' ')[0]
            except:
                reviews_dict['helpful_review_vote_number'][reviewer] = None
    except TypeError:
        pass
    return reviews_dict

def get_page_review(driver, company, page):
    if os.path.exists('data/{}/page_{}.csv'.format(company[0], page)):
        print('page {} of {} exists'.format(page, company[0]))
    else:
        click_on_plus_button(driver)
        reviews = pd.DataFrame(get_reviews(driver))
        reviews.to_csv('data/{}/page_{}.csv'.format(company[0], page), sep = '|', index = False)
        print('page {} of {} done'.format(page, company[0]))

def get_reviews_companies(company_dict):
    company_list = list(company_dict.items())
    if not os.path.exists('data/'):
        os.mkdir('data/')
    for company in company_list:
        try:
            os.mkdir('data/{}/'.format(company[0]))
        except FileExistsError:
            pass
            
        count = 0
        while count < 5:
            try:
                driver = open_driver()
                print("driver opened")
                driver.get(company[1])
                next_page_exist = True
                page = 1
                print("start scraping")
                count = 5
            except (TimeoutException, WebDriverException): 
                sleep(10)
                count += 1
                pass

        while next_page_exist:
            try:
                get_page_review(driver, company, page)
                try:
                    click_on_next_button(driver)
                except:
                    next_page_exist = False
                    print('{} done'.format(company[0]))
                page += 1
            except (TimeoutException, WebDriverException):
                print('Driver TimeoutException')
                close_driver(driver)
                driver = open_driver()
                driver.get(company[1])
                next_page_exist = True
                page = 1
        sleep(30)

company_dict = yaml.load(open('company.yaml', 'r'))
print(company_dict)

get_reviews_companies(company_dict)
