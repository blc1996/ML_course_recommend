#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from heapq import nlargest

# following two def functions are taken from 
# https://stackoverflow.com/questions/33426864/replace-numbers-with-roman-numerals
# and https://stackoverflow.com/questions/28777219/basic-program-to-convert-integer-to-roman-numerals
def write_roman(num):
    """ write numerical numbers to roman numbers """
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"
    
    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])

def repl(match):
    """ pass numerical numbers to write_roman function """
    return write_roman(int(match.group(0)))

def data_preprocess():
    """ prepare data for similarity calculation """
    # read data from gpa dataset
    gpa = pd.read_csv("uiuc-gpa-dataset.csv")
    # drop Year, Term, YearTerm columns in the gpa dataset because there are 
    # too much replicates in these columns
    gpa = gpa.drop(columns = ["Year", "Term", "YearTerm"])
    # remove commas in the PrimaryInstructor column for easier usage
    gpa["PrimaryInstructor"] = gpa["PrimaryInstructor"].str.replace(',', '')
    # sum students count for the same course and instructor but different 
    # years and terms
    sum_gpa = gpa.groupby(["CourseTitle", "PrimaryInstructor", "Subject", "Number"],
                          as_index = False).sum()
    # count the total number of students in each course
    sum_gpa["count"] = sum_gpa.iloc[:,4:18].sum(axis=1)
    # count the percentage distribute for each grade 
    sum_gpa.iloc[:,4:18] = (sum_gpa.iloc[:,4:18]).div(sum_gpa['count'].values,axis=0)
    # drop the count column
    sum_gpa = sum_gpa.drop(columns = ["count"])
    
    # read data from profrating dataset
    rating = pd.read_csv("profrating.csv")
    # center ratings and level of difficulties
    rating.iloc[:,[1]] = rating.iloc[:,[1]] - 3.6
    rating.iloc[:,[2]] = rating.iloc[:,[2]] - 2.9
    
    # merge gpa dataset with professor ratings
    # NA elements are replaced with 0, which is the center of the dataset
    merge_gpa_rating = pd.merge(sum_gpa, rating.iloc[:,[0,1]], on = ["PrimaryInstructor"], 
                            how = "left").fillna(0)
    # merge gpa dataset with course level of difficulties
    # NA elements are replaced with 0, which is the center of the dataset
    merge_gpa_rating = pd.merge(merge_gpa_rating, rating.iloc[:,[0,2]], on = ["PrimaryInstructor"], 
                            how = "left").fillna(0)
    
    # remove punctuations in course titles
    merge_gpa_rating["CourseTitle"] = merge_gpa_rating["CourseTitle"].str.replace('[^\w\s]', '')
    # remove extra whitespaces in course titles
    merge_gpa_rating["CourseTitle"] = merge_gpa_rating["CourseTitle"].str.replace('\s+', ' ')
    # replace numbers in the course tiles to roman numbers
    # this is because numerical numbers are not counted in CountVectorizer when 
    # calculating similarities
    regex = re.compile(r"\b\d+\b")
    merge_gpa_rating["CourseTitle"] = merge_gpa_rating["CourseTitle"].str.replace(regex, repl)
    
    # combine instructors' full names into one string because we want to avoid
    # situations that two different instructors are counted as similar due to their 
    # same first name or last name. Only the same professor should be counted as 
    # the same for similarity calculation
    merge_gpa_rating["PrimaryInstructor"] = merge_gpa_rating["PrimaryInstructor"].str.replace('[^\w]', '')
    
    # return the processed dataframe for similarity calculation
    return merge_gpa_rating

def cos_similarity(merge_gpa_rating):
    """ calculate cosine similarity (Pearson correlation) for the dataset
    five cosine similarity matrixes are calculated:
        1. course title similarity (range [0, 1])
        2. professor name similairy (boolen)
        3. major similarity (boolen)
        4. gpa distribution similarity (range [0, 1])
        5. rating and level of difficulty similariy (range [0, 1])
    """
    #title similarity
    title = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
    title_matrix = title.fit_transform(merge_gpa_rating["CourseTitle"])
    title_cosine = cosine_similarity(title_matrix, title_matrix)
    
    #professor similarity
    prof = CountVectorizer()
    prof_matrix = prof.fit_transform(merge_gpa_rating["PrimaryInstructor"])
    prof_cosine = cosine_similarity(prof_matrix, prof_matrix)
    
    #major similarity
    major = CountVectorizer()
    major_matrix = major.fit_transform(merge_gpa_rating["Subject"])
    major_cosine = cosine_similarity(major_matrix, major_matrix)
    
    #gpa similarity
    gpa_matrix = merge_gpa_rating.iloc[:,4:18]
    gpa_cosine = cosine_similarity(gpa_matrix, gpa_matrix)
    
    #rating and level of difficulty similarity
    rating_matrix = merge_gpa_rating.iloc[:,18:20]
    rating_cosine = cosine_similarity(rating_matrix, rating_matrix)
    
    # sum all similarity matrix
    similarity = title_cosine + prof_cosine * 0.3 + major_cosine * 0.3 + gpa_cosine + rating_cosine
    
    # return the total cosine similarity 
    return similarity


# function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(course, professor):
    """ functino to perform course recommendations 
    input:
        course name
        professor name
    output:
        10 recommended courses 
    """
    recommended_course = []
    
    # gettin the index of the movie that matches the title
    idx = idx_course.loc[(idx_course["CourseTitle"] == course) & 
                  (idx_course["PrimaryInstructor"] == professor)].index.tolist()

    # getting the indexes of the 10 most similar movies
    top_10 = nlargest(11, enumerate(similarity[int(idx[0])]), key=lambda x: x[1])[1:11]
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10:
        recommended_course.append(merge_gpa_rating.iloc[i[0],[0,1,2]].values)
        
    return recommended_course

def run(course, professor):
    global merge_gpa_rating
    merge_gpa_rating = data_preprocess()
    
    # calculate total cosine similarity
    global similarity
    similarity = cos_similarity(merge_gpa_rating)
    
    # courese index
    global idx_course
    idx_course = merge_gpa_rating.iloc[:,[0,1]]
    
    return recommendations(course, professor)

if __name__ == '__main__':
    """ main funciton """
    # data prepocess
    merge_gpa_rating = data_preprocess()
    
    # calculate total cosine similarity
    similarity = cos_similarity(merge_gpa_rating)
    
    # courese index
    idx_course = merge_gpa_rating.iloc[:,[0,1]]
    
    # examples
    [list(i) for i in recommendations('Database Systems', 'ChangKevinC')]
    [list(i) for i in recommendations('Railroad Transportation Engrg', 'BarkanChristopherP')]
    [list(i) for i in recommendations('Audience Analysis', 'SarSela')]
    [list(i) for i in recommendations('New Product Development', 'MehtaRaviP')]

