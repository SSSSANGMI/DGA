import re
import collections
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class FeatureExtraction():
    
    def __init__(self):
        self.result = 0
    
    @staticmethod
    def featureExtractionAtOnce(preprocessList, domain_df, suffixed):

        if suffixed == True:
            obj = 'suffixed'
        else:
            obj = 'domain'

        topLevelDomain = []
        with open('./data/tlds-alpha-by-domain.txt', 'r') as content:
            for line in content:
                topLevelDomain.append((line.strip('\n')))

        domain_withFeatures = domain_df.copy()
        if 'DNL' in preprocessList:
            domain_withFeatures['DNL'] = domain_withFeatures['domain'].apply(lambda x:FeatureExtraction.domain_length(str(x)))
        if 'NoS' in preprocessList:
            domain_withFeatures['NoS'] = domain_withFeatures[obj].apply(lambda x:FeatureExtraction.subdomains_number(str(x)))
        if 'SLM' in preprocessList:
            domain_withFeatures['SLM'] = domain_withFeatures[obj].apply(lambda x:FeatureExtraction.subdomain_length_mean(str(x)))
        if 'HwP' in preprocessList:
            domain_withFeatures['HwP'] = domain_withFeatures['domain'].apply(lambda x:FeatureExtraction.has_www_prefix(str(x)))
        if 'HVTLD' in preprocessList:
            domain_withFeatures['HVTLD'] = domain_withFeatures['domain'].apply(lambda x:FeatureExtraction.has_hvltd(topLevelDomain, str(x)))
        if 'CSCS' in preprocessList:
            domain_withFeatures['CSCS'] = domain_withFeatures[obj].apply(lambda x:FeatureExtraction.contains_single_character_subdomain(str(x)))
        if 'CTS' in preprocessList:
            domain_withFeatures['CTS'] = domain_withFeatures[obj].apply(lambda x:FeatureExtraction.contains_TLD_subdomain(topLevelDomain, str(x)))
        if 'UR' in preprocessList:
            domain_withFeatures['UR'] = domain_withFeatures[obj].apply(lambda x:FeatureExtraction.underscore_ratio(str(x)))
        if 'CIPA' in preprocessList:
            domain_withFeatures['CIPA'] = domain_withFeatures['domain'].apply(lambda x:FeatureExtraction.contains_IP_address(str(x)))
        if 'contains_digit' in preprocessList:
            domain_withFeatures['contains_digit']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.contains_digit(str(x)))
        if 'vowel_ratio' in preprocessList:
            domain_withFeatures['vowel_ratio']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.vowel_ratio(str(x)))
        if 'digit_ratio' in preprocessList:
            domain_withFeatures['digit_ratio']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.digit_ratio(str(x)))
        if 'RRC' in preprocessList:
            domain_withFeatures['RRC']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.prc_rrc(str(x)))
        if 'RCC' in preprocessList:
            domain_withFeatures['RCC']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.prc_rcc(str(x)))
        if 'RCD' in preprocessList:
            domain_withFeatures['RCD']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.prc_rcd(str(x)))
        if 'Entropy' in preprocessList:
            domain_withFeatures['Entropy']= domain_withFeatures[obj].apply(lambda x:FeatureExtraction.prc_entropy(str(x)))

        return domain_withFeatures
    
    

    # Generate Domain Name Length (DNL)
    @staticmethod
    def domain_length(domain):

        return len(domain)
    
    @staticmethod
    def subdomains_number(domain):
        # Generate Number of Subdomains (NoS)

        return (domain.count('.') + 1)

    @staticmethod
    def subdomain_length_mean(domain):
        # enerate Subdomain Length Mean (SLM)

        result = (len(domain) - domain.count('.')) / (domain.count('.') + 1)
        return result

    @staticmethod
    def has_www_prefix(domain):
        # Generate Has www Prefix (HwP)
        if domain.split('.')[0] == 'www':
            return 1
        else:
            return 0

    @staticmethod
    def has_hvltd(topLevelDomain, domain):
        # Generate Has a Valid Top Level Domain (HVTLD)

        if domain.split('.')[len(domain.split('.')) - 1].upper() in topLevelDomain:
            return 1
        else:
            return 0

    @staticmethod
    def contains_single_character_subdomain(domain):
        # Generate Contains Single-Character Subdomain (CSCS)
        str_split = domain.split('.')
        minLength = len(str_split[0])
        for i in range(0, len(str_split) - 1):
            minLength = len(str_split[i]) if len(str_split[i]) < minLength else minLength
        if minLength == 1:
            return 1
        else:
            return 0

    @staticmethod
    def contains_TLD_subdomain(topLevelDomain, domain):
        # Generate Contains TLD as Subdomain (CTS)
        str_split = domain.split('.')
        for i in range(0, len(str_split) - 1):
            if str_split[i].upper() in topLevelDomain:
                return 1
        return 0

    @staticmethod
    def underscore_ratio(domain):
        # Generate Underscore Ratio (UR) on dataset

        result = domain.count('_') / (len(domain) - domain.count('.'))
        return result

    @staticmethod
    def contains_IP_address(domain):
        # Generate Contains IP Address (CIPA) on datasetx
        splitSet = domain.split('.')
        for element in splitSet:
            if(re.match("\d+", element)) == None:
                return 0
        return 1

    @staticmethod
    def contains_digit(domain):
        """
        Contains Digits
        """
        for item in domain:
            if item.isdigit():
                return 1
        return 0

    @staticmethod
    def vowel_ratio(domain):
        """
        calculate Vowel Ratio
        """
        VOWELS = set('aeiou')
        v_counter = 0
        a_counter = 0
        ratio = 0

        for item in domain:
            if item.isalpha():
                a_counter+=1
                if item in VOWELS:
                    v_counter+=1
        if a_counter>1:
            ratio = v_counter/a_counter
        return ratio

    @staticmethod
    def digit_ratio(domain):
        """
        calculate digit ratio
        """
        d_counter = 0
        counter = 0
        ratio = 0
        for item in domain:
            if item.isalpha() or item.isdigit():
                counter+=1
                if item.isdigit():
                    d_counter+=1
        if counter>1:
            ratio = d_counter/counter
        return ratio

    @staticmethod
    def prc_rrc(domain):
        """
        calculate the Ratio of Repeated Characters in a subdomain
        """
        domain = re.sub("[.]", "", domain)
        char_num=0
        repeated_char_num=0
        d = collections.defaultdict(int)
        for c in list(domain):
            d[c] += 1
        for item in d:
            char_num +=1
            if d[item]>1:
                repeated_char_num +=1
        ratio = repeated_char_num/char_num
        return ratio

    @staticmethod
    def prc_rcc(domain):
        """
        calculate the Ratio of Consecutive Consonants
        """
        VOWELS = set('aeiou')
        counter = 0
        cons_counter=0
        for item in domain:
            i = 0
            if item.isalpha() and item not in VOWELS:
                counter+=1
            else:
                if counter>1:
                    cons_counter+=counter
                counter=0
            i+=1
        if i==len(domain) and counter>1:
            cons_counter+=counter
        ratio = cons_counter/len(domain)
        return ratio

    @staticmethod
    def prc_rcd(domain):
        """
        calculate the ratio of consecutive digits
        """
        counter = 0
        digit_counter=0
        for item in domain:
            i = 0
            if item.isdigit():
                counter+=1
            else:
                if counter>1:
                    digit_counter+=counter
                counter=0
            i+=1
        if i==len(domain) and counter>1:
            digit_counter+=counter
        ratio = digit_counter/len(domain)
        return ratio

    @staticmethod
    def prc_entropy(domain):
        """
        calculate the entropy of subdomain
        :param domain_str: subdomain
        :return: the value of entropy
        """
        # get probability of chars in string
        prob = [float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy


    @staticmethod
    def corr_features(df):

        corrmat = df.corr()

        plt.figure(figsize=(15,15))
        sns.heatmap(corrmat, annot=True, cmap= "RdBu_r")

        k = 10#number of variables for heatmap
        cols = corrmat.nlargest(k, 'label')['label'].index
        cm = np.corrcoef(df[cols].values.T)
        f, ax = plt.subplots(figsize=(16, 16))
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cmap = "RdBu_r", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()



    @staticmethod
    def drop_features(df, columns):
        return df.drop(columns, axis = 1)