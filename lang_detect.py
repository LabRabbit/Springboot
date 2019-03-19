
# Importing useful inbuilt libraries

import os, re, sys
import numpy as np
from collections import defaultdict
import pickle

# iso language codes

iso_lang_codes = dict([('ab', 'Abkhaz'), ('aa', 'Afar'), ('af', 'Afrikaans'), ('ak', 'Akan'), 
('sq', 'Albanian'), ('am', 'Amharic'), ('ar', 'Arabic'), ('an', 'Aragonese'), ('hy', 'Armenian'),
('as', 'Assamese'), ('av', 'Avaric'), ('ae', 'Avestan'), ('ay', 'Aymara'), ('az', 'Azerbaijani'),
('bm', 'Bambara'), ('ba', 'Bashkir'), ('eu', 'Basque'), ('be', 'Belarusian'), ('bn', 'Bengali'),
('bh', 'Bihari'), ('bi', 'Bislama'), ('bs', 'Bosnian'), ('br', 'Breton'), ('bg', 'Bulgarian'),
('my', 'Burmese'), ('ca', 'Catalan; Valencian'), ('ch', 'Chamorro'), ('ce', 'Chechen'), 
('ny', 'Chichewa; Chewa; Nyanja'), ('zh', 'Chinese'),  ('cv', 'Chuvash'), ('kw', 'Cornish'),
('co', 'Corsican'), ('cr', 'Cree'), ('hr', 'Croatian'), ('cs', 'Czech'), ('da', 'Danish'), 
('dv', 'Divehi; Maldivian;'), ('nl', 'Dutch'), ('dz', 'Dzongkha'), ('en', 'English'), ('eo', 'Esperanto'), 
('et', 'Estonian'), ('ee', 'Ewe'), ('fo', 'Faroese'), ('fj', 'Fijian'),('fi', 'Finnish'), ('fr', 'French'),
('ff', 'Fula'),  ('gl', 'Galician'), ('ka', 'Georgian'), ('de', 'German'), ('el', 'Greek, Modern'), 
('gn', 'Guaraní'), ('gu', 'Gujarati'), ('ht', 'Haitian'), ('ha', 'Hausa'), ('he', 'Hebrew (modern)'),  
('hz', 'Herero'), ('hi', 'Hindi'), ('ho', 'Hiri Motu'), ('hu', 'Hungarian'), ('ia', 'Interlingua'), 
('id', 'Indonesian'), ('ie', 'Interlingue'), ('ga', 'Irish'), ('ig', 'Igbo'), ('ik', 'Inupiaq'), 
('io', 'Ido'), ('is', 'Icelandic'), ('it', 'Italian'), ('iu', 'Inuktitut'), ('ja', 'Japanese'),
('jv', 'Javanese'), ('kl', 'Kalaallisut'), ('kn', 'Kannada'), ('kr', 'Kanuri'), ('ks', 'Kashmiri'), 
('kk', 'Kazakh'), ('km', 'Khmer'), ('ki', 'Kikuyu, Gikuyu'), ('rw', 'Kinyarwanda'), ('ky', 'Kirghiz, Kyrgyz'),
('kv', 'Komi'),  ('kg', 'Kongo'), ('ko', 'Korean'), ('ku', 'Kurdish'), ('kj', 'Kwanyama, Kuanyama'), 
('la', 'Latin'), ('lb', 'Luxembourgish'), ('lg', 'Luganda'), ('li', 'Limburgish'), ('ln', 'Lingala'), 
('lo', 'Lao'), ('lt', 'Lithuanian'), ('lu', 'Luba-Katanga'), ('lv', 'Latvian'), ('gv', 'Manx'),
('mk', 'Macedonian'),  ('mg', 'Malagasy'), ('ms', 'Malay'), ('ml', 'Malayalam'),  ('mt', 'Maltese'),
('mi', 'Māori'), ('mr', 'Marathi (Marāṭhī)'), ('mh', 'Marshallese'), ('mn', 'Mongolian'), ('na', 'Nauru'),
('nv', 'Navajo, Navaho'), ('nb', 'Norwegian Bokmål'), ('nd', 'North Ndebele'), ('ne', 'Nepali'), ('ng', 'Ndonga'),
('nn', 'Norwegian Nynorsk'), ('no', 'Norwegian'), ('ii', 'Nuosu'), ('nr', 'South Ndebele'), ('oc', 'Occitan'),
('oj', 'Ojibwe, Ojibwa'), ('cu', 'Old Church Slavonic'), ('om', 'Oromo'), ('or', 'Oriya'), ('os', 'Ossetian, Ossetic'),
('pa', 'Panjabi, Punjabi'), ('pi', 'Pāli'), ('fa', 'Persian'), ('pl', 'Polish'), ('ps', 'Pashto, Pushto'),
('pt', 'Portuguese'), ('qu', 'Quechua'), ('rm', 'Romansh'), ('rn', 'Kirundi'), ('ro', 'Romanian'),
('ru', 'Russian'), ('sa', 'Sanskrit (Saṁskṛta)'), ('sc', 'Sardinian'), ('sd', 'Sindhi'), ('se', 'Northern Sami'),
('sm', 'Samoan'),  ('sg', 'Sango'), ('sr', 'Serbian'), ('gd', 'Scottish Gaelic'), ('sn', 'Shona'), ('si', 'Sinhala, Sinhalese'),
('sk', 'Slovak'), ('sl', 'Slovene'), ('so', 'Somali'), ('st', 'Southern Sotho'), ('es', 'Spanish; Castilian'), ('su', 'Sundanese'),
('sw', 'Swahili'), ('ss', 'Swati'), ('sv', 'Swedish'), ('ta', 'Tamil'), ('te', 'Telugu'), ('tg', 'Tajik'), ('th', 'Thai'),
('ti', 'Tigrinya'),  ('bo', 'Tibetan'), ('tk', 'Turkmen'), ('tl', 'Tagalog'), ('tn', 'Tswana'), ('to', 'Tonga'), ('tr', 'Turkish'),
('ts', 'Tsonga'), ('tt', 'Tatar'), ('tw', 'Twi'), ('ty', 'Tahitian'), ('ug', 'Uighur, Uyghur'), ('uk', 'Ukrainian'), 
('ur', 'Urdu'), ('uz', 'Uzbek'), ('ve', 'Venda'),  ('vi', 'Vietnamese'), ('vo', 'Volapük'), ('wa', 'Walloon'), ('cy', 'Welsh'),
('wo', 'Wolof'), ('fy', 'Western Frisian'), ('xh', 'Xhosa'), ('yi', 'Yiddish'), ('yo', 'Yoruba'), ('za', 'Zhuang, Chuang'),
('zu', 'Zulu'),])

def preprocessor(text):
    """
    Given a piece of text, preprocessor will preprocess the text so that it can be classified later.
    """
    
    SYMBOL_REPLACE_RE = re.compile('[\'\"/(){}\[\]\|@,\.\-;]')  # symbols to be replaced.
    
    EXTRA_SPACES = re.compile('  ')
    
    text = text.lower() # lowercase text
    text = re.sub(SYMBOL_REPLACE_RE," ",text)
    text = re.sub(EXTRA_SPACES," ",text)
    return text

def learn_language_weights(total_count_dict, languages):
    """
    This function learns the weights given to the word freuqency in each language. 
    The weight is inversely proportional to the number of words in the corpus for that language
    """
    
    word_frequency_weights = dict()
    min_count = float('inf')
    
    for language in total_count_dict:
        if total_count_dict[language] < min_count:
            min_count = total_count_dict[language]

    for language in languages:
        word_frequency_weights[language] = min_count/total_count_dict[language]
        
    return word_frequency_weights

def learn_word_lang_prob(word_lang_frequency_dict,word_frequency_weights,vocabulary,languages):
    """
    Learns the probability for each (word,language) pair that the word belongs to that language 
    """
    
    num_languages = len(languages)
    
    # takes care of the probabilities of unknown words
    word_lang_log_prob_dict = defaultdict(lambda:1/num_languages) 
    
    for word in vocabulary:
        weighted_total_word_count = 0
        
        for language in languages:
            weighted_total_word_count += word_lang_frequency_dict[(word,language)]*word_frequency_weights[language]
        
        for language in languages:
            
            # smoothing is applied by adding 1 in the denominator and n in the numerator
            
            word_lang_log_prob_dict[(word,language)] = \
            np.log(word_lang_frequency_dict[(word,language)]*word_frequency_weights[language] + 1)
            
            word_lang_log_prob_dict[(word,language)] -= np.log(weighted_total_word_count + num_languages)
    
    return word_lang_log_prob_dict

def train_prob(corpora_location, save_freqs=1, model_name=""):
    """
    The folder corpora_location should contain 1 text file for each language. 
    The name of that corpus file should begin with the ISO code of the language and 
    the next character in the file should be "_"(underscore). 
    """
    
    # Create a dictionary with default value 0
    word_lang_frequency_dict = dict()
    word_global_frequency_dict = dict()
    total_count_dict = dict()
   
    for file in os.listdir(corpora_location):
        
        lang_code = file.split("_")[0]
        
        print("Reading the " + lang_code + " corpus file: " + file)
        
        text = open(corpora_location+'/'+file,"r").read()
        
        text = preprocessor(text)
        
        text = text.split()
                
        total_count_dict[lang_code] = len(text)
        
        for word in text:
            if (word,lang_code) in word_lang_frequency_dict:
                word_lang_frequency_dict[(word,lang_code)] = word_lang_frequency_dict[(word,lang_code)] + 1
            else:
                word_lang_frequency_dict[(word,lang_code)] = 1
            if word in word_global_frequency_dict:
                word_global_frequency_dict[word] = word_global_frequency_dict[word] + 1
            else:
                word_global_frequency_dict[word] = 1
    
    languages = list(total_count_dict.keys())
    
    vocab = list(word_global_frequency_dict.keys())
    
    # Deleting all words with count <= 4 over all corpora
    for word in vocab:
        if word_global_frequency_dict[word]<=4:
            del word_global_frequency_dict[word]
            for language in languages:
                if (word,language) in word_lang_frequency_dict:
                    del word_lang_frequency_dict[(word,language)]
        
    vocab = list(word_global_frequency_dict.keys())
    
    word_frequency_weights = learn_language_weights(total_count_dict, languages)
    
    if save_freqs==1:
        pickle.dump(word_lang_frequency_dict,open(model_name + "_word_lang_freqs.dict","wb"))
        pickle.dump(word_frequency_weights,open(model_name + "_word_freq_weights.dict","wb"))
        pickle.dump(languages,open(model_name + "_languages.list","wb"))
        pickle.dump(vocab,open(model_name + "_vocab.list","wb"))

    word_lang_frequency_dict = defaultdict(lambda:0, word_lang_frequency_dict)
    word_frequency_weights = defaultdict(lambda:0, word_frequency_weights)
    
    word_lang_log_prob_dict = \
    learn_word_lang_prob(word_lang_frequency_dict,word_frequency_weights,vocab,languages)
    
    return word_lang_log_prob_dict,languages
    
def load_model(model_name):
    """
        Loads the set of log probabilities for (word,language) pairs as well as
        the list of languages it was trained on
    """
    # Loading the files using pickle
    word_lang_frequency_dict = pickle.load(open(model_name + "_word_lang_freqs.dict","rb"))
    word_frequency_weights = pickle.load(open(model_name + "_word_freq_weights.dict","rb"))
    languages = pickle.load(open(model_name + "_languages.list","rb"))
    vocab = pickle.load(open(model_name + "_vocab.list","rb"))
    
    # Conveting them into default dictionaryk
    word_lang_frequency_dict = defaultdict(lambda:0, word_lang_frequency_dict)
    word_frequency_weights = defaultdict(lambda:0, word_frequency_weights)
    
    word_lang_log_prob_dict = \
    learn_word_lang_prob(word_lang_frequency_dict,word_frequency_weights,vocab,languages)
    
    print("model successfully loaded")
    
    return word_lang_log_prob_dict,languages
        
    
def prob_document(text, word_lang_log_prob_dict, language):
    """
        Returns the un-normalized probability that the text belongs to the given language. 
        Probability that a document belongs to a language is modelled as the multiplication
        of the probabilities of its constituent words belonging to that language. 
        This method works for languages where words are separated by a delimiter.
    """
    
    log_prob = 0
    for word in text.split():
        log_prob += word_lang_log_prob_dict[(word,language)]
    return log_prob

def language_detector(text,word_lang_log_prob_dict,languages):
    """
        Returns the best-fit language for the text given. The text can be any document - phrase, sentence or paragraph
    """
    
    text = preprocessor(text)
    
    best_log_prob = -1*float("inf")
    
    best_language = None
    
    for language in languages:
        log_prob = prob_document(text, word_lang_log_prob_dict, language)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_language = language
        
    return best_language
    

def main():
    
    print("-"*60)
    
    print("Welcome to Ritika's language detector")
    print("-"*60)
    
    print("Choose one of the options: model \"europarl\" is already supplied with the code which is trained"\
         + " on 90 MB of europarl corpora of different languages of the EU.")
    print("1.Train")
    print("2.Load model and test on stdin")
    print("3.Load model and test on a file which is a single document")
    print("4.Load model and test on a file where each line is a separate sentence")
    
    choice = input()
    if choice == "1":
        print("-"*60)
        
        print("The folder corpora_location should contain 1 text file for each language." +\
        " The name of that corpus file should begin with the ISO code of the language and " +\
        "the next character in the file should be \"_\"(underscore). ")
        
        print("Enter the the folder which contains the corpora to train on")
        corpora_location = input()
        
        print("Enter the name of the model")
        model_name = input()
        
        print("-"*60)
        print("Training started")
        print("-"*60)
        
        word_lang_log_prob_dict,languages = train_prob(corpora_location=corpora_location, save_freqs=1, model_name=model_name)
        
        print("\nTraining over")
        print("-"*60)
        
        print("Do you want to use this model?")
        print("1.Yes")
        print("2.No")
        second_choice = input()
        
        if second_choice == "2":
            exit()
        
        if second_choice == "1":
            print("-"*60)
            print("Select one.")
            print("1.Use model and test on stdin")
            print("2.Use model and test on a file which is a single document")
            print("3.Use model and test on a file where each line is a separate sentence")
            
            second_choice = input()

            if second_choice == "1":
                print("Enter each sentence one by one. Use Ctrl + D to enforce the end of input")
                
                for sentence in sys.stdin:
                    detected_language = iso_lang_codes[language_detector(sentence,word_lang_log_prob_dict,languages)]
                    print("The detected language is:", detected_language)
            
            elif second_choice == "2":
                
                print("Enter the file name which is a single document:")
                file_name = input()
                
                text = open(file_name,"r").read()
                
                detected_language = iso_lang_codes[language_detector(text,word_lang_log_prob_dict,languages)]
                print("The detected language for the document in file is:", detected_language)
                print("Exiting. You can load the saved model to use it again.")
                
            elif second_choice == "3":
                
                print("Enter the file name where each line is a separate sentence")
                
                file_name = input()
                
                text = open(file_name,"r")
                
                lines = text.readlines()
                
                print("Enter a file name where the language tags can be outputted")
                output_file_name = input()
                
                output_file = open(output_file_name,"w")
                
                for line in lines:
                    detected_language = iso_lang_codes[language_detector(line,word_lang_log_prob_dict,languages)]
                    output_file.write("The sentence : " + line)
                    output_file.write("Language : " + detected_language)
                    output_file.write("\n")
                    
                print("Output file saved. Exiting. You can load the saved model to use it again.")
            else:
                print("Wrong choice! Exiting")
                exit()
            
        
    elif (choice == "2" or choice == "3" or choice == "4"):
        
        print("Enter the name of the model")
        
        model_name = input()
        
        word_lang_log_prob_dict,languages = load_model(model_name)
        
        if choice == "2":
            print("Enter each sentence one by one. Use Ctrl + D to enforce the end of input")
                
            for sentence in sys.stdin:
                detected_language = iso_lang_codes[language_detector(sentence,word_lang_log_prob_dict,languages)]
                print("The detected language is:", detected_language)
        
        if choice == "3":
            print("Enter the file name which is a single document:")
            file_name = input()
                
            text = open(file_name,"r").read()
                
            detected_language = iso_lang_codes[language_detector(text,word_lang_log_prob_dict,languages)]
            print("The detected language for the document in file is:", detected_language)
            print("Exiting. You can load the saved model to use it again.")
            
        if choice == "4":
                
            print("Enter the file name where each line is a separate sentence")
                
            file_name = input()
                
            text = open(file_name,"r")
                
            lines = text.readlines()
                
            print("Enter a file name where the language tags can be outputted")
            output_file_name = input()
                
            output_file = open(output_file_name,"w")
                
            for line in lines:
                detected_language = iso_lang_codes[language_detector(line,word_lang_log_prob_dict,languages)]
                output_file.write("The sentence : " + line)
                output_file.write("Language : " + detected_language)
                output_file.write("\n")
                    
            print("Output file saved. Exiting. You can load the saved model to use it again.")
            
    else:
        print("Wrong Choice! Exiting")
        exit()

if __name__ == '__main__':
    main()
    
