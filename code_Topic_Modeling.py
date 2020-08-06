#importing different packages
import re , os
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def format_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    # main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else: break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def compute_coherence(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


path = '   '                                                     # Path of the directory where complete dataset is stored
dirs = sorted(os.listdir(path))
for f in dirs:
    dirr = sorted(os.listdir(path+"\\"+f))
    for fil in dirr:
        if str(fil) == str(f+".csv") : continue
        ua=pd.read_csv( path+"\\"+f+"\\"+fil ,names=["no","id","user","date","tweet","none1","place1"])
        u= ua["tweet"]

        data = u.values.tolist()                                  # converting each tweet into list
        data = [re.sub('\s+', ' ', sent) for sent in data]        # Removing new line characters
        data = [re.sub("\'", "", sent) for sent in data]          # Removing unwanted single quotes
        data_words = list(sent_to_words(data))                    # tokenizing words

        # Building the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        data_words_nostops = remove_stopwords(data_words)         # Removing Stop Words from text
        data_words_bigrams = make_bigrams(data_words_nostops)     # Forming Bigrams
        nlp = spacy.load('en', disable=['parser', 'ner'])
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])     #lemmatizing


        id2word = corpora.Dictionary(data_lemmatized)             # Creating the Dictionary
        texts = data_lemmatized                                   # Creating Corpus
        corpus = [id2word.doc2bow(text) for text in texts]


        mallet_path = '  '                                        # path to  mallet-2.0.8/bin/mallet
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=6, id2word=id2word)
        print(ldamallet.show_topics(formatted=False))             # topics
        coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('\nCoherence Score: ', coherence_ldamallet)         # printing coherence score


        # Finding Dominent topic in each tweet
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,num_topics=6,random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
        model_list, coherence_values = compute_coherence(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
        optimal_model = model_list[3]
        model_topics = optimal_model.show_topics(formatted=False)
        df_topic_sents_keywords = format_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']


        # Finding topic distribution across tweets
        topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()         # Number of tweets for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)                  # Percentage of tweets for Each Topic
        topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
        df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
        
