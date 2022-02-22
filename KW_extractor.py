from keybert import KeyBERT
import RAKE
import numpy as np

class keybert_extractor():
    def __init__(self, model='bert-base-uncased'):
        """
        this class is used to extract key words using keybert model
        :param model:
        """
        self.kb = KeyBERT(model=model)

    def sentence_process(self,txt,k=1):
        """
        in case of extracting k words for each sentence
        :param txt: the full movie plot
        :param k: number of words to extract for each sentence
        :return: the extract function that will extract k kw per sentence
        """
        split = txt.split('.')
        n=len(split)
        return self.extract(split, n*[k])

    def parts_process(self,txt, p): ##TODO: complete the function
        sentences = txt.split('. ')
        n = len(sentences)
        tmp_n = (n-2)//(p-1)
        split = [sentences[0], sentences[1]]
        if tmp_n>0:
            split += ['. '.join(sentences[2+i*tmp_n:2+(i+1)*tmp_n]) for i in range(p-2)]
        split += ['. '.join(sentences[2+(p-2)*tmp_n:])]
        n_list = [(1,1,1), (1,1,1)]
        if n<=15:
            x = int(np.ceil(tmp_n*0.8))
            n_list += [(x if x > 0 else 1, 1, 1)] * (len(split) - 2)
        else:
            x = int(np.floor(tmp_n * 0.3))
            n_list += [(x if x > 0 else 1, 1, 1)] * (len(split) - 2)
        return self.extract(split, n_list)


    def extract(self,txt_list,n_list):
        """
        :param txt_list: list of texts to extracts kw fromm
        :param n_list:  number of key words to extract for every item in txt_list (len(n_list)==len(txt_list))
        :return: list of keywords for all texts in txt_list
        """
        kw = []
        for txt,n in zip(txt_list,n_list):
            kw += [x[0] for x in self.kb.extract_keywords(txt, top_n=n)]
        return kw

class Rake_extractor():
    def __init__(self, stopwords = RAKE.SmartStopList()):
        """
        this class is used to extract key words using RAKE
        :param stopwords:
        """
        self.rake_obj = RAKE.Rake(stopwords)

    def sentence_process(self,txt,k=1):
        """
        in case of extracting k words for each sentence
        :param txt: the full movie plot
        :param k: number of words to extract for each sentence
        :return: the extract function that will extract k kw per sentence
        """
        split = txt.split('.')
        n=len(split)
        return self.extract(split, n*[(k,1,1)])

    def thirds_process(self,txt): ##TODO: complete the function
        sentences = txt.split('.')
        n = len(sentences)

    def parts_process(self,txt, p): ##TODO: complete the function
        sentences = txt.split('. ')
        n = len(sentences)
        tmp_n = (n-2)//(p-1)
        split = [sentences[0], sentences[1]]
        if tmp_n>0:
            split += ['. '.join(sentences[2+i*tmp_n:2+(i+1)*tmp_n]) for i in range(p-2)]
        split += ['. '.join(sentences[2+(p-2)*tmp_n:])]
        n_list = [(1,1,1), (1,1,1)]
        if n<=15:
            x = int(np.ceil(tmp_n*0.8))
            n_list += [(x if x>0 else 1, 1,1)]*(len(split)-2)
        else:
            x=int(np.floor(tmp_n*0.3))
            n_list += [(x if x>0 else 1, 1,1)]*(len(split)-2)
        return self.extract(split, n_list)

    def extract(self,txt_list,n_list):
        """
        FIXME
        :param txt_list: list of texts to extracts kw fromm
        :param n_list:  number of key words to extract for every item in txt_list (len(n_list)==len(txt_list))
        :return: list of keywords for all texts in txt_list
        """
        kw = []
        for txt,n in zip(txt_list,n_list):
            k=n[0]
            kw_tmp = self.rake_obj.run(txt, minCharacters=3, maxWords=n[1],minFrequency=n[2])
            kw_tmp = sorted(kw_tmp, key=lambda x: x[1])
            kw += [x[0] for x in kw_tmp[-k:]]
        return ', '.join(kw)

