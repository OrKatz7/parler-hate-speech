import random
import re
import pandas as pd
from nltk import sent_tokenize
from tqdm import tqdm
from albumentations.core.transforms_interface import DualTransform, BasicTransform
class NLPTransform(BasicTransform):
    LANGS = {
        'en': 'english',
    }

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, self.LANGS.get(lang, 'english'))
    
class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)
        return ' '.join(sentences), lang

class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = []
        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()
            if sentence not in sentences:
                sentences.append(sentence)
        return ' '.join(sentences), lang
    
class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang 
class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang
class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang
class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'https?\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang
class SwapWordsTransform(NLPTransform):
    """ Swap words next to each other """
    def __init__(self, swap_distance=1, swap_probability=0.1, always_apply=False, p=0.5):
        """  
        swap_distance - distance for swapping words
        swap_probability - probability of swapping for one word
        """
        super(SwapWordsTransform, self).__init__(always_apply, p)
        self.swap_distance = swap_distance
        self.swap_probability = swap_probability
        self.swap_range_list = list(range(1, swap_distance+1))

    def apply(self, data, **params):
        text, lang = data
        words = text.split()
        words_count = len(words)
        if words_count <= 1:
            return text, lang

        new_words = {}
        for i in range(words_count):
            if random.random() > self.swap_probability:
                new_words[i] = words[i]
                continue
    
            if i < self.swap_distance:
                new_words[i] = words[i]
                continue
    
            swap_idx = i - random.choice(self.swap_range_list)
            new_words[i] = new_words[swap_idx]
            new_words[swap_idx] = words[i]

        return ' '.join([v for k, v in sorted(new_words.items(), key=lambda x: x[0])]), lang
class CutOutWordsTransform(NLPTransform):
    """ Remove random words """
    def __init__(self, cutout_probability=0.05, always_apply=False, p=0.5):
        super(CutOutWordsTransform, self).__init__(always_apply, p)
        self.cutout_probability = cutout_probability

    def apply(self, data, **params):
        text, lang = data
        words = text.split()
        words_count = len(words)
        if words_count <= 1:
            return text, lang
        
        new_words = []
        for i in range(words_count):
            if random.random() < self.cutout_probability:
                continue
            new_words.append(words[i])

        if len(new_words) == 0:
            return words[random.randint(0, words_count-1)], lang

        return ' '.join(new_words), lang
class AddNonToxicSentencesTransform(NLPTransform):
    """ Add random non toxic statement """
    def __init__(self, non_toxic_sentences, sentence_range=(1, 3), always_apply=False, p=0.5):
        super(AddNonToxicSentencesTransform, self).__init__(always_apply, p)
        self.sentence_range = sentence_range
        self.non_toxic_sentences = non_toxic_sentences

    def apply(self, data, **params):
        text, lang = data

        sentences = self.get_sentences(text, lang)
        for i in range(random.randint(*self.sentence_range)):
            sentences.append(random.choice(self.non_toxic_sentences))
        
        random.shuffle(sentences)
        return ' '.join(sentences), lang 
    
import albumentations
def get_train_transforms():
    return albumentations.Compose([
        ExcludeDuplicateSentencesTransform(p=0.9),
        ShuffleSentencesTransform(p=0.8),
        ExcludeNumbersTransform(p=0.8),
        ExcludeHashtagsTransform(p=0.5),
        ExcludeUsersMentionedTransform(p=0.9),
        ExcludeUrlsTransform(p=0.9),
        CutOutWordsTransform(p=0.1),
        SwapWordsTransform(p=0.1),
    ])