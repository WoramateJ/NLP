import json
import io
import glob
import os

wordlist = []
Unigrams = {}
Vocab = set()
path = '/Users/aimpitcha/cpe/cpe3-2/NLP/20news-bydate/training_data'
for filename in glob.glob(os.path.join(path, '*.json')):
    with open(filename) as json_data:
        data = json.load(json_data, strict=False)
        for sentence in data["sentences"]:
            for wordlist in sentence["tokens"]:
                word = wordlist["originalText"]
                if word in Unigrams:
                    Unigrams[word] += 1
                    if Unigrams[word] > 5:
                        Vocab.add(word)
                else:
                    Unigrams[word] = 1
        sortedVocab = sorted(Vocab)
        output_file = io.open('vocabulary.txt', 'w', encoding='utf8')
        for word in sortedVocab:
          #  count = Unigrams[unigram]
            output_file.write(word + '\n')
        output_file.close()

#if want to improve, add vocab type set
