import os

os.chdir("./stanford-corenlp-full-2018-10-05/")
os.system('java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')
