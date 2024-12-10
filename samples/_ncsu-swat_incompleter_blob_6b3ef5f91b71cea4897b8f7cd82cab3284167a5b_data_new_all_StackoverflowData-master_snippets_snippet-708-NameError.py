#Source: https://stackoverflow.com/questions/53755893/facing-attributeerror-for-tag-using-spacy-in-python
doc = nlp('can you please help me to install wifi')
for i in doc:
    print (i.text, i.tag_)