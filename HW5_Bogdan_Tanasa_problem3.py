#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Problem 3.1: Using the Huggingface’s Transformer library and the ‘sentiment-analysis’ pipeline, 
#              analyze the sentiments of the following sentences.
# I like NLP course.
# I hate when my computer crashes.


# In[2]:


from transformers import pipeline
import tensorflow


# In[3]:


classifier = pipeline("sentiment-analysis",  model="distilbert-base-uncased-finetuned-sst-2-english", 
                                             device = -1,  framework="pt")


# In[5]:


sentences = ["I like NLP course.", "I hate when my computer crashes."]
sentiment_results = classifier(sentences)

for sentence, result in zip(sentences, sentiment_results):
    print(f"Sentence: '{sentence}' :: Sentiment: {result}")


# In[ ]:





# In[6]:


# Problem 3.2 : Using the Huggingface’s Transformer library and ‘zero-shot-classification’ pipeline,
#               classify the following sentence in one of the three given classification categories.
# Text: Los Angeles Clippers is a good basketball team
# Given Classification categories = sports, politics, education


# In[9]:


from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "Los Angeles Clippers is a good basketball team"
labels = ["sports", "politics", "education"]

classification_results = classifier(text, labels)


# In[12]:


print(f"Text: '{text}'")
print(f"Labels: '{labels}'")
print("Analysis Results:")
for label, score in zip(classification_results["labels"], classification_results["scores"]):
    print(f"  {label}: {score:.4f}")


# In[ ]:





# In[13]:


# Problem 3.3: Using the Huggingface’s Transformer library and ‘text-generation’ pipeline, complete the following sentence.
# In this month, the stock market will


# In[14]:


from transformers import pipeline

classifier3 = pipeline("text-generation", 
                          model="gpt2", 
                          device = -1,  
                          framework="pt")

sentence = "In this month, the stock market will"


# In[15]:


generated_sentence = classifier3(sentence, max_length=50, num_return_sequences=1)

print(f"Prompt: '{sentence}'\n Newly Generated Text: {generated_sentence[0]['generated_text']}")


# In[ ]:





# In[16]:


# Problem 3.4: Using the Huggingface’s Transformer library and ‘fill-mask’ pipeline, fill in the blanks.
# Math course will teach you about <mask> topics


# In[17]:


from transformers import pipeline

fillings = pipeline("fill-mask", model="bert-base-uncased")
masked_sentence = "Math course will teach you about [MASK] topics"
fillings_results = fillings(masked_sentence, top_k=5)  

print("Possible text:")
for result in fillings_results:
    print(f"{result['sequence']} (score: {result['score']:.4f})")


# In[ ]:





# In[18]:


# Problem 3.5: Using the Huggingface’s Transformer library and ‘ner’ (Name Entity Recognition) pipeline, identify name, 
# organization, and place.
# Tim Cook is the CEO of Apple located in San Jose


# In[19]:


from transformers import pipeline

ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

sentence = "Tim Cook is the CEO of Apple located in San Jose."
ner_model_results = ner_model(sentence)

print("Identified Components:")
for component in ner_model_results:
    print(f"{component['word']}: {component['entity_group']} (score: {component['score']:.4f})")


# In[ ]:





# In[20]:


# Problem 3.6: Using the Huggingface’s Transformer library and ‘question-answering’ pipeline, let the system find the answer 
# to the following question in the given context.
# Question: In which state Los Angeles located
# Context: Los Angeles is in California


# In[21]:


from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

question = "In which state is Los Angeles located?"
context = "Los Angeles is in California."

resultsqa = qa(question=question, context=context)

print(f"Question: '{question}'")
print(f"Answer: {resultsqa['answer']}")


# In[ ]:





# In[22]:


# Problem 3.7: Using the Huggingface’s Transformer library and ‘summarize’ pipeline, summarize the following text.
# Text:
# Australia was celebrated for its initial response to the Covid-19 pandemic,
# and for getting its economy more or less back on track long ago. But with
# that security has come complacency, particularly in the federal government,
# which failed to secure enough vaccine doses to prevent the regular "circuit
# breaker" lockdowns that come every time a handful of cases emerge, or even
# the longer restrictions that Sydney is experiencing now. Australia's
# borders, controlled by strict quarantine measures, have been all but shut
# for more than a year. Now Australians, who basked in their early successes,
# are wondering how much longer this can go on.


# In[24]:


from transformers import pipeline

summarizing = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Australia was celebrated for its initial response to the Covid-19 pandemic,
and for getting its economy more or less back on track long ago. But with
that security has come complacency, particularly in the federal government,
which failed to secure enough vaccine doses to prevent the regular "circuit
breaker" lockdowns that come every time a handful of cases emerge, or even
the longer restrictions that Sydney is experiencing now. Australia's
borders, controlled by strict quarantine measures, have been all but shut
for more than a year. Now Australians, who basked in their early successes,
are wondering how much longer this can go on.
"""

summary_results = summarizing(text, max_length=50, min_length=25, do_sample=False)
print("Summary:")
print(summary_results[0]['summary_text'])


# In[ ]:




