## Setting Up
use this line to instal depenencies if it doesnt work try to build the virtual env from the reqirements.txt

pip install -q -U elasticsearch langchain transformers huggingface_hub torch

## Running the model
simply execute all the cell in , if you wish you may change the question by just editing the string with the question

### Preprocessing the document and splitting into fixed size chunks
fix size is easier to debug if something goes bad, i selected a simple text with minimal amount of footers/headers but still needed to clear it from excessive carrige return/new line characters. Decided to leave the quotation marks in the text, as well as some gibberish at the begining end (it is fraction of the total dataset)

### Indexing using ElasticSearch, due to some error problems on their side tryign multiple times

State of the art, easier to scale-up/down (just rent more compute units from the elastisearch) and monitor durin deployment (they have nice dashboard to monitor the progress and various jobs status). For the retrieval model I used ELSER (Elastic Learned Sparse EncodeR) with the newest version 2. Compared to say BERT is better for indexing large amounts of data (like books we are deadlin with).

### Initialize the tokenizer and the model (`google/gemma-2b-it`)

Lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. Since these are pretty new and hot right now I chose them over variation of different old and tested methods like T5, BART, XLNet Albert, RoBERT with text-generation variation implemtation in hugging face. For the parameter tunning, I set the temperature to 0.7 experimentaly it seemed to me to give a bit better results on the dataset. Since it runs localy on my PC weights must be top 8GB in RAM which limits possible models. ON smaller model inference time is also quicker so its simpler to experiment with.

### Evaluation Metrics

For metric I could use supervised metrics (e.g. I would manualy craft 20 questions regardin the text with my answer and they dry to detect keyword in chatboot solution to see if it found them, or even read through them and qualitatively assume responses) and more unsupervised techniques e.g. ensembling couple of models, then prompting them with the same question and then clustering their answers together (assuming that consensus is right) and benchamr against this.

