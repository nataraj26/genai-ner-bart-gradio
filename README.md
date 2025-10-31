## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Text data, especially from the web or internal documents, is vast and unstructured. Manually extracting specific pieces of information, such as names of people (PER), organizations (ORG), locations (LOC), or other miscellaneous entities (MISC), is time-consuming, prone to error, and inefficient.

The problem is to develop an automated system that can process any given text, accurately identify these named entities, and classify them into their predefined categories. This system must also be wrapped in a simple, accessible web interface, allowing non-technical users to easily input text and visualize the model's predictions for testing and validation.
### DESIGN STEPS:


#### STEP 1:

Install Libraries: Install the necessary Python packages: transformers (from Hugging Face for the model), torch (as the backend for the model), and gradio (for the web UI).

Load NER Pipeline: Utilize the transformers library to instantiate an NER pipeline.

Select Model: Specify a pre-trained BART model that has been fine-tuned for NER. A robust and popular choice is dslim/bart-large-ner.

Configure Pipeline: Set the aggregation_strategy="simple" parameter. This ensures that multi-token entities (e.g., "New" and "York") are correctly grouped together as a single entity ("New York") rather than being identified as separate parts

#### STEP 2:

Create Function: Define a Python function (e.g., recognize_entities) that accepts a single string (text) as input. This function will be the core backend logic for the Gradio app.

Model Inference: Inside this function, pass the input text to the NER pipeline loaded in Step 1. The pipeline will return a list of dictionaries, where each dictionary represents a found entity (e.g., {'entity_group': 'LOC', 'word': 'New York', 'start': 30, 'end': 38, ...}).

Format Output: Process this list of entities to make it compatible with Gradio's HighlightedText component. This involves:

Sorting the entities by their start index.

Iterating through the sorted entities and building a new list.

This new list will contain (text, label) tuples, including the text between entities (which will have a None label) and the text of the entities (which will have a PER, LOC, ORG, or MISC label).

#### STEP 3:

Import Gradio: Import the gradio as gr library.

Define UI Components:

Input: Use gr.Textbox() for the user to type or paste their text. Add a placeholder for guidance.

Output: Use gr.HighlightedText() for the output. This component is specifically designed to display text with colored highlights for different labels. Define a color_map to assign specific colors to each entity type (e.g., PER: Red, ORG: Blue).

Instantiate Interface: Create a gr.Interface object, passing it:

fn: The processing function from Step 2 (recognize_entities).

inputs: The input textbox component.

outputs: The output HighlightedText component.

title and description: To provide context for the user.

Launch App: Call the .launch() method on the interface object. This starts a local web server, making the application accessible from any browser.


### PROGRAM:
```python
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

text = ('''The tower is 324 metres (1,063 ft) tall, about the same height
        as an 81-storey building, and the tallest structure in Paris. 
        Its base is square, measuring 125 metres (410 ft) on each side. 
        During its construction, the Eiffel Tower surpassed the Washington 
        Monument to become the tallest man-made structure in the world,
        a title it held for 41 years until the Chrysler Building
        in New York City was finished in 1930. It was the first structure 
        to reach a height of 300 metres. Due to the addition of a broadcasting 
        aerial at the top of the tower in 1957, it is now taller than the 
        Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
        Eiffel Tower is the second tallest free-standing structure in France 
        after the Millau Viaduct.''')

get_completion(text)

import gradio as gr
def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']
    
gr.close_all()
demo = gr.Interface(fn=summarize, inputs="text", outputs="text")
demo.launch(share=True, server_port=int(os.environ['PORT1']))

import gradio as gr

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

gr.close_all()
demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                   )
demo.launch(share=True, server_port=int(os.environ['PORT2']))

```
#### Building a Named Entity Recognition app
```python
API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Andrew, I'm building DeepLearningAI and I live in California"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Andrew and I live in California", "My name is Poli and work at HuggingFace"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```
#### Adding a helper function to merge tokens
```python
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))

gr.close_all()
```

### OUTPUT:



### RESULT:
