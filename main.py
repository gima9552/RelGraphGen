import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import requests
import json
from json.decoder import JSONDecodeError
import re
import torch
from sentence_transformers import SentenceTransformer
from matplotlib.colors import colorConverter, Colormap
from pyvis.network import Network
import webbrowser
import os
from openai import OpenAI
from config import *  # OPERNROUTER_API_KEY, MODEL_LIST, GRAPHS, HISTORY_CONVS, HISTORY_LOGS



client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

# Embedding model for graph node comparison/convergence

model_name_or_path = "infgrad/Jasper-Token-Compression-600M"
embed_model = SentenceTransformer(
    model_name_or_path,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",  #flash_attention_2; sdpa; eager
        "trust_remote_code": True
        },
    trust_remote_code=True,
    tokenizer_kwargs={"padding_side": "left"},
    device="cpu",
    )


# Prompt types for causal bond extraction and LLM choir integration

prompt = lambda abst: f""" Your task is to extract explanations from text by identifying causal, contrastive, and compositional relationships. Return all explanatory bonds in this JSON format:

{{
  “document”: [
    {{
      “bond”: “Type of explanatory bond”,
      “span_1": “Origin concept/cause (or case/outcome A for contrastive; or whole/entity for compositional)“,
      “span_2”: “Endpoint concept/effect (or case/outcome B for contrastive; or part/constituent for compositional)“,
      “gloss”: “One sentence explaining why this bond matters for understanding the text.”
    }}
  ]
}}

A span is a short phrase copied from the input text (it does not need to be a full sentence). Keep spans minimal but sufficient.

Centrality rule (salience selection)

Extract only bonds that are salient for understanding the document’s main explanatory purpose. Prefer the explanatory backbone over exhaustive coverage:
	•	Prefer bonds that express the main thesis/lede, recurring themes, or necessary links in a causal chain.
	•	Avoid minor examples, colorful illustrations, local details, or background facts unless they are needed to understand the main claims.

Span rules (minimal)
	•	Resolve pronouns/anaphora when needed (“this”, “they”, “it”) to the intended referent.
	•	Prefer noun-phrase-like spans where possible; avoid full clauses unless needed for meaning.
	•	Light reformulation is allowed for clarity, but keep key terminology and do not invent information.
	•	Use consistent terminology across bonds (avoid unnecessary paraphrase variants for the same concept).

Gloss rules (one sentence, grounded)
	•	Write exactly one sentence per bond.
	•	The gloss should explain the role of the bond in the document’s explanatory arc (why the reader needs this relation).
	•	Do not introduce new facts not supported by the text.
	•	Do not quote the text verbatim; paraphrase at a high level.
	•	Keep it concise (aim ~12–25 words).

Bond types
	•	Strong Causation: Explains a phenomenon by specifying explicit causal mechanisms or multiple steps in a causal chain. The underlying process is made clear.
Example: “You are jittery because you drank way too much coffee this morning. That’s why you dropped the mug.”
	•	Weak Causation: Explains through correlational or probabilistic relationships between variables without specifying the underlying mechanism. Often involves associations, tendencies, or indirect influences.
Example: “Diseases with uncertain causes were about 50% more likely to attract religious or magical treatments.”
	•	Contrastive: Explains a phenomenon by highlighting differences between alternative conditions or cases (A vs B).
Example: “In the treatment group, 70% recovered; in the placebo group, only 30% recovered.”
Important rule for contrastive extraction: If the text is phrased as a comparison (e.g., “whereas”, “in contrast”, “compared with”) without explicit causal language, extract only one Contrastive bond and do not split it into two causal bonds.
	•	Compositional: States what something is made of, contains, consists of, or is defined as (whole/entity → part/definition).
Typical markers: “is made of”, “consists of”, “contains”, “is composed of”, “is defined as”, “is a type of”.
Example: “Protons and neutrons are combinations of even tinier particles, called quarks.”
Important rule: Do not treat “is made of / consists of / is defined as” statements as causation unless the text explicitly frames them as a causal process. When they describe constitution/definition, label them as Compositional.
	•	No Bond: The text contains no causal, contrastive, or compositional relationships.

Do not output any bond types beyond the ones listed.

Examples

Input

Also known as the aurora borealis in the north, (or aurora australis in the Southern Hemisphere) these night sky events get their start on the sun’s surface after coronal mass ejections (CMEs) spew ionized clouds of high energy particles towards Earth. The radiation then interacts with the planet’s magnetosphere and generates the vivid colors in Earth’s atmosphere–as well as the occasional electrical grid and satellite array headache.

Output

{{
  “document”: [
    {{
      “bond”: “Strong Causation”,
      “span_1”: “coronal mass ejections (CMEs) spew ionized clouds of high energy particles towards Earth”,
      “span_2”: “aurora borealis”,
      “gloss”: “This links the solar event to the origin of auroras, establishing the document’s main physical explanation.”
    }},
    {{
      “bond”: “Strong Causation”,
      “span_1”: “radiation then interacts with the planet’s magnetosphere”,
      “span_2”: “generates the vivid colors in Earth’s atmosphere”,
      “gloss”: “This supplies the mechanism that turns incoming particles into the observable atmospheric light display.”
    }}
  ]
}}


Input

Argument mining is a subfield of NLP focused on identifying claims and premises in text. Typical systems label spans as claims, evidence, or non-argumentative content. Recent work explores transformer-based architectures and benchmark datasets for evaluation.

Output

{{
  “document”: [
    {{
      “bond”: “No Bond”,
      “span_1": “”,
      “span_2": “”,
      “gloss”: “The passage defines a topic area but does not describe causal, contrastive, or compositional explanatory relations.”
    }}
  ]
}}


Input

Patients who received reminder texts were more likely to attend their follow-up appointment than patients who did not. This may be due to the reminders reducing forgetfulness and making scheduling feel more urgent.

Output

{{
  “document”: [
    {{
      “bond”: “Contrastive”,
      “span_1”: “patients who received reminder texts were more likely to attend their follow-up appointment”,
      “span_2”: “patients who did not”,
      “gloss”: “This establishes the key comparison that motivates the explanation of why outcomes differ between groups.”
    }},
    {{
      “bond”: “Weak Causation”,
      “span_1”: “the reminders reducing forgetfulness”,
      “span_2”: “more likely to attend their follow-up appointment”,
      “gloss”: “This offers a plausible contributing factor that could account for higher attendance without proving a mechanism fully.”
    }},
    {{
      “bond”: “Weak Causation”,
      “span_1”: “making scheduling feel more urgent”,
      “span_2”: “more likely to attend their follow-up appointment”,
      “gloss”: “This adds another plausible influence that helps explain the attendance increase in the reminder group.”
    }}
  ]
}}



Perform the task with the following input.

Input: {abst}

Output: """


prompt_choir = lambda abst, responses: f"""You are evaluating different attempts at retrieving explanations from text by identifying causal, contrastive, and compositional relationships. The original text is the following document:

**Document:** {abst}

Here are the responses from different models (anonymized) as lists of JSONs:

{responses}

Your task is to evaluate each response individually.

For each response, explain what it does well and what it does poorly, referring to the following definitions for explanatory bonds: 
•	Strong Causation: Explains a phenomenon by specifying explicit causal mechanisms or multiple steps in a causal chain. The underlying process is made clear.
Example: “You are jittery because you drank way too much coffee this morning. That’s why you dropped the mug.”
	•	Weak Causation: Explains through correlational or probabilistic relationships between variables without specifying the underlying mechanism. Often involves associations, tendencies, or indirect influences.
Example: “Diseases with uncertain causes were about 50% more likely to attract religious or magical treatments.”
	•	Contrastive: Explains a phenomenon by highlighting differences between alternative conditions or cases (A vs B).
Example: “In the treatment group, 70% recovered; in the placebo group, only 30% recovered.”
Important rule for contrastive extraction: If the text is phrased as a comparison (e.g., “whereas”, “in contrast”, “compared with”) without explicit causal language, extract only one Contrastive bond and do not split it into two causal bonds.
	•	Compositional: States what something is made of, contains, consists of, or is defined as (whole/entity → part/definition).
Typical markers: “is made of”, “consists of”, “contains”, “is composed of”, “is defined as”, “is a type of”.
Example: “Protons and neutrons are combinations of even tinier particles, called quarks.”
Important rule: Do not treat “is made of / consists of / is defined as” statements as causation unless the text explicitly frames them as a causal process. When they describe constitution/definition, label them as Compositional.
	•	No Bond: The text contains no causal, contrastive, or compositional relationships.

You then have to provide a final ranking.

IMPORTANT: Your ranking MUST be formatted in JSON, and MUST be formatted EXACTLY as follows:
{{"rankings":[{{"answer": one of the answers, "points": a number ranging from 0 (very poor answer) to 10 (perfect answer) , "thinking": the reasoning behind your jusdgement}}, ... ]}}

Example of the correct format for your ENTIRE response:

{{"rankings":[{{"answer":"Reponse A", "points":2, "thinking":"provides good detail on X but misses Y and Z..."}}, {{"answer":"Reponse B", "points":5, "thinking":"is accurate but lacks depth on Z..."}}, {{"answer":"Reponse C", "points":10, "thinking":"is the most comprehensive answer"}}]}}

Now provide your response:"""

######################
# Functions Definition

# Model running to get single query reponse and store result for council
# Returns a list

def multiLLM_run(list_of_models, document, prompt):
  results = []
  errors_log = ""
  query_id = ''.join(e for e in document[:8] if e.isalnum())
  convo = "############\n### USER ###\n############\n"+prompt(document)
  for mod in list_of_models:
    selected_model = mod
    modname_results = selected_model.split("/")[1]
    hdr = "#"*(len(modname_results)+8)+"\n"
    bdy = "### "+modname_results+" ###\n"
    completion = client.chat.completions.create(
      model=selected_model,
      seed=77777,
      temperature=0.0,
      max_tokens=1028,
      response_format={
          "type": "json_schema",
          "json_schema": {
              "name": "bond",
              "strict": True,
              "schema": {
                  "type": "object",
                  "properties": {
                      "document": {
                          "type": "array",
                          "description": "List of explanatory bonds in the document",
                          "items": {
                              "type": "object",
                              "properties": {
                                  "bond": {
                                      "type": "string",
                                      "description": "Type of explanatory bond"
                                      },
                                  "span_1": {
                                      "type": "string",
                                      "description": "Origin concept/cause (or case/outcome A for contrastive; or whole/entity for compositional)"
                                      },
                                  "span_2": {
                                      "type": "string",
                                      "description": "Endpoint concept/effect (or case/outcome B for contrastive; or part/constituent for compositional)"
                                      },
                                  "gloss": {
                                      "type": "string",
                                      "description": "One sentence explaining why this bond matters for understanding the text."
                                      }
                                  },
                              "required": ["bond", "span_1", "span_2", "gloss"],
                              "additionalProperties": False
                              }
                          }
                      },
                  "required": ["document"],
                  "additionalProperties": False
                  },
              },
      },
      messages=[
        {
          "role": "user",
          "content": prompt(document)
        }
      ]
    )
    val = completion.choices[0].message.content
    current_out = hdr+bdy+hdr+str(val)
    convo += current_out+"\n"
    u = list()
    try:
      v = json.loads(val)
      u.extend(v["document"])
    except JSONDecodeError:
      errors_log += current_out+"\n\n\n"
    results.append({"model":modname_results, "results":u})
  with open(HISTORY_LOGS+"/errlog_"+query_id+".txt", "a") as f:
      f.write(errors_log)
  with open(HISTORY_CONVS+"/"+query_id+".txt", "a") as f:
      f.write(convo)
  return results

def llm_choir(list_of_models, document, results, prompt_choir):
  errors_log = ""
  query_id = ''.join(e for e in document[:8] if e.isalnum())
  labels = [chr(65 + i) for i in range(len(results))]  # A, B, C, ...
  # Create mapping from label to model name
  label_to_model = {f"Response {label}": result['model'] for label, result in zip(labels, results)}
  responses = "\n\n".join([f"Response {label}:\n{result['results']}" for label, result in zip(labels, results)])
  convo = "############\n### USER ###\n############\n"+prompt_choir(document,responses)
  model_rankings = {result['model']:0 for result in results}
  for mod in list_of_models:
    selected_model = mod
    modname_results = selected_model.split("/")[1]
    hdr = "#"*(len(modname_results)+8)+"\n"
    bdy = "### "+modname_results+" ###\n"
    completion = client.chat.completions.create(
      model=selected_model,
      seed=77777,
      temperature=0.0,
      max_tokens=1028,
      response_format={
          "type": "json_schema",
          "json_schema": {
              "name": "ranking",
              "strict": True,
              "schema": {
                  "type": "object",
                  "properties": {
                      "rankings": {
                          "type": "array",
                          "description": "List of answers with points",
                          "items": {
                              "type": "object",
                              "properties": {
                                  "answer": {
                                      "type": "string",
                                      "description": "Label for the evaluated answer"
                                      },
                                  "points": {
                                      "type": "number",
                                      "description": "Grade of the answer from 0 (poor) to 10 (perfect)"
                                      },
                                  "thinking": {
                                      "type": "string",
                                      "description": "Reasoning behind the judgement"
                                      }
                                  },
                              "required": ["answer", "points", "thinking"],
                              "additionalProperties": False
                              }
                          }
                      },
                  "required": ["rankings"],
                  "additionalProperties": False
                  },
              },
      },
      messages=[
        {
          "role": "user",
          "content": prompt_choir(document, responses)
        }
      ]
    )
    val = completion.choices[0].message.content
    current_out = hdr+bdy+hdr+str(val)
    convo += current_out+"\n"
    try:
      v = json.loads(val)
      ranks = v["rankings"]
      for i in ranks:
        model_rankings[label_to_model[i["answer"]]] += int(i["points"])     
    except JSONDecodeError:
      errors_log += current_out+"\n\n\n"
  best_model = sorted(model_rankings, key=model_rankings.get, reverse=True)[0]
  print(model_rankings, f"\nBest performing model: {best_model}")
  with open(HISTORY_LOGS+"/errlog_"+query_id+".txt", "a") as f:
      f.write(errors_log)
  with open(HISTORY_CONVS+"/"+query_id+".txt", "a") as f:
      f.write(convo)
  for i in results:
    if i['model'] == best_model:
      return i['results'], query_id

def single_graphmaker_html(results, query_id):
  colordict = {'Strong Causation':'red', 'Weak Causation':'orange', 'Contrastive':'cyan', 'Compositional':'green'}
  graph_dict = dict()
  bondinfo = []
  for i in results:
    if i["bond"] != "No Bond":
      obj = [i["span_1"], i["span_2"], {'color':colordict[i["bond"]]}]
      bondinfo.append(obj)
  for i in range(1, len(bondinfo)):
    it0 = bondinfo[i-1][1]
    it1 = bondinfo[i][0]
    embeddings = embed_model.encode([it0,it1], normalize_embeddings=True, compression_ratio=0.6666)
    if float(embed_model.similarity(embeddings[0], embeddings[1])) >= 0.75:
      bondinfo[i][0] = it0
  DCG = nx.DiGraph()
  # nodes in a graph become a set, so repetition is not an issue
  nodes = [x[:2] for x in bondinfo]
  nodelist = list()
  for x in nodes:
    nodelist.extend(x)
  DCG.add_nodes_from(nodelist)
  # adding edges, saving in dict for later JSON
  DCG.add_edges_from([tuple(x) for x in bondinfo])
  colours = [DCG[u][v]['color'] for u,v in DCG.edges]
  graph_dict[query_id] = [tuple(x) for x in bondinfo]
  # conversion from NetworkX graph to PyVis visualizer
  nt = Network('1080px', '1080px', notebook=True, directed=True)
  nt.from_nx(DCG)
  nt.toggle_physics(False)
  nt.save_graph(GRAPHS+"/"+query_id+".html")
  with open(query_id+".json", "w") as fout:
    json.dump(graph_dict, fout)

######################
# Running

if __name__ == "__main__":
  f = input("Please write the name of the file to examine:   ")
  document = ""
  with open(f, "r") as fin:
    document = fin.read()
  print("Running the causal-bond function...")
  results = multiLLM_run(MODEL_LIST, document, prompt)
  print("Done!")
  print("Running the LLM choir function...")
  r, q = llm_choir(MODEL_LIST, document, results, prompt_choir)
  print("Done!")
  print("Running the graph maker...")
  single_graphmaker_html(r, q)
  print("Done!")
  filename = 'file:///'+os.getcwd()+'/'+GRAPHS+"/"+q+".html"
  webbrowser.open_new_tab(filename)
