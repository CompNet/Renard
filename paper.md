---
title: "Renard: A Modular Pipeline for Extracting Character Networks from Narrative Texts"
tags:
  - Python
  - character networks
  - pipeline
  - nlp
authors:
  - name: Arthur Amalvy
    orcid: 0000-0003-4629-0923
	affiliation: 1
  - name: Vincent Labatut
	orcid: 0000-0002-2619-2835
	affiliation: 1
  - name: Richard Dufour
	orcid: 0000-0003-1203-9108
	affiliation: 2
affiliations:
  - name: Laboratoire Informatique d'Avignon
	index: 1
  - name: Laboratoire des Sciences du Numérique de Nantes
	index: 2
date: 29 February 2024
bibliography: paper.bib
---

# Summary

Renard (*Relationships Extraction from NARrative Documents*) is a Python library that allows to define custom natural language processing (NLP) pipelines to extract character networks from narrative texts. Contrarily to the few existing tools, Renard can extract *static* as well as *dynamic* networks. Renard pipelines are modular: the user can choose the implementation of each NLP subtask needed to extract a character network. This allows to specialize pipelines to particular types of texts and to study the impact of each subtask on the extracted network.

# Statement of Need

Character networks (that is, graphs where nodes represent characters and edges represent their relationships) extracted from narrative texts are useful in a number of applications, from visualization to literary analysis [@labatut-2019]. There are different ways of modeling relationships (co-occurrences, conversations, actions...), and networks can be static or dynamic, which allows extracting different kinds of networks depending on the targeted applications. While some authors extract these networks by relying on manually annotated data [@rochat-2014-phd_thesis; @rochat-2015-character_networks_zola; @rochat-2017; @park-2013-character_networks; @park-2013-character_networksb], it is a time-costly endeavor, and the fully automatic extraction of these networks is therefore of interest. Unfortunately, there are only a few existing software and tools that can extract character networks [@marazzato-2014-chaplin; @metrailler-2023-charnetto], and none of these can output dynamic networks. Furthermore, automatically extracting a character network requires solving several successive natural language processing tasks, such as named entity recognition (NER) or coreference resolution, and algorithms carrying these tasks are bound to make errors. To our knowledge, the cascading impact of these errors on the quality of the extracted networks has yet to be studied extensively. This is an important issue since knowing which tasks have more influence on the extracted networks would allow prioritizing research efforts.

Renard is a fully configurable pipeline that can extract static and dynamic networks from narrative texts. We design it so that it is as modular as possible, which allows the user to select the implementation of each extraction step as needed. This has several advantages:

1. The pipeline can be specialized for a specific type of texts, allowing for better performance.
2. The pipeline can easily incorporate new advances in NLP, by simply implementing a new step when necessary.
3. One can study the impact of the performance of each step on the quality of the extracted networks.

# Design and Main Features

Renard is centered about the concept of *pipeline*. In Renard, a pipeline is a series of sequential *steps* that are run one after the other in order to extract a character network from a text. When using Renard, the user simply *describes* this pipeline in Python by specifying this series of steps, and can apply it to different texts afterwards.

```python
from renard.pipeline import Pipeline
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import NLTKNamedEntityRecognizer
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

with open("./my_doc.txt") as f:
	text = f.read()

pipeline = Pipeline(
	[
		NLTKTokenizer(),
		NLTKNamedEntityRecognizer(),
		GraphRulesCharacterUnifier(min_appearance=10),
		CoOccurrencesGraphExtractor(co_occurrences_dist=25)
	]
)

out = pipeline(text)
```

To allow for custom needs, Renard is designed to be very flexible. If a step is not available in Renard, we encourage users to either:

- Externally perform the computation corresponding to the desired step, and inject the results back into the pipeline,
- Integrate their custom processing into Renard by implementing their own step by subclassing the existing `PipelineStep` class.


The flexibility of this approach introduces the possibility of creating invalid pipelines because steps often require information computed by previously ran steps: for example, solving the NER task requires a tokenized version of the input text. To counteract this issue, each step therefore declares its requirements and the new information it produces, which allows Renard to check whether a pipeline is valid, and to explain at runtime to the user why it may not be.

| Task                   | Step                                | Supported Languages                                                                      |
|------------------------|-------------------------------------|------------------------------------------------------------------------------------------|
| Preprocessing          | `CustomSubstitutionPreprocessor`    | any                                                                                      |
| Tokenization           | `NLTKTokenizer`                     | eng, ces, dan, nld, est, fin, fra, deu, ell, ita, nor, pol, por, rus, slv, spa, swe, tur |
| Quote Detection        | `QuoteDetector`                     | any                                                                                      |
| NER                    | `NLTKNamedEntityRecognizer`         | eng, rus                                                                                 |
|                        | `BertNamedEntityRecognizer`         | eng, fra                                                                                 |
| Coreference Resolution | `BertCoreferenceResolver`           | eng                                                                                      |
|                        | `SpacyCorefereeCoreferenceResolver` | eng                                                                                      |
| Character Unification  | `NaiveCharacterUnifier`             | any                                                                                      |
| Character Unification  | `GraphRulesCharacterUnifier`        | eng, fra                                                                                 |
| Speaker Attribution    | `BertSpeakerDetector`               |                                                                                          |
| Graph Extraction       | `CoOccurencesGraphExtractor`        | any                                                                                      |
|                        | `ConversationalGraphExtractor`      | any                                                                                      |
\label{tab:languages}

Renard lets the user select the targeted language of its custom pipeline. A pipeline can be configured to run in any language, as long as each of its steps supports it. Table \ref{tab:languages} shows the supported languages for all the available steps in Renard.


# References
