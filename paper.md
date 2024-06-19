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
  - name: Laboratoire Informatique d'Avignon, France
    index: 1
  - name: Laboratoire des Sciences du Numérique de Nantes, France
    index: 2
date: 29 February 2024
bibliography: paper.bib
---

# Summary

Renard (*Relationships Extraction from NARrative Documents*) is a Python library that allows users to define custom natural language processing (NLP) pipelines to extract character networks from narrative texts. Contrary to the few existing tools, Renard can extract *dynamic* networks, as well as the more common static networks. Renard pipelines are modular: users can choose the implementation of each NLP subtask needed to extract a character network. This allows users to specialize pipelines to particular types of texts and to study the impact of each subtask on the extracted network.

# Statement of Need

Character networks (i.e., graphs where vertices represent characters and edges represent their relationships) extracted from narrative texts are useful in a number of applications, from visualization to literary analysis [@labatut-2019]. There are different ways of modeling relationships (co-occurrences, conversations, actions, etc.), and networks can be static or dynamic (i.e., series of networks representing the evolution of relationships through time). This variety means one can extract different kinds of networks depending on the targeted applications. While some authors extract these networks by relying on manually annotated data [@rochat-2014-phd_thesis; @rochat-2015-character_networks_zola; @rochat-2017; @park-2013-character_networks; @park-2013-character_networksb], it is a time-costly endeavor, and the fully automatic extraction of these networks is therefore of interest. Unfortunately, there are only a few existing software packages and tools that can extract character networks [@sparavigna-2015-chaplin; @metrailler-2023-charnetto], but none of these can output dynamic networks. Furthermore, automatically extracting a character network requires solving several successive natural language processing tasks, such as named entity recognition (NER) or coreference resolution, and algorithms carrying these tasks are bound to make errors. To our knowledge, the cascading impact of these errors on the quality of the extracted networks has yet to be studied extensively. This is an important issue since knowing which tasks have more influence on the extracted networks would allow prioritizing research efforts.

Renard is a fully configurable pipeline that can extract static and dynamic networks from narrative texts. We base Renard on the generic character network extraction framework highlighted by the survey of @labatut-2019. We design it so that it is as modular as possible, which allows the user to select the implementation of each extraction step as needed. This has several advantages:

1. Depending on the input text, the user can choose the most relevant series of steps and configure each of them as needed. Therefore, the pipeline can be specialized for different types of texts, allowing for better performance.
2. The pipeline can easily incorporate new advances in NLP, by simply implementing a new step when necessary.
3. One can study the impact of the performance of each step on the quality of the extracted networks.

We intend for Renard to be used by digital humanities researchers as well as NLP researchers and practitioners. The former category of users can use Renard to quickly extract character networks for literary analysis. Meanwhile, the latter can use Renard to easily represent textual content using networks, which can be used as inputs for downstream NLP tasks (classification, recommendation...). 


# Design and Main Features

Renard is centered about the concept of a *pipeline*. In Renard, a pipeline is a series of sequential *steps* that are run one after the other in order to extract a character network from a text. When using Renard, the user simply *describes* this pipeline in Python by specifying this series of steps, and can apply it to different texts afterwards. The following code block exemplifies that philosophy:

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
        # users can pass 'dynamic=True' and specify the
        # 'dynamic_window' argument to extract a dynamic network
        # instead of a static one.
		CoOccurrencesGraphExtractor(
            co_occurrences_dist=10, dynamic=False
        )
	]
)

out = pipeline(text)
```

![Co-occurrence character network of Jane Austen's "Pride and Prejudice", extracted automatically using Renard. Vertex size and color denote degree, while edge thickness and color denote the number of co-occurrences between two characters.](./pp.pdf){#fig:pp_network height=30% }

As an example, Figure \ref{fig:pp_network} shows the co-occurrence character network Jane Austen's 1813 novel "Pride and Prejudice", extracted using the Renard pipeline above. While this network is static, users can also extract a dynamic network by passing the `dynamic=True` argument to the last step of the pipeline, and specifying the `dynamic_window` argument: in that case, Renard outputs a list of graphs corresponding to a dynamic network instead of a single network[^1]. Renard uses the NetworkX Python library [@hagberg-2008-networkx] to manipulate graphs, ensuring compatibility with a wide array of tools and formats.

[^1]: See [the documentation on dynamic networks](https://compnet.github.io/Renard/pipeline.html#dynamic-graphs) for more details.

To allow for custom needs, we design Renard to be very flexible. If a step is not available in Renard, we encourage users to either:

- Externally perform the computation corresponding to the desired step, and inject the results back into the pipeline at runtime,
- Implement their own step to integrate their custom processing into Renard by subclassing the existing `PipelineStep` class. If necessary, this `PipelineStep` can act as an adapter to an external process that may or may not be written in Python.


The flexibility of this approach introduces the possibility of creating invalid pipelines because steps often require information computed by previously ran steps: for example, solving the NER task requires a tokenized version of the input text. To counteract this issue, each step therefore declares its requirements and the new information it produces, which allows Renard to check whether a pipeline is valid, and to explain at runtime to the user why it may not be[^2].

[^2]: See [the documentation](https://compnet.github.io/Renard/pipeline.html#the-pipeline) for more details on steps requirements.

: Existing steps and their supported languages in Renard. \label{tab:steps}

| Step                                                                        | Supported Languages                   |
|-----------------------------------------------------------------------------|---------------------------------------|
| `StanfordCoreNLPPipeline`                                                   | eng                                   |
| `CustomSubstitutionPreprocessor`                                            | any                                   |
| `NLTKTokenizer`                                                             | eng, fra, rus, ita, spa... (12 other) |
| `QuoteDetector`                                                             | any                                   |
| `NLTKNamedEntityRecognizer`                                                 | eng, rus                              |
| `BertNamedEntityRecognizer`                                                 | eng, fra                              |
| `BertCoreferenceResolver`                                                   | eng                                   |
| `SpacyCorefereeCoreferenceResolver`                                         | eng                                   |
| `NaiveCharacterUnifier`                                                     | any                                   |
| `GraphRulesCharacterUnifier` (inspired from @vala-2015-character_detection) | eng, fra                              |
| `BertSpeakerDetector`                                                       | eng                                   |
| `CoOccurencesGraphExtractor`                                                | any                                   |
| `ConversationalGraphExtractor`                                              | any                                   |


Renard lets users select the targeted language of its their custom pipelines. A pipeline can be configured to run in any language, as long as each of its steps supports it. Table \ref{tab:steps} shows all the currently available steps in Renard and their supported languages.



# References
