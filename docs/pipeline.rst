============
The Pipeline
============

Renard's central concept is the :class:`.Pipeline`. A
:class:`.Pipeline` is a list of :class:`.PipelineStep` that are run
sequentially in order to extract a characters graph from a
document. Here is a simple example:

.. code-block:: python

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
           GraphRulesCharacterUnifier(min_appearances=10),
           CoOccurrencesGraphExtractor(co_occurences_dist=25)
       ]
   )

   out = pipeline(text)


Each step of a pipeline may require informations from previous steps
before running : therefore, it is possible to create intractable
pipelines when a step's requirements are not satisfied. To
troubleshoot these issues more easily, a :class:`.Pipeline` checks its
validity at run time, and throws an exception with an helpful message
in case it is intractable.

You can also specify the result of certains steps manually when
calling the pipeline if you already have those results or if you want
to compute them yourself :

.. code-block:: python

   from renard.pipeline import Pipeline
   from renard.pipeline.ner import NLTKNamedEntityRecognizer
   from renard.pipeline.character_unification import GraphRulesCharacterUnifier
   from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

   with open("./my_doc.txt") as f:
       text = f.read()

   # note that this pipeline doesn't have any tokenizer
   pipeline = Pipeline(
       [
           NLTKNamedEntityRecognizer(),
           GraphRulesCharacterUnifier(min_appearances=10),
           CoOccurrencesGraphExtractor(co_occurences_dist=25)
       ]
   )

   # tokens are passed at call time
   out = pipeline(text, tokens=my_tokenization_function(text))


For simplicity, one can use one of the preconfigured pipelines:

.. code-block:: python

   from renard.pipeline.preconfigured import bert_pipeline

   with open("./my_doc.txt") as f:
       text = f.read()

   pipeline = bert_pipeline(
       graph_extractor_kwargs={"co_occurences_dist": (1, "sentences")}
   )
   out = pipeline(text)


Pipeline Output: the Pipeline State
===================================

The :class:`.PipelineState` represents a state that is propagated and
annotated during the execution of a :class:`.Pipeline`. It is the
final value returned when running a pipeline with
:meth:`.Pipeline.__call__`. As such, one can use it to do different
things. For example, one can access the extracted character network as
a networkx graph:

>>> out.characters_graph
<networkx.classes.graph.Graph object at 0x7fd9e9115900>

one can also access the output of each :class:`.PipelineStep`.

A few matplotlib-based plot functions are provided for convenience
(:meth:`.PipelineState.plot_graph`,
:meth:`.PipelineState.plot_graph_to_file`):

>>> import matplotlib.pyplot as plt
>>> out.plot_graph()
>>> plt.show()

These functions should be seen more as exploration and debug tools
rather than fully-fledged visualisation platforms. If you want a
fully-featured visualisation tool, you can export your graph to
Gephi's `gexf` format:

>>> out.export_graph_to_gexf("./graph.gexf")


Available Steps: An Overview
============================

Below is an overview of the different steps that can make up a
pipeline. Note that :class:`.StanfordCoreNLPPipeline` is a special
case and regroup several steps as the same time.

Preprocessing
-------------

:class:`.CustomSubstitutionPreprocessor` allows to make regex-based
substitutions in the text.


Tokenization
------------

Tokenization is the task of cutting text in *tokens*. It is usually
the first task to apply to a text. 2 tokenizer are available:

- :class:`.NLTKTokenizer`
- :class:`.StanfordCoreNLPPipeline` does contain a tokenizer as part
  of its full NLP pipeline.


Named Entity Recognition
------------------------

Named entity recognition (NER) detects entities occurences in the
text. 3 modules are available:

- :class:`.NLTKNamedEntityRecognizer`
- :class:`.BertNamedEntityRecognizer`
- :class:`.StanfordCoreNLPPipeline` contains a NER model as part of
  its full NLP pipeline.


Coreference Resolution
----------------------

- :class:`.SpacyCorefereeCoreferenceResolver`
- :class:`.BertCoreferenceResolver`, using the Tibert library.
- :class:`.StanfordCoreNLPPipeline` can execute a coreference
  resolution model as part of its pipeline.


Quote Detection
---------------

- :class:`.QuoteDetector`


Sentiment Analysis
------------------

- :class:`.NLTKSentimentAnalyzer` leverages NLTK's Vader for sentiment
  analysis


Characters Extraction
---------------------

Characters extraction (or alias resolution) extract characters from
occurences detected using NER. This is done by assigning each mention
to a unique character.

- :class:`.NaiveCharacterUnifier`
- :class:`.GraphRulesCharacterUnifier`


Speaker Attribution
-------------------

- :class:`.BertSpeakerDetector`


Graph Extraction
----------------

- :class:`.CoOccurrencesGraphExtractor`
- :class:`.ConversationalGraphExtractor`


Dynamic Graphs
==============

Renard can also extract *dynamic graphs*: graphs that evolve through
time. In Renard, such graphs are representend by a ``List`` of
``networkx.Graph``.

.. code-block:: python

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
           GraphRulesCharacterUnifier(min_appearances=10),
           CoOccurrencesGraphExtractor(
	       co_occurences_dist=25,
	       dynamic=True,     # note the 'dynamic'
	       dynamic_window=20 # and the 'dynamic_window' argument
	   )
       ]
   )

   out = pipeline(text)


When executing the above block of code, the output attribute
``characters_graph`` will be a list of networkx graphs:

>>> out.characters_graph
[<networkx.classes.graph.Graph object at 0x7fd9e9115900>]

Plot and export functions work as one would expect
intuitively. :meth:`.PipelineState.plot_graph` allow to visualize the
dynamic graph using a slider, and
:meth:`.PipelineState.plot_graphs_to_dir` saves plots of the dynamic
graph to a directory. Meanwhile,
:meth:`.PipelineState.export_graph_to_gexf` correctly exports the
dynamic graph to the Gephi format.
