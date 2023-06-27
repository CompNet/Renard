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
   from renard.pipeline.characters_extraction import NaiveCharactersExtractor
   from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

   with open("./my_doc.txt") as f:
       text = f.read()

   pipeline = Pipeline(
       [
           NLTKTokenizer(),
           NLTKNamedEntityRecognizer(),
           NaiveCharactersExtractor(min_appearance=10),
           CoOccurrencesGraphExtractor(co_occurences_dist=25)
       ]
   )

   out = pipeline(text)
   out.export_graph_to_gexf("./network.gexf")


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
   from renard.pipeline.characters_extraction import NaiveCharactersExtractor
   from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

   with open("./my_doc.txt") as f:
       text = f.read()

   # note that this pipeline doesn't have any tokenizer
   pipeline = Pipeline(
       [
           NLTKNamedEntityRecognizer(),
           NaiveCharactersExtractor(min_appearance=10),
           CoOccurrencesGraphExtractor(co_occurences_dist=25)
       ]
   )

   # tokens are passed at call time
   out = pipeline(text, tokens=my_tokenization_function(text))
   out.export_graph_to_gexf("./network.gexf")	


For simplicity, one can use one of the preconfigured pipelines:

.. code-block:: python

   from renard.pipeline.preconfigured import bert_pipeline

   with open("./my_doc.txt") as f:
       text = f.read()

   pipeline = bert_pipeline()
   out = pipeline(text)
   out.export_graph_to_gexf("./network.gexf")	


Pipeline Output: the Pipeline State
===================================

The :class:`.PipelineState` represents a state that is propagated and
annotated during the execution of a :class:`.Pipeline`. It is the
final value returned when running a pipeline with
:meth:`.Pipeline.__call__`. As such, one can use it to do different
things. For example, one can access the extracted character network:

>>> out = pipeline(text)
>>> out.characters_graph
<networkx.classes.graph.Graph object at 0x7fd9e9115900>

one can also access the output of each :class:`.PipelineStep`.

A few plot functions are provided for convenience
(:meth:`.PipelineState.plot_graph`,
:meth:`.PipelineState.plot_graph_to_file`,
:meth:`.PipelineState.plot_graphs_to_dir`). These functions should be
seen more as exploration and debug tools rather than fully-fledged
visualisation platforms. If you want a fully-featured visualisation
tool, you can export your graph to Gephi's `gexf` format:

>>> out.export_graph_to_gexf("./graph.gexf")



Pipeline Steps
==============

A pipeline is a sequential series of
:class:`.PipelineStep`, that are applied in order.
