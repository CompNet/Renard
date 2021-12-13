========
Pipeline
========


Core
====

A :class:`renard.pipeline.core.Pipeline` is a list of
:class:`renard.pipeline.core.PipelineStep` that are run sequentially
in order to extract a characters graph from a document. Here is a
simple example :

.. code-block:: python

   import networkx as nx
   from renard.pipeline.core import Pipeline
   from renard.pipeline.tokenization import NLTKWordTokenizer
   from renard.pipeline.ner import NLTKNamedEntityRecognizer
   from renard.pipeline.characters_extraction import NaiveCharactersExtractor
   from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor

   with open("./my_doc.txt") as f:
       text = f.read()

   pipeline = Pipeline(
       (
           NLTKWordTokenizer(),
           NLTKNamedEntityRecognizer(),
           NaiveCharactersExtractor(min_appearance=10),
           CoOccurencesGraphExtractor(co_occurences_dist=25)
       )
   )

   out = pipeline(text)
   nx.write_gexf(out["characters_graph"], "./network.gexf")


Each step of a pipeline may require informations from previous steps
before running : therefore, it is possible to create intractable
pipelines when a step's requirements are not satisfied. To
troubleshoot those issues more easily, a
:class:`renard.pipeline.core.Pipeline` checks its validity at
instantiation time, and throws an exception with an helpful message in
case it is intractable.


.. automodule:: renard.pipeline.core
   :members:


Preprocessing
=============

.. automodule:: renard.pipeline.preprocessing
   :members:


Tokenization
============

.. automodule:: renard.pipeline.tokenization
   :members:


NER
===

.. automodule:: renard.pipeline.ner
   :members:


Characters Extraction
=====================

.. automodule:: renard.pipeline.characters_extraction
   :members:



Graph Extraction
================

.. automodule:: renard.pipeline.graph_extraction
   :members:



Stanford CoreNLP Pipeline
=========================

.. automodule:: renard.pipeline.stanford_corenlp
   :members:
