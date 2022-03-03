========
Pipeline
========


Core
====

The Pipeline Object
-------------------

A :class:`renard.pipeline.core.Pipeline` is a list of
:class:`renard.pipeline.core.PipelineStep` that are run sequentially
in order to extract a characters graph from a document. Here is a
simple example :

.. code-block:: python

   from renard.pipeline import Pipeline
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
   out.export_graph_to_gexf("./network.gexf")


Each step of a pipeline may require informations from previous steps
before running : therefore, it is possible to create intractable
pipelines when a step's requirements are not satisfied. To
troubleshoot those issues more easily, a
:class:`renard.pipeline.core.Pipeline` checks its validity at
instantiation time, and throws an exception with an helpful message in
case it is intractable.


.. autoclass:: renard.pipeline.core.Pipeline
   :members:


Pipeline State
--------------

A state is propagated and annotated during the execution of a
:class:`renard.pipeline.core.Pipeline`.

It is the final value returned when running a pipeline with
:func:`renard.pipeline.core.Pipeline.__call__`.


.. autoclass:: renard.pipeline.core.PipelineState
   :members:


Pipeline Steps 
--------------

A pipeline is a sequential series of
:class:`renard.pipeline.core.PipelineStep`, that are applied in order.

.. autoclass:: renard.pipeline.core.PipelineStep
   :members:


Creating new steps
~~~~~~~~~~~~~~~~~~

Usually, steps must implement at least four functions :

- ``__init__`` : is used to pass options at step init time
- ``__call__`` : is called at pipeline run time
- ``needs`` : declares the set of informations needed from the pipeline
  state by this step
- ``production`` : declares the set of informations produced by this
  step


Here is an example of creating a basic tokenization step :

.. code-block:: python

   from typing import Dict, Any, Set
   from renard.pipeline.core import PipelineStep

   class BasicTokenizerStep(PipelineStep):

       def __init__(self):
           pass

       def __call__(self, text: str, **kwargs) -> Dict[str, Any]: 
           return {"tokens": text.split(" ")}

       def needs(self) -> Set[str]: 
           return {"text"}

       def production(self) -> Set[str]: 
           return {"tokens"}



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
