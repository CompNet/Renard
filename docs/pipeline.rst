========
Pipeline
========


Core
====

The Pipeline Object
-------------------

A :class:`.Pipeline` is a list of :class:`.PipelineStep` that are run
sequentially in order to extract a characters graph from a
document. Here is a simple example :

.. code-block:: python

   from renard.pipeline import Pipeline
   from renard.pipeline.tokenization import NLTKTokenizer
   from renard.pipeline.ner import NLTKNamedEntityRecognizer
   from renard.pipeline.characters_extraction import NaiveCharactersExtractor
   from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor

   with open("./my_doc.txt") as f:
       text = f.read()

   pipeline = Pipeline(
       [
           NLTKTokenizer(),
           NLTKNamedEntityRecognizer(),
           NaiveCharactersExtractor(min_appearance=10),
           CoOccurencesGraphExtractor(co_occurences_dist=25)
       ]
   )

   out = pipeline(text)
   out.export_graph_to_gexf("./network.gexf")


Each step of a pipeline may require informations from previous steps
before running : therefore, it is possible to create intractable
pipelines when a step's requirements are not satisfied. To
troubleshoot those issues more easily, a :class:`.Pipeline` checks its
validity at instantiation time, and throws an exception with an
helpful message in case it is intractable.

You can also specify the result of certains steps manually when
calling the pipeline if you already have those results or if you wan't
to compute them yourself :

.. code-block:: python

   from renard.pipeline import Pipeline
   from renard.pipeline.ner import NLTKNamedEntityRecognizer
   from renard.pipeline.characters_extraction import NaiveCharactersExtractor
   from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor

   with open("./my_doc.txt") as f:
       text = f.read()

   # note that this pipeline doesn't have any tokenizer
   pipeline = Pipeline(
       [
           NLTKNamedEntityRecognizer(),
           NaiveCharactersExtractor(min_appearance=10),
           CoOccurencesGraphExtractor(co_occurences_dist=25)
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


.. autoclass:: renard.pipeline.core.Pipeline
   :members:


Pipeline State
--------------

A state is propagated and annotated during the execution of a
:class:`.Pipeline`.

It is the final value returned when running a pipeline with
:meth:`.Pipeline.__call__`.


.. autoclass:: renard.pipeline.core.PipelineState
   :members:


Pipeline Steps 
--------------

A pipeline is a sequential series of
:class:`.PipelineStep`, that are applied in order.

.. autoclass:: renard.pipeline.core.PipelineStep
   :members:


Creating new steps
~~~~~~~~~~~~~~~~~~

Usually, steps must implement at least four functions :

- :meth:`.PipelineStep.__init__`: is used to pass options at step init time
- :meth:`.PipelineStep.__call__`: is called at pipeline run time
- :meth:`.PipelineStep.needs`: declares the set of informations needed
  from the pipeline state by this step. Each returned string should be
  an attribute of :class:`.PipelineState`.
- :meth:`.PipelineStep.production`: declares the set of informations
  produced by this step. As in :meth:`.PipelineStep.needs`, each
  returned string should be an attribute of :class:`.PipelineState`.


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

Additionally, the following methods can be overridden:

- :meth:`.PipelineStep.optional_needs`: specifies optional
  dependencies the same way as :meth:`.PipelineStep.needs`.
- :meth:`.PipelineStep._pipeline_init_`: is used for pipeline-wide
  arguments, such as language settings. This method is called at
  by the pipeline at pipeline run time.
- :meth:`.PipelineStep.supported_langs`: declares the set of supported
  languages as a set of ISO 639-3 codes (or the special value
  ``"any"``). By default, will be ``{"eng"}``.


Preprocessing
=============

.. automodule:: renard.pipeline.preprocessing
   :members:


Tokenization
============

NLTKTokenizer
-------------

.. autoclass:: renard.pipeline.tokenization.NLTKTokenizer
   :members:

BertTokenizer
-------------

.. autoclass:: renard.pipeline.tokenization.BertTokenizer
   :members:


Named Entity Recognition
========================

.. autoclass:: renard.pipeline.ner.NEREntity
   :members:

BertNamedEntityRecognizer
-------------------------

.. autoclass:: renard.pipeline.ner.BertNamedEntityRecognizer
   :members:

NLTKNamedEntityRecognizer
-------------------------

.. autoclass:: renard.pipeline.ner.NLTKNamedEntityRecognizer
   :members:


Coreference Resolution
======================

A coreference resolver returns a list of coreference chains, each
chain being :class:`.Mention`.

.. autoclass:: renard.pipeline.core.Mention
   :members:

BertCoreferenceResolver
-----------------------

.. autoclass:: renard.pipeline.corefs.BertCoreferenceResolver
   :members:


SpacyCorefereeCoreferenceResolver
---------------------------------

.. autoclass:: renard.pipeline.corefs.SpacyCorefereeCoreferenceResolver
   :members:

      
Characters Extraction
=====================

.. autoclass:: renard.pipeline.characters_extraction.Character
   :members:

NaiveCharactersExtractor
------------------------

.. autoclass:: renard.pipeline.characters_extraction.NaiveCharactersExtractor
   :members:

GraphRulesCharactersExtractor
-----------------------------

.. autoclass:: renard.pipeline.characters_extraction.GraphRulesCharactersExtractor
   :members:


Graph Extraction
================

.. autoclass:: renard.pipeline.graph_extraction.CoOccurencesGraphExtractor
   :members:



Stanford CoreNLP Pipeline
=========================

.. automodule:: renard.pipeline.stanford_corenlp
   :members:
