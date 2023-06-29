=========
Reference
=========


Core
====

Pipeline
--------

.. autoclass:: renard.pipeline.core.Pipeline
   :members:


Pipeline State
--------------

.. autoclass:: renard.pipeline.core.PipelineState
   :members:


Pipeline Steps
--------------

.. autoclass:: renard.pipeline.core.PipelineStep
   :members:


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


Sentiment Analysis
==================

NLTKSentimentAnalyzer
---------------------

.. autoclass:: renard.pipeline.sentiment_analysis.NLTKSentimentAnalyzer
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

.. autoclass:: renard.pipeline.graph_extraction.CoOccurrencesGraphExtractor
   :members:



Stanford CoreNLP Pipeline
=========================

.. automodule:: renard.pipeline.stanford_corenlp
   :members:


Resources
=========

Hypocorism
----------

.. autoclass:: renard.resources.hypocorisms.HypocorismGazetteer
   :members:
