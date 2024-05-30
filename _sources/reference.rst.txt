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


Quote Detection
===============

QuoteDetector
-------------

.. autoclass:: renard.pipeline.quote_detection.QuoteDetector
   :members:


Sentiment Analysis
==================

NLTKSentimentAnalyzer
---------------------

.. autoclass:: renard.pipeline.sentiment_analysis.NLTKSentimentAnalyzer
   :members:

      
Characters Unification
======================

.. autoclass:: renard.pipeline.character_unification.Character
   :members:

NaiveCharacterUnifier
---------------------

.. autoclass:: renard.pipeline.character_unification.NaiveCharacterUnifier
   :members:

GraphRulesCharacterUnifier
--------------------------

.. autoclass:: renard.pipeline.character_unification.GraphRulesCharacterUnifier
   :members:


Speaker Attribution
===================

.. autoclass:: renard.pipeline.speaker_attribution.BertSpeakerDetector
   :members:


Graph Extraction
================

CoOccurrencesGraphExtractor
---------------------------

.. autoclass:: renard.pipeline.graph_extraction.CoOccurrencesGraphExtractor
   :members:

ConversationalGraphExtractor
----------------------------

.. autoclass:: renard.pipeline.graph_extraction.ConversationalGraphExtractor
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


Utils
=====

.. automodule:: renard.utils
   :members:

Graph utils
-----------

.. automodule:: renard.graph_utils
   :members:


Plot utils
----------

.. automodule:: renard.plot_utils
   :members:


NER utils
---------

.. automodule:: renard.ner_utils
   :members:
