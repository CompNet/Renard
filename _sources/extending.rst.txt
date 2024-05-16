================
Extending Renard
================


Creating new steps
==================

Usually, steps must implement at least four functions :

- :meth:`.PipelineStep.__init__`: is used to pass options at step init
  time. Options passed at step init time should be valid for a
  collection of texts, and not be text specific.
- :meth:`.PipelineStep.__call__`: is called at pipeline run time.
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
