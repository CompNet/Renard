from typing import Any, Dict, List, Tuple, Set, Optional
from tqdm import tqdm


class PipelineStep:
    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    def needs(self) -> Set[str]:
        raise NotImplementedError()

    def produces(self) -> Set[str]:
        raise NotImplementedError()


class Pipeline:
    """A flexible NLP pipeline"""

    def __init__(self, steps: Tuple[PipelineStep]) -> None:
        self.steps = steps

    def check_valid(self) -> Tuple[bool, Optional[str]]:
        """Check that the current pipeline can be run, which is
        possible if all steps needs are satisfied

        :return: a tuple :
            - ``(True, None)`` if the pipeline is valid
            - ``(False, "an error message")`` otherwise
        """
        pipeline_state = {"text"}
        for i, step in enumerate(self.steps):
            if not step.needs().issubset(pipeline_state):
                return (
                    False,
                    f"step {i + 1} ({step.__class__.__name__}) has unsatisfied needs (needs : {step.needs()}, available : {pipeline_state})",
                )
            pipeline_state = pipeline_state.union(step.produces())
        return (True, None)

    def __call__(self, text: str) -> Dict[str, Any]:
        """Run the pipeline sequentially

        :return: the output of the last step of the pipeline
        """
        is_valid, reason = self.check_valid()
        if not is_valid:
            raise ValueError(reason)

        kwargs = {"text": text}
        tqdm_steps = tqdm(self.steps, total=len(self.steps))
        for step in tqdm_steps:
            tqdm_steps.set_description_str(f"{step.__class__.__name__}")
            kwargs = {**kwargs, **step(**kwargs)}

        return kwargs
