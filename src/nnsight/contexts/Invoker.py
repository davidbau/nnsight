from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Dict

from ..tracing.Proxy import Proxy
from .Tracer import Tracer


class Invoker(AbstractContextManager):
    """An Invoker is meant to work in tandem with a :class:`nnsight.contexts.Tracer.Tracer` to enter input and manage intervention tracing.

    Attributes:
        tracer (nnsight.contexts.Tracer.Tracer): Tracer object to enter input and manage context.
        input (Any): Initially entered input, then post-processed input from model's _prepare_inputs method.
        scan (bool): If to use a 'meta' version of the  post-processed input to run through the model using it's _scan method,
            in order to update the potential sizes/dtypes of all module's inputs/outputs as well as validate things work correctly.
            Scanning is not free computation wise so you may want to turn this to false when running in a loop.
            When making interventions, you made get shape errors if scan is false as it validates operations based on shapes so
            for looped calls where shapes are consistent, you may want to have scan=True for the first loop. Defaults to True.
        args (List[Any]): Positional arguments passed to the model's _prepare_inputs method.
        kwargs (Dict[str,Any]): Keyword arguments passed to the model's _prepare_inputs method.
    """

    def __init__(
        self,
        tracer: Tracer,
        input: Any,
        *args,
        scan: bool = True,
        **kwargs,
    ) -> None:
        self.tracer = tracer
        self.input = input
        self.scan = scan
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> Invoker:
        """Enters a new invocation context with a given input.

        Sets the generation_idx to 0.
        Calls the model's _prepare_inputs method using the input and other arguments.
        If scan is True, uses the model's _scan method to update and validate module inputs/outputs.
        Gets a batched version of the post processed input using the model's _batched_inputs method to update the Tracer's
            current batch_size and batched_input.

        Returns:
            Invoker: Invoker.
        """
        # Were in a new invocation so set generation_idx to 0,
        self.tracer.generation_idx = 0

        self.input = self.tracer.model._prepare_inputs(
            self.input, *self.args, **self.kwargs
        )

        if self.scan:
            self.tracer.model._scan(self.input, *self.tracer.args, **self.tracer.kwargs)
        else:
            for name, module in self.tracer.model.meta_model.named_modules():
                module._output = None
                module._input = None
                module._backward_output = None
                module._backward_input = None

        batched_inputs = self.tracer.model._batched_inputs(self.input)

        self.tracer.batch_size = len(batched_inputs)
        self.tracer.batched_input.extend(batched_inputs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def next(self, increment: int = 1) -> None:
        """Designates subsequent interventions should be applied to the next generation for multi-iteration generation runs.

        Args:
            increment (int): How many generation_idx to increment at once. Defaults to 1.
        """
        # .next() increases which generation idx the interventions happen.
        self.tracer.generation_idx += increment

        if self.scan:
            # Run graph with singe token input.
            self.inputs = self.tracer.model._prepare_inputs(
                self.tracer.model._example_input(), *self.args, **self.kwargs
            )
            self.tracer.model._scan(
                self.inputs, *self.tracer.args, **self.tracer.kwargs
            )
        else:
            for name, module in self.tracer.model.meta_model.named_modules():
                module._output = None
                module._input = None
                module._backward_output = None
                module._backward_input = None

    def save_all(self) -> Dict[str, Proxy]:
        """Saves the output of all modules and returns a dictionary of [module_path -> save proxy]

        Returns:
            Dict[str, Proxy]: Dictionary of all modules saved, keyed by their module_path.
        """
        result = {}

        for name, module in self.tracer.model.meta_model.named_modules():
            result[module.module_path] = module.output.save()

        return result
