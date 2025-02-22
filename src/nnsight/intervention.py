"""This module contains logic to interleave a computation graph (an intervention graph) with the computation graph of a model.

The :class:`InterventionProxy <nnsight.intervention.InterventionProxy>` class extends the functionality of a base nnsight.fx.Proxy.Proxy object and makes it easier for users to interact with.

:func:`intervene() <nnsight.intervention.intervene>` is the entry hook into the models computation graph in order to interleave an intervention graph.

The :class:`HookModel <nnsight.intervention.HookModel>` provides a context manager for adding input and output hooks to modules and removing them upon context exit.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, List, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle

from . import util
from .tracing.Graph import Graph
from .tracing.Proxy import Proxy


class InterventionProxy(Proxy):

    """Sub-class for Proxy that adds additional user functionality to proxies.

    Examples:

        Saving a proxy so it is not deleted at the completion of it's listeners is enabled with ``.save()``:

        .. code-block:: python

            with runner.invoke('The Eiffel Tower is in the city of') as invoker:
                hidden_states = model.lm_head.input.save()
                logits = model.lm_head.output.save()

            print(hidden_states.value)
            print(logits.value)

        This works and would output the inputs and outputs to the model.lm_head module.
        Had you not called .save(), calling .value would have been None.

        Indexing by token of hidden states can easily done using ``.token[<idx>]`` or ``.t[<idx>]``

        .. code-block:: python

            with runner.invoke('The Eiffel Tower is in the city of') as invoker:
                logits = model.lm_head.output.t[0].save()

            print(logits.value)

        This would save only the first token of the output for this module.
        This should be used when using multiple invokes as the batching and padding of multiple inputs could mean the indices for tokens shifts around and this take care of that.

        Calling ``.shape`` on an InterventionProxy returns the shape or collection of shapes for the tensors traced through this module.

        Calling ``.value`` on an InterventionProxy returns the actual populated values, updated during actual execution of the model.
        
    """

    def save(self) -> InterventionProxy:
        """Method when called, indicates to the intervention graph to not delete the tensor values of the result.

        Returns:
            InterventionProxy: Save proxy.
        """

        # Add a 'null' node with the save proxy as an argument to ensure the values are never deleted.
        # This is because 'null' nodes never actually get set and therefore there will always be a
        # dependency for the save proxy.
        self.node.graph.add(
            value=None,
            target="null",
            args=[self.node],
        )

        return self

    def retain_grad(self):

        self.node.graph.add(
            target=torch.Tensor.retain_grad,
            args=[self.node]
        )

        # We need to set the values of self to values of self to add this into the computation graph so grad flows through it
        # This is because in intervene(), we call .narrow on activations which removes it from the grad path
        self.node.graph.add(
            target=Proxy.proxy_update,
            args=[self.node, self.node]
        )

    @property
    def token(self) -> TokenIndexer:
        """Property used to do token based indexing on a proxy.
        Directly indexes the second dimension of tensors.
        Makes positive indices negative as tokens are padded on the left.

        Example:
            
            .. code-block:: python

                model.transformer.h[0].mlp.output.token[0]

            Is equivalent to:

            .. code-block:: python

                model.transformer.h[0].mlp.output.token[:,-3]

            For a proxy tensor with 3 tokens.

        Returns:
            TokenIndexer: Object to do token based indexing.
        """
        return TokenIndexer(self)

    @property
    def t(self) -> TokenIndexer:
        """Property as alias for InterventionProxy.token"""
        return self.token

    @property
    def shape(self) -> Union[torch.Size, Collection[torch.Size]]:
        """Property to retrieve the shape of the traced proxy value.

        Returns:
            Union[torch.Size,Collection[torch.Size]]: Proxy value shape or collection of shapes.
        """
        return util.apply(self.node.proxy_value, lambda x: x.shape, torch.Tensor)

    @property
    def value(self) -> Any:
        """Property to return the value of this proxy's node.

        Returns:
            Any: The stored value of the proxy, populated during execution of the model.
        """

        return self.node.value


def intervene(activations: Any, module_path: str, graph: Graph, key: str):
    """Entry to intervention graph. This should be hooked to all modules involved in the intervention graph.

    Forms the current module_path key in the form of <module path>.<output/input>.<graph generation index>
    Checks the graphs argument_node_names attribute for this key.
    If exists, value is a list of node names to iterate through.
    Node args for argument type nodes should be ``[module_path, batch_size, batch_start]``.
    Using batch_size and batch_start, apply torch.narrow to tensors in activations to select
    only batch indexed tensors relevant to this intervention node. Sets the value of a node
    using the indexed values. Using torch.narrow returns a view of the tensors as opposed to a copy allowing
    subsequent downstream nodes to make edits to the values only in the relevant tensors, and have it update the original
    tensors. This both prevents interventions from effecting bathes outside their preview and allows edits
    to the output from downstream intervention nodes in the graph.

    Args:
        activations (Any): Either the inputs or outputs of a torch module.
        module_path (str): Module path of the current relevant module relative to the root model.
        graph (Graph): Intervention graph to interleave with the computation "graph" of the model.
        key (str): Key denoting either "input" or "output" of module.

    Returns:
        Any: The activations, potentially modified by the intervention graph.
    """

    # Key to module activation argument nodes has format: <module path>.<output/input>.<generation index>
    module_path = f"{module_path}.{key}.{graph.generation_idx}"

    if module_path in graph.argument_node_names:
        
        argument_node_names = graph.argument_node_names[module_path]

        # multiple argument nodes can have same module_path if there are multiple invocations.
        for argument_node_name in argument_node_names:
            node = graph.nodes[argument_node_name]

            # args for argument nodes are (module_path, batch_size, batch_start)
            _, batch_size, batch_start = node.args

            # We set its result to the activations, indexed by only the relevant batch idxs.
            node.set_value(
                util.apply(
                    activations,
                    lambda x: x.narrow(0, batch_start, batch_size),
                    torch.Tensor,
                )
            )
    return activations


class HookModel(AbstractContextManager):
    """Context manager that applies input and/or output hooks to modules in a model.

    Registers provided hooks on __enter__ and removes them on __exit__.

    Attributes:
        model (torch.nn.Module): Root model to access modules and apply hooks to.
        modules (List[Tuple[torch.nn.Module, str]]): List of modules to apply hooks to along with their module_path.
        input_hook (Callable): Function to apply to inputs of designated modules.
            Should have signature of [inputs(Any), module_path(str)] -> inputs(Any)
        output_hook (Callable): Function to apply to outputs of designated modules.
            Should have signature of [outputs(Any), module_path(str)] -> outputs(Any)
        handles (List[RemovableHandle]): Handles returned from registering hooks as to be used when removing hooks on __exit__.
    """
    #TODO maybe only apply the necassay hooks (e.x if a module has a input hook, all hooks will be added)
    def __init__(
        self,
        model: torch.nn.Module,
        modules: List[str],
        input_hook: Callable = None,
        output_hook: Callable = None,
        backward_input_hook:Callable = None,
        backward_output_hook:Callable = None
    ) -> None:
        self.model = model
        self.modules: List[Tuple[torch.nn.Module, str]] = [
            (util.fetch_attr(self.model, module_path), module_path)
            for module_path in modules
        ]
        self.input_hook = input_hook
        self.output_hook = output_hook
        self.backward_input_hook = backward_input_hook
        self.backward_output_hook = backward_output_hook

        self.handles: List[RemovableHandle] = []

    def __enter__(self) -> HookModel:
        """Registers input and output hooks to modules if they are defined.

        Returns:
            HookModel: HookModel object.
        """

        for module, module_path in self.modules:
            if self.input_hook is not None:

                def input_hook(module, input, module_path=module_path):
                    return self.input_hook(input, module_path)

                self.handles.append(module.register_forward_pre_hook(input_hook))

            if self.output_hook is not None:

                def output_hook(module, input, output, module_path=module_path):
                    return self.output_hook(output, module_path)

                self.handles.append(module.register_forward_hook(output_hook))

            if self.backward_input_hook is not None:

                def backward_input_hook(module, input, output, module_path=module_path):
                    return self.backward_input_hook(input, module_path)

                self.handles.append(module.register_full_backward_hook(backward_input_hook))

            if self.backward_output_hook is not None:

                def backward_output_hook(module, output, module_path=module_path):
                    return self.backward_output_hook(output, module_path)

                self.handles.append(module.register_full_backward_pre_hook(backward_output_hook))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Removes all handles added during __enter__."""
        for handle in self.handles:
            handle.remove()


class TokenIndexer:
    """Helper class to directly access token indices of hidden states.
    Directly indexes the second dimension of tensors.
    Makes positive indices negative as tokens are padded on the left.

    Args:
        proxy (InterventionProxy): Proxy to aid in token indexing.
    """

    def __init__(self, proxy: InterventionProxy) -> None:
        self.proxy = proxy

    def convert_idx(self, idx: int):
        if idx >= 0:
            n_tokens = self.proxy.node.proxy_value.shape[1]
            idx = -(n_tokens - idx)

        return idx

    def __getitem__(self, key: int) -> Proxy:
        key = self.convert_idx(key)

        return self.proxy[:, key]

    def __setitem__(self, key: int, value: Union[Proxy, Any]) -> None:
        key = self.convert_idx(key)

        self.proxy[:, key] = value
