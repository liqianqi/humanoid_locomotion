from dataclasses import MISSING

import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab.managers.action_manager import ActionManager
from isaaclab.managers.observation_manager import ObservationManager


@configclass
class SymmetryCfg:
    """
    Configuration for the symmetry of the environment.

    For each possible symmetry configuration, specify the observation / action term names
    and the indices of elements inside the term to perform the operation on.
    """

    # A_sym, B_sym = B, A
    observation_swap_terms: dict[str, list[tuple[int, int]]] = MISSING  # pyright: ignore[reportAssignmentType]

    # A_sym, B_sym = -B, -A
    observation_swap_negate_terms: dict[str, list[tuple[int, int]]] = MISSING  # pyright: ignore[reportAssignmentType]

    # A_sym = -A
    observation_negate_terms: dict[str, list[int]] = MISSING  # pyright: ignore[reportAssignmentType]

    # A_sym, B_sym = B, A
    action_swap_terms: dict[str, list[tuple[int, int]]] = MISSING  # pyright: ignore[reportAssignmentType]

    # A_sym, B_sym = -B, -A
    action_swap_negate_terms: dict[str, list[tuple[int, int]]] = MISSING  # pyright: ignore[reportAssignmentType]

    # A_sym = -A
    action_negate_terms: dict[str, list[int]] = MISSING  # pyright: ignore[reportAssignmentType]


def __get_observation_term_index(observation_manager: ObservationManager, group_name: str, term_name: str) -> int | None:
    """
    Get the index of the first element of the specified observation term in the observation tensor.

    Args:
        observation_manager (ObservationManager): The observation manager.
        group_name (str): The name of the group to get the index of.
        term_name (str): The name of the term to get the index of.

    Returns:
        int: The index of the first element of the specified observation term in the observation tensor.
    """
    term_index = 0
    for name, dims in zip(
        observation_manager._group_obs_term_names[group_name],
        observation_manager._group_obs_term_dim[group_name],
    ):
        if name == term_name:
            return term_index

        term_index += dims[-1]

    # print(f"WARNING: Term {term_name} not found in observation manager for group {group_name}")
    return None


def __get_action_term_index(action_manager: ActionManager, term_name: str) -> int:
    """
    Get the index of the first element of the specified action term in the action tensor.

    Args:
        action_manager (ActionManager): The action manager.
        term_name (str): The name of the term to get the index of.

    Returns:
        int: The index of the first element of the specified action term in the action tensor.
    """
    term_index = 0
    for (name, term) in action_manager._terms.items():
        dim = term.action_dim
        if name == term_name:
            break
        term_index += dim
    return term_index


def symmetry_data_augmentation_function(
    env: RslRlVecEnvWrapper,
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    The symmetry data augmentation function.

    This function implements symmetry-based data augmentation for the G1 robot walking task.
    It swaps and negates certain actions components to create symmetric training samples.

    Args:
        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (torch.Tensor | None): The observation tensor. If None, the observation is not used.
        actions (torch.Tensor | None): The actions tensor. If None, the actions is not used.
        obs_type (str): The name of the observation type. Defaults to "policy".
            This is useful when handling augmentation for different observation groups.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None]: A tuple containing the augmented observation and actions tensors.
    """
    assert obs_type == "policy" or obs_type == "critic"

    # TODO: change to search by class type, instead of attribute name
    symmetry_cfg: SymmetryCfg | None = getattr(getattr(env, "cfg", None), "symmetry", None)

    if symmetry_cfg is None:
        print("WARNING: No symmetry configuration found")
        return obs, actions

    # Augment the observation
    if obs is not None:
        symmetry_obs = obs.clone()
        manager = env.unwrapped.observation_manager  # pyright: ignore[reportAttributeAccessIssue]

        # Swap pairs
        for term_name in symmetry_cfg.observation_swap_terms:
            # calculate the location of the term in the observation
            term_index = __get_observation_term_index(manager, obs_type, term_name)

            if term_index is None:
                continue

            for id_pair in symmetry_cfg.observation_swap_terms[term_name]:
                # NOTE: a tuple-swap ``x[a], x[b] = x[b], x[a]`` is BROKEN for tensors -- the RHS are
                # views that alias x, so both columns end up = x[b]. Clone first. (This bug in the
                # plain-swap silently corrupted pitch-joint mirroring and poisoned training.)
                i0, i1 = term_index + id_pair[0], term_index + id_pair[1]
                tmp = symmetry_obs[..., i0].clone()
                symmetry_obs[..., i0] = symmetry_obs[..., i1]
                symmetry_obs[..., i1] = tmp

        # Swap and negate pairs
        for term_name in symmetry_cfg.observation_swap_negate_terms:
            # calculate the location of the term in the observation
            term_index = __get_observation_term_index(manager, obs_type, term_name)

            if term_index is None:
                continue

            for id_pair in symmetry_cfg.observation_swap_negate_terms[term_name]:
                symmetry_obs[..., term_index + id_pair[0]], symmetry_obs[..., term_index + id_pair[1]] = (
                    -symmetry_obs[..., term_index + id_pair[1]],
                    -symmetry_obs[..., term_index + id_pair[0]],
                )

        # Negate
        for term_name in symmetry_cfg.observation_negate_terms:
            # calculate the location of the term in the observation
            term_index = __get_observation_term_index(manager, obs_type, term_name)

            if term_index is None:
                continue

            for id in symmetry_cfg.observation_negate_terms[term_name]:
                symmetry_obs[..., term_index + id] = -symmetry_obs[..., term_index + id]

        augmented_obs = torch.cat([obs, symmetry_obs], dim=0)
    else:
        augmented_obs = None

    # Augment the actions
    if actions is not None:
        symmetry_actions = actions.clone()
        manager = env.unwrapped.action_manager  # pyright: ignore[reportAttributeAccessIssue]

        # Swap pairs
        for term_name in symmetry_cfg.action_swap_terms:
            # calculate the location of the term in the action
            term_index = __get_action_term_index(manager, term_name)

            for id_pair in symmetry_cfg.action_swap_terms[term_name]:
                # clone-swap (see the observation swap above -- tuple-swap aliases tensor views)
                i0, i1 = term_index + id_pair[0], term_index + id_pair[1]
                tmp = symmetry_actions[..., i0].clone()
                symmetry_actions[..., i0] = symmetry_actions[..., i1]
                symmetry_actions[..., i1] = tmp

        # Swap and negate pairs
        for term_name in symmetry_cfg.action_swap_negate_terms:
            # calculate the location of the term in the action
            term_index = __get_action_term_index(manager, term_name)

            for id_pair in symmetry_cfg.action_swap_negate_terms[term_name]:
                symmetry_actions[..., term_index + id_pair[0]], symmetry_actions[..., term_index + id_pair[1]] = (
                    -symmetry_actions[..., term_index + id_pair[1]],
                    -symmetry_actions[..., term_index + id_pair[0]],
                )

        # Negate
        for term_name in symmetry_cfg.action_negate_terms:
            # calculate the location of the term in the action
            term_index = __get_action_term_index(manager, term_name)

            for id in symmetry_cfg.action_negate_terms[term_name]:
                symmetry_actions[..., term_index + id] = -symmetry_actions[..., term_index + id]

        augmented_actions = torch.cat([actions, symmetry_actions], dim=0)
    else:
        augmented_actions = None

    return augmented_obs, augmented_actions
