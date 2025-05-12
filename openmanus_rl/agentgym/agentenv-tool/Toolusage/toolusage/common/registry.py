"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


class Registry:
    mapping = {
        "environment_name_mapping": {},
        "agent_name_mapping": {},
        "llm_name_mapping": {},
        "task_name_mapping": {},
        "state": {},
    }
    @classmethod
    def register_environment(cls, name):
        r"""Register an environment to registry with key 'name'

        Args:
            name: Key with which the environment will be registered.

        Usage:

            from common.registry import registry
        """

        def wrap(env_cls):
           # from environment.base_env import BaseEnvironment

           # assert issubclass(
           #     env_cls, BaseEnvironment
           # ), "All environments must inherit BaseEnvironment class"
            if name in cls.mapping["environment_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["environment_name_mapping"][name]
                    )
                )
            cls.mapping["environment_name_mapping"][name] = env_cls
            return env_cls

        return wrap
    
    @classmethod
    def register_agent(cls, name):
        r"""Register an agent to registry with key 'name'

        Args:
            name: Key with which the agent will be registered.

        Usage:

            from common.registry import registry
        """

        def wrap(agent_cls):
            from agents.base_agent import BaseAgent

            assert issubclass(
                agent_cls, BaseAgent
            ), "All builders must inherit BaseAgent class, found {}".format(
                agent_cls
            )
            if name in cls.mapping["agent_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["agent_name_mapping"][name]
                    )
                )
            cls.mapping["agent_name_mapping"][name] = agent_cls
            return agent_cls

        return wrap

    
    @classmethod
    def register_llm(cls, name):
        r"""Register an llm to registry with key 'name'

        Args:
            name: Key with which the llm will be registered.

        Usage:

            from common.registry import registry
        """

        def wrap(llm_cls):
            
            if name in cls.mapping["llm_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["llm_name_mapping"][name]
                    )
                )
            cls.mapping["llm_name_mapping"][name] = llm_cls
            return llm_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""Register an task to registry with key 'name'

        Args:
            name: Key with which the llm will be registered.

        Usage:

            from common.registry import registry
        """

        def wrap(task_cls):
            
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap

    
    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    

    @classmethod
    def get_environment_class(cls, name):
        return cls.mapping["environment_name_mapping"].get(name, None)

    @classmethod
    def get_llm_class(cls, name):
        return cls.mapping["llm_name_mapping"].get(name, None)

    @classmethod
    def get_agent_class(cls, name):
        return cls.mapping["agent_name_mapping"].get(name, None)
    
    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def list_environments(cls):
        return sorted(cls.mapping["environment_name_mapping"].keys())
    
    @classmethod
    def list_agents(cls):
        return sorted(cls.mapping["agent_name_mapping"].keys())
    
    @classmethod
    def list_llms(cls):
        return sorted(cls.mapping["llm_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())


    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
