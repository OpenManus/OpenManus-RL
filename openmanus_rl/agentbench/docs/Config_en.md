# Configuration System

[🌏中文版](Config_cn.md)

## Basic Syntax

The configuration system uses the YAML format. To simplify the configuration, we made some extensions on top of the
basic YAML syntax. `import`, `default`, `overwrite` are our extended keywords.

### import

The `import` keyword is used to import configurations from other files. For example, the following two ways are
equivalent:

Method one:

```yaml
# config.yaml
definition:
  def1: something...
  def2: something...
```

Method two:

```yaml
# def1.yaml
def1: something...

# def2.yaml
def2: something...

# config.yaml
definition:
  import:
    - def1.yaml
    - def2.yaml
```

The `import` keyword supports using a string or list as its value, corresponding to importing a single file or multiple
files respectively.

During the import process, if there's an `import` keyword in the imported file, the `import` in the imported file will
be executed first. The same applies to the other two keywords.

During the import process, if a key conflict occurs, the system will try to recursively merge the values of the
conflicting keys. If the merge is not possible, the later value will overwrite the earlier one.

### default

The `default` keyword is used to specify default values. The following two ways are equivalent:

Method one:

```yaml
definition:
  def1:
    type: int
    value: 1
  def2:
    type: int
    value: 2
  def3:
    type: float
    value: 1.1
```

Method two:

```yaml
definition:
  default:
    type: int
  def1:
    value: 1
  def2:
    value: 2
  def3:
    type: float
    value: 1.1
```

The `default` keyword supports a string, list, or dictionary as its value. The config parser will try to merge
the `default` value with the value of the keys at the same level as `default`. In the event of a conflict, the `default`
keyword value has a lower priority.

### overwrite

The `overwrite` keyword works similarly to `default`. However, in the event of a conflict, the `overwrite` keyword value
has a higher priority. This keyword is often used with `import` to set the required values under this configuration
file.

## Configuration File

The main directory structure of the configuration file is as follows:

```
configs
├── assignments
│ ├── definition.yaml
│ ├── default.yaml
│ └── ...
├── agents
├── tasks
│ ├── task_assembly.yaml
│ └── ...
└── start_task.yaml
```

### assignments

The `assignments` directory contains all task configuration files. The `definition.yaml` collects all task definitions
and model definitions.

A single task configuration file mainly requires the following fields:

- `definition`: usually imported from `definition.yaml`, for defining tasks and models.
- `concurrency`: defines the maximum concurrency of the model.
- `assignments`: accepts multiple `assignment`, defining the specific allocation of tasks.
- `output`: defines the path of the output file.

A single `assignment` requires two fields:

- `agent`: the name of the agent required for this task.
- `task`: the name of the task required for this task.

### agents

The `agents` directory contains all agent configuration files. The key in the configuration is the agent's name, and the
value is the agent's configuration. A single agent configuration requires the following fields:

- `module`: defines the corresponding agent client module.
- `parameters`: defines the parameters to be passed to the corresponding module.

### tasks

The `tasks` directory contains all task configuration files. The `task_assembly.yaml` collects all task definitions. If
you only want to run existing tasks, you generally do not need to modify the files in this directory.

Similar to the agent configuration, the key is the task's name, and the value is the task's configuration. A single task
configuration requires the following fields:

- `module`: defines the corresponding task module.
- `parameters`: defines the parameters to be passed to the corresponding module.

### start_task.yaml

This configuration file is used in conjunction with `src.start_task` to automate the bulk launch of task_workers. This
file's fields are as follows:

- `definition`: used to define tasks, usually imported from `task_assembly.yaml`.
- `start(Optional)`: used to specify the tasks to start, the key is the task name, and the value is the number of
  workers to start.
- `controller_address(Optional)`: used to specify the address of the controller, default is http://localhost:5000/api/
