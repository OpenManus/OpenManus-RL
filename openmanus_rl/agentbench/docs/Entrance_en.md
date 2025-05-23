# Framework Entry Points

[🌏中文版](Entrance_cn.md)

The main entry points of the framework are:

- `src.server.task_controller`: For manually starting the task_controller.
- `src.start_task`: For starting the task_worker.
- `src.assigner`: For launching evaluations.
- `src.server.task_worker`: For manually starting the task_worker.

## src.server.task_controller

The task_controller is the core of the task server, responsible for managing all task_workers. It should be started
first and is recommended to be always on. It's also advised to keep it globally unique unless necessary. By default,
task_controller runs on port 5000, but you can also specify with the `--port -p` argument. All interfaces have a unified
prefix `/api/`.

Example of starting the task_controller on port 3000:

```bash
python -m src.server.task_controller -p 3000
```

The task_controller has the following monitoring interfaces:

| Interface      | Method | Parameters | Description                                                                                     |
|----------------|--------|------------|-------------------------------------------------------------------------------------------------|
| /list_workers  | GET    | None       | Returns all task_workers                                                                        |
| /list_sessions | GET    | None       | Returns all sessions                                                                            |
| /sync_all      | POST   | None       | Syncs all sessions running on task_workers, call this first if controller restarts unexpectedly |
| /cancel_all    | POST   | None       | Cancels all sessions running on task_workers                                                    |

## src.start_task

The start_task is a script for starting the task_worker. Its main function is to read the configuration file and start
the task_worker. The configuration file for start_task is `configs/start_task.yaml`. More details can be found in the
configuration file documentation.

Parameters for start_task:

- `[--config CONFIG]`: Specifies the configuration file to read. The default is `configs/start_task.yaml`, usually no
  need to change.
- `[--start | -s [TASK_NAME NUM [TASK_NAME NUM ...]]]`: Specifies tasks to start. The format is `TASK_NAME NUM`
  where `TASK_NAME` is the task name and `NUM` is the number of workers to start. If this parameter is specified, it
  will override **all** settings in the configuration file.
- `[--auto-controller | -a]`: Specifies whether to automatically start the task_controller. Default is off.
- `[--base-port | -p PORT]`: Specifies the base port for task_worker. Default is 5001. Task_workers will start
  sequentially from PORT. If there are N task_workers, then their ports will range from PORT to PORT+N-1.

## src.assigner

The assigner script starts evaluations. It reads the configuration file, launches the evaluations, and saves results in
real-time to the specified output directory.

Parameters for assigner:

- `[--config CONFIG]`: Specifies the configuration file to read. Default is `configs/assignments/default.yaml`.
- `[--auto-retry]`: Auto retry failed samples.

If the `output` field in the configuration contains `{TIMESTAMP}`, it will be replaced with the current time for
subsequent operations. If the directory specified in the `output` field already exists, the assigner will attempt to
read the existing evaluation results and continue the evaluation.

Every time the assigner is launched, it will parse the read configuration and save it to the directory specified in
the `output` field. **If a configuration file already exists in the directory, it will be overwritten**.

## src.server.task_worker

A task_worker corresponds to a task process. The same task can have multiple task_workers. It's **not recommended** to
manually start the task_worker unless necessary; instead, use `src.start_task`.

Parameters for task_worker:

- `NAME` is the task name, specifying which task to start.
- `[--config | -c CONFIG]` Specifies the configuration file to read. Default is `configs/tasks/task_assembly.yaml`.
- `[--port | -p PORT]` Specifies the port for the task_worker. Default is 5001.
- `[--controller | -C ADDRESS]` Specifies the address of the task_controller. Default is http://localhost:5000/api.
- `[--self ADDRESS]` Specifies the address of the task_worker. Default is http://localhost:5001/api. This address is
  used by the task_controller to communicate with the task_worker, so make sure the task_controller can access this
  address.
