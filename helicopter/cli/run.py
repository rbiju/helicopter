import click

from .entrypoint import helicopter


@helicopter.command()
@click.option('-n', '--name', required=True, type=click.STRING, help="Name of task. Must be a subclass of helicopter.tasks.Task")
@click.option('-c', '--config', type=click.STRING, help="(Optional) path to configuration file")
@click.option('-d', '--debug', is_flag=True, default=False, help="Enable debug mode")
def run(name: str, config: str = None, debug: bool = False):
    import importlib
    import inspect
    import pkgutil

    from helicopter.configuration import LocalHydraConfiguration
    from helicopter.tasks.base import Task, get_config_path
    import helicopter.tasks as tasks

    def task_exists(class_name: str) -> bool:
        for _, module_name, _ in pkgutil.iter_modules(tasks.__path__):
            module = importlib.import_module(f"{tasks.__name__}.{module_name}")
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                        _name == class_name
                        and issubclass(obj, Task)
                        and obj is not Task
                        and obj.__module__ == module.__name__
                ):
                    return True
        return False

    if not task_exists(name):
        raise ValueError(f"Object {name} does not exist or is not a Task")

    try:
        if config:
            config_path = config
        else:
            config_path = get_config_path(name)
        configuration = LocalHydraConfiguration(config_path)
    except FileNotFoundError:
        click.echo(f"Task configuration for '{name}' not found")
        return

    task: Task = configuration.resolve(list(configuration.cfg.keys())[0])
    if not debug:
        click.echo(f"Running task {type(task).__name__}")
        task.run(configuration_path=config_path)
    else:
        click.echo(f"Running task {type(task).__name__} in debug mode")
        task.debug_run(configuration_path=config_path)


if __name__ == "__main__":
    from click.testing import CliRunner

    test_args = ["run", "--name", "Measure", "-c", "/home/ray/projects/helicopter/configs/tasks/measure.yaml", "-d"]

    runner = CliRunner()
    result = runner.invoke(helicopter(), test_args)
