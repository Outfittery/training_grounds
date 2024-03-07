import os
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


@dataclass(frozen=True)
class Environment:
    name: str
    extra: str
    input_files: List[str]
    output_file: str


def discover_python_version():
    import toml

    python_version = (
        toml.load("pyproject.toml").get("project", {}).get("requires-python", "3.8")
    )
    python_version = re.search(r"([0-9.]+)", python_version)
    if python_version:
        python_version = python_version.group()
        print("Discovered python version from pyproject.toml:", python_version)
    else:
        python_version = "3.8"
        print("Could not discover python version from pyproject.toml, using 3.8.")

    return python_version


def docker_run(image: str, command: List[str]):
    import docker

    print("Running in docker image:", image)
    docker_client = docker.from_env()

    container = docker_client.containers.run(
        image=image,
        command=command,
        auto_remove=True,
        detach=True,
        platform="linux/amd64",
        working_dir="/app",
        volumes=[f"{os.getcwd()}:/app"],
    )
    for log in container.logs(stream=True):
        print(log.decode())


def run(command: List[str]):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    result = result.returncode
    if result == 0:
        print("Done.", end="\n\n")
    else:
        print("Failed.", end="\n\n")
        exit(1)


def install(
    libraries: List[str],
    editable: bool = False,
    deps: bool = True,
    upgrade: bool = False,
):
    command = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        command.append("--upgrade")
    if not deps:
        command.append("--no-deps")
    if editable:
        command.append("-e")
    command.extend(libraries)
    run(command)


UNSAFE_PACKAGES = [
    "pip",
    "wheel",
    "setuptools",
    "pip-tools",
    "appnope",
]


def compile(
    input_files: List[str],
    extra: Optional[str],
    output_file: str,
    upgrade: Union[bool, List[str]] = False,
):
    command = ["pip-compile"]
    for package in UNSAFE_PACKAGES:
        command.append(f"--unsafe-package={package}")
    if upgrade:
        if isinstance(upgrade, bool):
            command.append("--upgrade")
        elif isinstance(upgrade, list):
            command.extend([f"--upgrade-package={package}" for package in upgrade])
    command.extend(input_files)
    if extra:
        command.append(f"--extra={extra}")
    command.extend(["-o", output_file])
    run(command)


def sync(requirements_file: str):
    if not os.path.isfile(requirements_file):
        print(
            f"File {requirements_file} does not exist. If it is a new repository, run `compile` first."
        )
        exit(1)
    run(["pip-sync", requirements_file])


FUNDAMENTAL_LIBRARIES = [
    "pip",
    "wheel",
    "setuptools",
    "pip-tools",
    "typer[all]",
    "docker",
    "toml",
]


def get_typer_app(envs: List[Environment]):
    print("Installing fundamental libraries...")
    install(libraries=FUNDAMENTAL_LIBRARIES, upgrade=True)

    import typer

    env_dict = {e.name: e for e in envs}
    env_enum = Enum("Environments", {e.name: e.name for e in envs})

    app = typer.Typer(help="Manage project requirements.")

    @app.callback()
    def main(
        ctx: typer.Context,
        run_in_docker: bool = typer.Option(
            default=False, help="Run the command in a docker container."
        ),
    ):
        if run_in_docker:
            image = f"python:{discover_python_version()}"
            command = ["python"] + list(
                filter(lambda x: x != "--run-in-docker", sys.argv)
            )
            docker_run(image=image, command=command)
            raise typer.Exit()

    @app.command(name="compile")
    def compile_requirements(
        upgrade_all: bool = typer.Option(
            default=False, help="Upgrade all the packages to the latest version."
        ),
        upgrade_package: List[str] = typer.Option(
            default=[], help="Upgrade a specific package to the latest version."
        ),
    ):
        """Compiles the requirements files."""
        upgrade = upgrade_package if upgrade_package else upgrade_all
        for env in envs:
            compile(
                input_files=env.input_files,
                extra=env.extra,
                output_file=env.output_file,
                upgrade=upgrade,
            )

    @app.command(name="install")
    def install_app(
        env: env_enum = typer.Option(
            default="dev", help="Environment to install requirements for."
        ),
        editable: bool = typer.Option(default=True, help="Install in editable mode."),
    ):
        """Installs the requirements for the app and the app itself."""
        sync(requirements_file=env_dict[env.name].output_file)
        install(libraries=["."], editable=editable, deps=False)

    return app
