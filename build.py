#   -*- coding: utf-8 -*-
import shutil

from pybuilder.core import use_plugin, init, Project, Logger, task

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.distutils")


name = "coral-bleaching-service"
default_task = "publish"
version="0.8.0"

@init
def set_properties(project: Project):
    project.depends_on_requirements("requirements.txt")


@task
def copy_package(project: Project, logger: Logger):
    shutil.copy(
        project.expand_path(
            f"$dir_dist/dist/coral_bleaching_service-{version}-py3-none-any.whl"
        ),
        f"/Users/jeffreysheehan/Development/packages/coral_bleaching_service-{version}-py3-none-any.whl",
    )
