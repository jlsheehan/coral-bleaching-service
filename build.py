#   -*- coding: utf-8 -*-
import glob
import os
import shutil

from pybuilder.core import use_plugin, init, Project, Logger, task

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.distutils")


name = "coral-bleaching-service"
default_task = "publish"
version = "1.0.9"


@init
def set_properties(project: Project):
    project.depends_on_requirements("requirements.txt")


@task
def copy_package(project: Project, logger: Logger):
    dir_dist = os.path.join(project.expand_path("$dir_dist"), "dist")
    user_home = os.path.expanduser("~")
    packages_dir = os.path.join(user_home, "Development", "packages")
    files = glob.glob(f"{dir_dist}/*.whl")
    if len(files) == 1:
        logger.info("Copying package %s", files[0])
        shutil.copy(
            files[0],
            packages_dir,
        )
    else:
        logger.error("No file found to copy")
