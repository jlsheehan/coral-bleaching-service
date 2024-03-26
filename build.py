#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init, Project

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.distutils")


name = "coral-bleaching-service"
default_task = "publish"
version="0.1.0"

@init
def set_properties(project: Project):
    project.build_depends_on_requirements("requirements.txt")
