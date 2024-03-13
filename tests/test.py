import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from autoroute.autoroute import AutoRouteHandler
AutoRouteHandler.run("/Users/ricky/autoroute-manager/config.yaml")