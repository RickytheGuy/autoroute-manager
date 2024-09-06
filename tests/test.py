import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

if __name__ == '__main__':
    from autoroute_manager.autoroute import AutoRoute
    ah = AutoRoute("/Users/ricky/autoroute-manager/config.yaml")
    ah.run()
    