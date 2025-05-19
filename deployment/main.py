"""import litserve as ls
from deployment.online.api import InferenceAPI

if __name__ == "__main__":
    ls.serve(InferenceAPI)
"""


"""import litserve as ls
print(dir(ls))"""

import litserve as ls
from deployment.online.api import InferenceAPI

if __name__ == "__main__":
    api_instance = InferenceAPI()
    server = ls.LitServer(api_instance)
    server.run()

