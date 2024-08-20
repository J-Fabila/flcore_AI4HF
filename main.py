import sys
import subprocess
import time

from src.utils import Parameters

if len(sys.argv) > 1:
    params = Parameters()
    config_file = sys.argv[1]
    params.GetParams(config_file)
else:
    print("*.yaml input file needed as command line argument")
    exit()

if params.federated == True:
    if params.set_server == True:
        try:
            print("Starting server")
            server_process = subprocess.Popen("python src/server.py "+config_file, shell=True)
            time.sleep(20)
            server_process.wait()
        except KeyboardInterrupt:
            server_process.terminate()
            server_process.wait()
            print("Server and clients stopped")
    else:
        try:
            print("Starting client " + str(params.client_id))
            client_process = subprocess.Popen("python src/client.py "+config_file, shell=True)
            time.sleep(20)
            client_process.wait()
        except KeyboardInterrupt:
            client_process.terminate()
            client_process.wait()
            print("Clients stopped")
else:
    try:
        print("Running centralized")
        centralized_process = subprocess.Popen("python src/centralized.py "+config_file, shell=True)
        time.sleep(20)
        centralized_process.wait()
    except KeyboardInterrupt:
        centralized_process.terminate()
        centralized_process.wait()
        print("Centralized stopped")