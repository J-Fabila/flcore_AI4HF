import torch 
import sys
import subprocess
import time

if len(sys.argv) > 1:
    num_clients = int(sys.argv[1])
else:
    print("Number of clients is needed as command line argument")
    exit()

try:
    print("Starting server")
    server_process = subprocess.Popen("python src/server.py configs/server.yaml", shell=True)
    time.sleep(20)

    client_processes = []
    for i in range(1, num_clients + 1):
        print("Starting client " + str(i))
        client_processes.append(
            subprocess.Popen("python src/client.py configs/client_" + str(i)+".yaml", shell=True)
        )

    server_process.wait()

except KeyboardInterrupt:
    server_process.terminate()
    server_process.wait()
    for client_process in client_processes:
        client_process.terminate()
        client_process.wait()
   
    print("Server and clients stopped")
