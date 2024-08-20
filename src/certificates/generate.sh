#!/bin/bash

# Generate root CA private key
openssl genpkey -algorithm RSA -out rootCA_key.pem

# Generate root CA certificate
openssl req -new -x509 -key rootCA_key.pem -out rootCA_cert.pem -days 36500 -subj "/CN=Root CA/O=UB/C=ES"

# Generate server private key
openssl genpkey -algorithm RSA -out server_key.pem

# Generate server certificate signing request (CSR)
openssl req -new -key server_key.pem -out server_csr.pem -subj "/CN=161.116.4.119/O=UB/C=ES"

# Generate server certificate
openssl x509 -req -in server_csr.pem -CA rootCA_cert.pem -CAkey rootCA_key.pem -out server_cert.pem -set_serial $(date +%s) -days 36500 -extfile <(printf "subjectAltName=DNS:161.116.4.119")

#-extfile <(printf "subjectAltName=DNS:bcnaim.ub.edu")
