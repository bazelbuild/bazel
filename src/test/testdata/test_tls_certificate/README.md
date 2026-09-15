# TLS self-signed certificates for integration tests.

**Valid until**: Jun 25 11:43:59 2999 GMT

**Generate on**: Apr 24 11:43:59 2019 GMT

**Generated with the script from**: https://github.com/grpc/grpc-java/tree/master/examples/example-tls

```bash
mkdir -p /tmp/sslcert
pushd /tmp/sslcert
# Changes these CN's to match your hosts in your environment if needed.
SERVER_CN=localhost
CLIENT_CN=localhost # Used when doing mutual TLS

echo Generate CA key:
openssl genrsa -passout pass:1111 -des3 -out ca.key 4096
echo Generate CA certificate:
# Generates ca.crt which is the trustCertCollectionFile
openssl req -passin pass:1111 -new -x509 -days 358000 -key ca.key -out ca.crt -subj "/CN=${SERVER_CN}"
echo Generate server key:
openssl genrsa -passout pass:1111 -des3 -out server.key 4096
echo Generate server signing request:
openssl req -passin pass:1111 -new -key server.key -out server.csr -subj "/CN=${SERVER_CN}"
echo Self-signed server certificate:
# Generates server.crt which is the certChainFile for the server
openssl x509 -req -passin pass:1111 -days 358000 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt
echo Remove passphrase from server key:
openssl rsa -passin pass:1111 -in server.key -out server.key
echo Generate client key
openssl genrsa -passout pass:1111 -des3 -out client.key 4096
echo Generate client signing request:
openssl req -passin pass:1111 -new -key client.key -out client.csr -subj "/CN=${CLIENT_CN}"
echo Self-signed client certificate:
# Generates client.crt which is the clientCertChainFile for the client (need for mutual TLS only)
openssl x509 -passin pass:1111 -req -days 358000 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt
echo Remove passphrase from client key:
openssl rsa -passin pass:1111 -in client.key -out client.key
echo Converting the private keys to X.509:
# Generates client.pem which is the clientPrivateKeyFile for the Client (needed for mutual TLS only)
openssl pkcs8 -topk8 -nocrypt -in client.key -out client.pem
# Generates server.pem which is the privateKeyFile for the Server
openssl pkcs8 -topk8 -nocrypt -in server.key -out server.pem
popd
```
