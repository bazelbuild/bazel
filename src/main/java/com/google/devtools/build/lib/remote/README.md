# Remote caching and execution with Bazel

Bazel can be configured to use a remote cache and to execute build and test actions remotely.

# Remote Caching

## Overview

A Bazel build consists of actions. One can think of an action as i.e. a compiler invocation. An action is defined by its command line, environment variables, its input files, and its output filenames. The result of an action is a complete list of the output filenames and hashes of their contents. Bazel can use a remote cache to store and lookup said action results and the outputs it references. Conceptually, the remote cache consists of two parts: (1) a map of action hashes to action results, and (2) a [content-addressable store](https://en.wikipedia.org/wiki/Content-addressable_storage) (CAS) of output files.

Remote caching works by Bazel looking up the hash of an action in the remote cache, and if successful retrieving the action result and the output files it references. If the lookup fails Bazel executes the action locally, uploads the output files to the CAS, and stores a list of output files keyed by the hash of the action in the action cache.

Bazel supports two caching protocols:

1. A HTTP-based REST protocol
2. [A gRPC-based protocol](https://github.com/googleapis/googleapis/blob/master/google/devtools/remoteexecution/v1test/remote_execution.proto)

## Remote caching using the HTTP REST protocol

The HTTP-based caching protocol is the recommended protocol to use for remote caching. The protocol uses HTTP PUT for uploads and HTTP GET for downloads. The action cache is expected under `/ac` and the CAS is expected under `/cas`.

For example, consider a remote cache running under `localhost:8080`. A request to fetch an action result from the action cache might look like below.

```
GET /ac/01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b HTTP/1.1
Host: localhost
```

An upload to the CAS might look as follows.

```
PUT /ac/01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b HTTP/1.1
Host: localhost
Content-Length: 10
Content-Type: application/octet-stream
```

Users have had success using a diverse set of caching backends including Hazelcast and NGINX (with WebDAV).

### Known Issues

When an input file is modified during a build, Bazel might upload invalid results to the remote cache. We are working on a solution for this problem. Please watch [#3360](https://github.com/bazelbuild/bazel/issues/3360) for updates. One can avoid this problem by not editing source files during a build.

### Bazel Setup

We recommend editing your `~/.bazelrc` to enable remote caching using the HTTP REST protocol. You will need to replace `http://server-address:port/cache` with the correct address for your HTTP REST server:

```
build --spawn_strategy=remote
build --remote_rest_cache=REPLACE_THIS:http://server-address:port/cache
# Bazel currently doesn't support remote caching in combination with workers.
# These two lines override the default strategy for Javac and Closure
# actions, so that they are also remotely cached.
# Follow https://github.com/bazelbuild/bazel/issues/3027 for more details:
build --strategy=Javac=remote
build --strategy=Closure=remote
```

#### Customizing The Digest Function

Bazel currently supports the following digest functions with the remote worker: SHA1, SHA256, and MD5. It defaults to SHA256. The digest function is passed via the `--host_jvm_args=-Dbazel.DigestFunction=###` startup option.

### Hazelcast with REST interface

[Hazelcast](https://hazelcast.org/) is a distributed in-memory cache which can be used by Bazel as a remote cache. You can download the standalone Hazelcast server [here](https://hazelcast.org/download/).

A simple single-machine setup is to run a single Hazelcast server with REST enabled. The REST endpoint will be `http://localhost:5701/hazelcast/rest/maps/`. Run the Hazelcast server with REST using this command:

```
java -cp hazelcast-all-3.8.5.jar -Dhazelcast.rest.enabled=true com.hazelcast.core.server.StartServer
```

You can also use Bazel with a Hazelcast cluster - as long as REST is enabled -, and also customize the configuration. Please see the Hazelcast [documentation](http://docs.hazelcast.org/docs/3.6/manual/html-single/index.html) for more details.

### NGINX with WebDAV

First you need to set up NGINX with WebDAV support. On Debian or Ubuntu Linux, you can install the `nginx-extras` package. On OSX you can install the [`nginx-full`](https://github.com/Homebrew/homebrew-nginx) package from homebrew with `brew install nginx-full --with-webdav`.

Once installed, edit nginx.conf with a section for uploading and serving cache objects.

```
location /cache/ {
    root /some/document/root;
    dav_methods PUT;
    autoindex on;
    allow all;
    client_max_body_size 256M;
}
```

You will need to change `/some/document/root` to a valid directory where NGINX can write to and
read from. You may need to change `client_max_body_size` option to a larger value in case the cache
object is too large.

### Apache HTTP Server with WebDAV module

Assuming Apache HTTP Server is installed with DAV modules installed. You need to edit `httpd.conf` to enable the following modules:

```
LoadModule dav_module libexec/apache2/mod_dav.so
LoadModule dav_fs_module libexec/apache2/mod_dav_fs.so
```

Edit `httpd.conf` to use a directory for uploading and serving cache objects. You may want to edit
this directory to include security control.

```
<Directory "/some/directory/for/cache">
    AllowOverride None
    Require all granted
    Options +Indexes

    Dav on
    <Limit HEAD OPTIONS GET POST PUT DELETE>
        Order Allow,Deny
        Allow from all
    </Limit>
    <LimitExcept HEAD OPTIONS GET POST PUT DELETE>
        Order Deny,Allow
        Deny from all
    </LimitExcept>
</Directory>
```

## Remote caching using the gRPC protocol

We're working on a [gRPC protocol](https://github.com/googleapis/googleapis/blob/master/google/devtools/remoteexecution/v1test/remote_execution.proto)
that supports both remote caching and remote execution. As of this writing, there is only a single server-side implementation, which is not intended for production use.

### Bazel Setup

We recommend editing your `~/.bazelrc` to enable remote caching using the gRPC protocol. Use the following build options to use the gRPC CAS endpoint for sharing build artifacts. Change `REPLACE_THIS:address:8080` to the correct server address and port number.

```
build --spawn_strategy=remote
build --remote_cache=REPLACE_THIS:address:8080
# Bazel currently doesn't support remote caching in combination with workers.
# These two lines override the default strategy for Javac and Closure
# actions, so that they are also remotely cached.
# Follow https://github.com/bazelbuild/bazel/issues/3027 for more details:
build --strategy=Javac=remote
build --strategy=Closure=remote
```

### Running the sample gRPC cache server

Bazel currently provides a sample gRPC CAS implementation with a SimpleBlobStore as caching backend. To use it you need to clone from [Bazel](https://github.com/bazelbuild/bazel) and then build it with:

```
bazel build //src/tools/remote:worker
```

The following command will then start the cache server listening on port 8080 using a local in-memory cache:

```
bazel-bin/src/tools/remote/worker --listen_port=8080
```
