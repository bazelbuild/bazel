# Remote caching and execution with Bazel

Bazel can be configured to use a remote cache and to execute build and test actions remotely.

# Remote Caching

## Overview

A Bazel build consists of many actions. Each action is defined by the command line, the environment variables, the list of input files and hashes of their contents, and the list of output files or output directories. The result of an action is a complete list of output files and hashes of their contents. Bazel can use a remote cache to store and lookup action results. Conceptually, the remote cache consists of two parts: (1) a map of action hashes to action results, and (2) a [content-addressable store](https://en.wikipedia.org/wiki/Content-addressable_storage) (CAS) of output files.

When it's configured to use a remote cache, Bazel computes a hash for each action, and then looks up the hash in the cache. If the cache already contains a result for that action, then Bazel downloads the output files from the CAS. Otherwise it executes the action locally, uploads the files to the CAS, and then adds the action result to the cache.

Starting at version 0.5.0, Bazel supports two caching protocols:

1. a HTTP-based REST protocol
2. [a gRPC-based protocol](https://github.com/googleapis/googleapis/blob/master/google/devtools/remoteexecution/v1test/remote_execution.proto)

## Remote caching using the HTTP REST protocol

If all you need is just distributed caching this is probably the most reliable path as the REST APIs are simple and will remain stable.

For a quick setup, you can use Hazelcast's REST interface. Alternatively you can use the NGINX server with WebDAV, or the Apache HTTP Server with WebDAV.

### Bazel Setup

We recommend editing your `~/.bazelrc` to enable remote caching using the HTTP REST protocol. You will need to replace `http://server-address:port/cache` with the correct address for your HTTP REST server:

```
startup --host_jvm_args=-Dbazel.DigestFunction=SHA1
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

Bazel currently supports the following digest functions with the remote worker: SHA1, SHA256, and MD5. The digest function is passed via the `--host_jvm_args=-Dbazel.DigestFunction=###` startup option. In the example above, SHA1 is used, but you can use any one of SHA1, SHA256, and MD5, provided that your remote execution server supports it and is configured to use the same one. For example, the provided remote worker (`//src/tools/remote_worker`) is configured to use SHA1 by default in the binary build rule. You can customize it there by modifying the `jvm_flags` attribute to use, for example, `"-Dbazel.DigestFunction=SHA256"` instead.


### Hazelcast with REST interface

[Hazelcast](https://hazelcast.org/) is a distributed in-memory cache which can be used by Bazel as a remote cache.

A simple single-machine setup is to run a single Hazelcast server with REST enabled. The REST endpoint will be `http://localhost:5701/hazelcast/rest/maps/cache`. Run with:

```
java -cp third_party/hazelcast/hazelcast-3.6.4.jar -Dhazelcast.rest.enabled=true com.hazelcast.core.server.StartServer
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
startup --host_jvm_args=-Dbazel.DigestFunction=SHA1
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

Bazel currently provides a sample gRPC CAS implementation with a SimpleBlobStore or Hazelcast as caching backend. To use it you need to clone from [Bazel](https://github.com/bazelbuild/bazel) and then build it with:

```
bazel build //src/tools/remote_worker
```

The following command will then start the cache server listening on port 8080 using a local in-memory cache:

```
bazel-bin/src/tools/remote_worker/remote_worker --listen_port=8080
```

To connect to a running instance of Hazelcast instead, use:

```
bazel-bin/src/tools/remote_worker/remote_worker --listen_port=8080 --hazelcast_node=address:port
```

If you want to change Hazelcast settings to enable distributed memory cache you can provide your own hazelcast.xml with the following command:

```
bazel-bin/src/tools/remote_worker/remote_worker --jvm_flags=-Dhazelcast.config=/path/to/hz.xml --listen_port 8080
```

You can copy and edit the [default](https://github.com/hazelcast/hazelcast/blob/master/hazelcast/src/main/resources/hazelcast-default.xml) Hazelcast configuration. Refer to Hazelcast [manual](http://docs.hazelcast.org/docs/3.6/manual/html-single/index.html#checking-configuration)
for more details.
