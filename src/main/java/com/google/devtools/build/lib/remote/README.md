# Introduction

This directory includes the implementation for distributed caching support in Bazel and support for
remote execution.

# Design

The detailed design document and discussion can be found in this forum thread.

https://groups.google.com/forum/#!msg/bazel-discuss/7JSbF6DT6OU/ewuXO6ydBAAJ

# Bazel Options

This section summarizes the options available in .bazelrc for distributed caching and remote
execution support.

* ```startup --host_jvm_args=-Dbazel.DigestFunction=SHA1```

This option is always needed to support distributed caching and remote execution.

* ```build --spawn_strategy=remote --remote_rest_cache=http://remote-cache:8080/cache```

This option enables distributed caching with a REST endpoint that supports GET, HEAD and PUT.

* ```build --spawn_strategy=remote --remote_cache=remote-cache:8081```

This option enables distributed caching using a gRPC content-addressable storage (CAS) service.

* ```build --spawn_strategy=remote --hazelcast_node=remote-cache:5701```

This option enables distributed caching using Hazelcast memory cluster as a content-addressable storage (CAS). Please watch for future announcement as this might be removed in favor of the REST endpoint.

* ```build --spawn_strategy=remote --remote_executor=grpc-builder:5000 --remote_cache=grpc-builder:5000```

This option enables remote execution with a gRPC service at ```grpc-builder:5000```. Remote execution requires a distributed caching service, which is also at ```grpc-builder:5000```.

# Distributed Caching

## Overview

Distributed caching support in Bazel depends heavily on [content-addressable storage](https://en.wikipedia.org/wiki/Content-addressable_storage).

A Bazel build consists of many actions. An action is defined by the command to execute, the
arguments and a list of input files. Before executing an action Bazel computes a hash code from
an action. This hash code will be used to lookup and index the output from executing the action.
Bazel will lookup the hash code in the content-addressable storage (CAS) backend. If there is a
match then the output files are downloaded. If there is no match the action will be executed and
the output files will be uploaded to the CAS backend.

There are 2 kinds of CAS backend support implemented in Bazel.

* REST endpoint that supports PUT, GET and HEAD.
* gRPC endpoint that implements the [distributed caching and remote execution protocol](https://github.com/bazelbuild/bazel/blob/master/src/main/protobuf/remote_protocol.proto).

### Distributed caching with REST endpoint

If all you need is just distributed caching this is probably the most reliable path as the REST
APIs are simple and will remain stable.

For quick setup you can use Hazelcast's REST interface. Alternatively you can also use NGINX with
WebDav module or Apache HTTP Server with WebDav enabled. This enables simple remote caching for
sharing between users.

#### Initial setup

You should enable SHA1 digest for Bazel with distributed caching. Edit `~/.bazelrc` and put the
following line:
```
startup --host_jvm_args=-Dbazel.DigestFunction=SHA1
```

#### Hazelcast with REST interface

We made it easy by providing Hazelcast in Bazel and it provides a REST interface for caching purposes.

Run using the following command and it will listen to port 5701. The REST endpoint will be ```http://localhost:5701/hazelcast/rest/maps/cache```.
```
bazel build //third_party:hazelcast_rest_server
bazel-bin/third_party/hazelcast_rest_server
```

You can customize the Hazelcast server by providing a configuration file during startup.
```
bazel-bin/third_party/hazelcast_rest_server --jvm_flags=-Dhazelcast.config=/path/to/hz.xml
```
You can copy and edit the [default](https://github.com/hazelcast/hazelcast/blob/master/hazelcast/src/main/resources/hazelcast-default.xml) Hazelcast configuration. Refer to Hazelcast [manual](http://docs.hazelcast.org/docs/3.6/manual/html-single/index.html#checking-configuration)
for more details.

#### NGINX with WebDav module

First you need to set up NGINX with WebDav support. On Debian or Ubuntu Linux you can install
`nginx-extras` package. On OSX you can install the [`nginx-full`](https://github.com/Homebrew/homebrew-nginx) package from
homebrew with `brew install nginx-full --with-webdav`.

Once installed, edit nginx.conf with a section for uploading and serving cache objects.

```
location /cache/ {
    root   /some/document/root;
    dav_methods PUT;
    autoindex on;
    allow all;
    client_max_body_size 256M;
}
```

You will need to change `/some/document/root` to a valid directory where NGINX can write to and
read from. You may need to change `client_max_body_size` option to a larger value in case the cache
object is too large.

#### Apache HTTP Server with WebDav module

Assuming Apache HTTP Server is installed with Dav modules installed. You need to edit `httpd.conf`
to enable the following modules:
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

#### Providing your own REST endpoint

Any REST endpoint with GET, PUT and HEAD support will be sufficient. GET is used to fetch a cache
object. PUT is used to upload a cache object and HEAD is used to check the existence of a cache
object.

#### Running Bazel with REST CAS endpoint

Once you have a REST endpoint that supports GET, PUT and HEAD then you can run Bazel with the
following options to enable distributed caching. Change `http://server-address:port/cache` to the
one that you provide. You may also put the options in `~/.bazelrc`.

```
build --spawn_strategy=remote --remote_rest_cache=http://server-address:port/cache
```

### Distributed caching with gRPC CAS endpoint

A gRPC CAS endpoint that implements the [distributed caching and remote execution protocol](https://github.com/bazelbuild/bazel/blob/master/src/main/protobuf/remote_protocol.proto) will
give the best performance and is the most actively developed distributed caching solution.

#### Initial setup

You should enable SHA1 digest for Bazel with distributed caching. Edit `~/.bazelrc` and put the
following line:
```
startup --host_jvm_args=-Dbazel.DigestFunction=SHA1
```

#### Running the sample gRPC cache server

Bazel currently provides a sample gRPC CAS implementation with a SimpleBlobStore or Hazelcast as caching backend.
To use it you need to clone from [Bazel](https://github.com/bazelbuild/bazel) and then build it.
```
bazel build //src/tools/remote_worker:all
```

The following command will then start the cache server listening on port 8080 using a local in-memory cache.
```
bazel-bin/src/tools/remote_worker/remote_worker --listen_port=8080
```

To connect to a running instance of Hazelcast instead, use.
```
bazel-bin/src/tools/remote_worker/remote_worker --listen_port=8080 --hazelcast_node=address:port
```

If you want to change Hazelcast settings to enable distributed memory cache you can provide your
own hazelcast.xml with the following command.
```
bazel-bin/src/tools/remote_worker/remote_worker --jvm_flags=-Dhazelcast.config=/path/to/hz.xml --listen_port 8080
```
You can copy and edit the [default](https://github.com/hazelcast/hazelcast/blob/master/hazelcast/src/main/resources/hazelcast-default.xml) Hazelcast configuration. Refer to Hazelcast [manual](http://docs.hazelcast.org/docs/3.6/manual/html-single/index.html#checking-configuration)
for more details.

#### Using the gRPC CAS service

Use the following build options to use the gRPC CAS endpoint for sharing build artifacts. Change
`address:8080` to the correct server address and port number.

```
build --spawn_strategy=remote --remote_cache=address:8080
```

### Distributed caching with Hazelcast (TO BE REMOVED)

Bazel can connect to a Hazelcast distributed memory cluster directly for sharing build artifacts.
This feature will be removed in the future in favor of the gRPC protocol for distributed caching.
Hazelcast may still be used as a distributed caching backend but Bazel will connect to it through
a gRPC CAS endpoint.

#### Starting a Hazelcast server

If you do not already have a Hazelcast memory cluster you can clone [Bazel](https://github.com/bazelbuild/bazel) and run this command:

```
java -cp third_party/hazelcast/hazelcast-3.6.4.jar com.hazelcast.core.server.StartServer
```

#### Using Hazelcast as distributed cache

You will need to put the following line in `~/.bazelrc`.
```
startup --host_jvm_args=-Dbazel.DigestFunction=SHA1
```

The following build options will use Hazelcast as a distributed cache during build. Change
`address:5701` to the actual server address assuming Hazelcast listens to port 5701.
```
build --hazelcast_node=address:5701 --spawn_strategy=remote
```

# Remote Execution (For Demonstration Only)

The implementation of remote execution worker in Bazel can only serve as a demonstration. The
client-side implementation is being actively developed in Bazel. However there is no fully
functional implementation of remote worker yet.

## Initial setup

You should enable SHA1 digest for Bazel with distributed caching. Edit `~/.bazelrc` and put the
following line:
```
startup --host_jvm_args=-Dbazel.DigestFunction=SHA1
```

## Running the sample gRPC remote worker / cache server
```
bazel build //src/tools/remote_worker:remote_worker
bazel-bin/src/tools/remote_worker/remote_worker --work_path=/tmp --listen_port 8080
```

The sample gRPC cache server and gRPC remote worker share the **same
distributed memory cluster** for storing and accessing CAS objects. It is important the CAS objects
are shared between the two server processes.

You can modify hazelcast configuration by providing a `hazelcast.xml`. Please refer to Hazelcast
manual for details. Make sure the cache server and the remote worker server shares the same
memory cluster.

## Running Bazel using gRPC for caching and remote execution

Use the following build options.
```
build --spawn_strategy=remote --remote_executor=localhost:8080 --remote_cache=localhost:8080
```
