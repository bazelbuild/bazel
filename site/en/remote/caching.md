Project: /_project.yaml
Book: /_book.yaml

# Remote Caching

{% include "_buttons.html" %}

This page covers remote caching, setting up a server to host the cache, and
running builds using the remote cache.

A remote cache is used by a team of developers and/or a continuous integration
(CI) system to share build outputs. If your build is reproducible, the
outputs from one machine can be safely reused on another machine, which can
make builds significantly faster.

## Overview {:#overview}

Bazel breaks a build into discrete steps, which are called actions. Each action
has inputs, output names, a command line, and environment variables. Required
inputs and expected outputs are declared explicitly for each action.

You can set up a server to be a remote cache for build outputs, which are these
action outputs. These outputs consist of a list of output file names and the
hashes of their contents. With a remote cache, you can reuse build outputs
from another user's build rather than building each new output locally.

To use remote caching:

* Set up a server as the cache's backend
* Configure the Bazel build to use the remote cache
* Use Bazel version 0.10.0 or later

The remote cache stores two types of data:

* The action cache, which is a map of action hashes to action result metadata.
* A content-addressable store (CAS) of output files.

Note that the remote cache additionally stores the stdout and stderr for every
action. Inspecting the stdout/stderr of Bazel thus is not a good signal for
[estimating cache hits](/remote/cache-local).

### How a build uses remote caching {:#remote-caching}

Once a server is set up as the remote cache, you use the cache in multiple
ways:

* Read and write to the remote cache
* Read and/or write to the remote cache except for specific targets
* Only read from the remote cache
* Not use the remote cache at all

When you run a Bazel build that can read and write to the remote cache,
the build follows these steps:

1. Bazel creates the graph of targets that need to be built, and then creates
a list of required actions. Each of these actions has declared inputs
and output filenames.
2. Bazel checks your local machine for existing build outputs and reuses any
that it finds.
3. Bazel checks the cache for existing build outputs. If the output is found,
Bazel retrieves the output. This is a cache hit.
4. For required actions where the outputs were not found, Bazel executes the
actions locally and creates the required build outputs.
5. New build outputs are uploaded to the remote cache.

## Setting up a server as the cache's backend {:#cache-backend}

You need to set up a server to act as the cache's backend. A HTTP/1.1
server can treat Bazel's data as opaque bytes and so many existing servers
can be used as a remote caching backend. Bazel's
[HTTP Caching Protocol](#http-caching) is what supports remote
caching.

You are responsible for choosing, setting up, and maintaining the backend
server that will store the cached outputs. When choosing a server, consider:

* Networking speed. For example, if your team is in the same office, you may
want to run your own local server.
* Security. The remote cache will have your binaries and so needs to be secure.
* Ease of management. For example, Google Cloud Storage is a fully managed service.

There are many backends that can be used for a remote cache. Some options
include:

* [nginx](#nginx)
* [bazel-remote](#bazel-remote)
* [Google Cloud Storage](#cloud-storage)

### nginx {:#nginx}

nginx is an open source web server. With its [WebDAV module], it can be
used as a remote cache for Bazel. On Debian and Ubuntu you can install the
`nginx-extras` package. On macOS nginx is available via Homebrew:

```posix-terminal
brew tap denji/nginx

brew install nginx-full --with-webdav
```

Below is an example configuration for nginx. Note that you will need to
change `/path/to/cache/dir` to a valid directory where nginx has permission
to write and read. You may need to change `client_max_body_size` option to a
larger value if you have larger output files. The server will require other
configuration such as authentication.


Example configuration for `server` section in `nginx.conf`:

```nginx
location /cache/ {
  # The path to the directory where nginx should store the cache contents.
  root /path/to/cache/dir;
  # Allow PUT
  dav_methods PUT;
  # Allow nginx to create the /ac and /cas subdirectories.
  create_full_put_path on;
  # The maximum size of a single file.
  client_max_body_size 1G;
  allow all;
}
```

### bazel-remote {:#bazel-remote}

bazel-remote is an open source remote build cache that you can use on
your infrastructure. It has been successfully used in production at
several companies since early 2018. Note that the Bazel project does
not provide technical support for bazel-remote.

This cache stores contents on disk and also provides garbage collection
to enforce an upper storage limit and clean unused artifacts. The cache is
available as a [docker image] and its code is available on
[GitHub](https://github.com/buchgr/bazel-remote/){: .external}.
Both the REST and gRPC remote cache APIs are supported.

Refer to the [GitHub](https://github.com/buchgr/bazel-remote/){: .external}
page for instructions on how to use it.

### Google Cloud Storage {:#cloud-storage}

[Google Cloud Storage] is a fully managed object store which provides an
HTTP API that is compatible with Bazel's remote caching protocol. It requires
that you have a Google Cloud account with billing enabled.

To use Cloud Storage as the cache:

1. [Create a storage bucket](https://cloud.google.com/storage/docs/creating-buckets){: .external}.
Ensure that you select a bucket location that's closest to you, as network bandwidth
is important for the remote cache.

2. Create a service account for Bazel to authenticate to Cloud Storage. See
[Creating a service account](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating_a_service_account){: .external}.

3. Generate a secret JSON key and then pass it to Bazel for authentication. Store
the key securely, as anyone with the key can read and write arbitrary data
to/from your GCS bucket.

4. Connect to Cloud Storage by adding the following flags to your Bazel command:
   * Pass the following URL to Bazel by using the flag:
       `--remote_cache=https://storage.googleapis.com{{ '<var>' }}/bucket-name{{ '</var>' }}` where `bucket-name` is the name of your storage bucket.
   * Pass the authentication key using the flag: `--google_credentials={{ '<var>' }}/path/to/your/secret-key{{ '</var>'}}.json`, or
     `--google_default_credentials` to use [Application Authentication](https://cloud.google.com/docs/authentication/production){: .external}.

5. You can configure Cloud Storage to automatically delete old files. To do so, see
[Managing Object Lifecycles](https://cloud.google.com/storage/docs/managing-lifecycles){: .external}.

### Other servers {:#other-servers}

You can set up any HTTP/1.1 server that supports PUT and GET as the cache's
backend. Users have reported success with caching backends such as [Hazelcast](https://hazelcast.com){: .external},
[Apache httpd](http://httpd.apache.org){: .external}, and [AWS S3](https://aws.amazon.com/s3){: .external}.

## Authentication {:#authentication}

As of version 0.11.0 support for HTTP Basic Authentication was added to Bazel.
You can pass a username and password to Bazel via the remote cache URL. The
syntax is `https://username:password@hostname.com:port/path`. Note that
HTTP Basic Authentication transmits username and password in plaintext over the
network and it's thus critical to always use it with HTTPS.

## HTTP caching protocol {:#http-caching}

Bazel supports remote caching via HTTP/1.1. The protocol is conceptually simple:
Binary data (BLOB) is uploaded via PUT requests and downloaded via GET requests.
Action result metadata is stored under the path `/ac/` and output files are stored
under the path `/cas/`.

For example, consider a remote cache running under `http://localhost:8080/cache`.
A Bazel request to download action result metadata for an action with the SHA256
hash `01ba4719...` will look as follows:

```http
GET /cache/ac/01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b HTTP/1.1
Host: localhost:8080
Accept: */*
Connection: Keep-Alive
```

A Bazel request to upload an output file with the SHA256 hash `15e2b0d3...` to
the CAS will look as follows:

```http
PUT /cache/cas/15e2b0d3c33891ebb0f1ef609ec419420c20e320ce94c65fbc8c3312448eb225 HTTP/1.1
Host: localhost:8080
Accept: */*
Content-Length: 9
Connection: Keep-Alive

0x310x320x330x340x350x360x370x380x39
```

## Run Bazel using the remote cache {:#run-remote-cache}

Once a server is set up as the remote cache, to use the remote cache you
need to add flags to your Bazel command. See list of configurations and
their flags below.

You may also need configure authentication, which is specific to your
chosen server.

You may want to add these flags in a `.bazelrc` file so that you don't
need to specify them every time you run Bazel. Depending on your project and
team dynamics, you can add flags to a `.bazelrc` file that is:

* On your local machine
* In your project's workspace, shared with the team
* On the CI system

### Read from and write to the remote cache {:#read-write-remote-cache}

Take care in who has the ability to write to the remote cache. You may want
only your CI system to be able to write to the remote cache.

Use the following flag to read from and write to the remote cache:

```posix-terminal
build --remote_cache=http://{{ '<var>' }}your.host:port{{ '</var>' }}
```

Besides `HTTP`, the following protocols are also supported: `HTTPS`, `grpc`, `grpcs`.

Use the following flag in addition to the one above to only read from the
remote cache:

```posix-terminal
build --remote_upload_local_results=false
```

### Exclude specific targets from using the remote cache {:#targets-remote-cache}

To exclude specific targets from using the remote cache, tag the target with
`no-remote-cache`. For example:

```starlark
java_library(
    name = "target",
    tags = ["no-remote-cache"],
)
```

### Delete content from the remote cache {:#delete-remote-cache}

Deleting content from the remote cache is part of managing your server.
How you delete content from the remote cache depends on the server you have
set up as the cache. When deleting outputs, either delete the entire cache,
or delete old outputs.

The cached outputs are stored as a set of names and hashes. When deleting
content, there's no way to distinguish which output belongs to a specific
build.

You may want to delete content from the cache to:

* Create a clean cache after a cache was poisoned
* Reduce the amount of storage used by deleting old outputs

### Unix sockets {:#unix-sockets}

The remote HTTP cache supports connecting over unix domain sockets. The behavior
is similar to curl's `--unix-socket` flag. Use the following to configure unix
domain socket:

```posix-terminal
   build --remote_cache=http://{{ '<var>' }}your.host:port{{ '</var>' }}
   build --remote_cache_proxy=unix:/{{ '<var>' }}path/to/socket{{ '</var>' }}
```

This feature is unsupported on Windows.

## Disk cache {:#disk-cache}

Bazel can use a directory on the file system as a remote cache. This is
useful for sharing build artifacts when switching branches and/or working
on multiple workspaces of the same project, such as multiple checkouts. Since
Bazel does not garbage-collect the directory, you might want to automate a
periodic cleanup of this directory. Enable the disk cache as follows:

```posix-terminal
build --disk_cache={{ '<var>' }}path/to/build/cache{{ '</var>' }}
```

You can pass a user-specific path to the `--disk_cache` flag using the `~` alias
(Bazel will substitute the current user's home directory). This comes in handy
when enabling the disk cache for all developers of a project via the project's
checked in `.bazelrc` file.

## Known issues {:#known-issues}

**Input file modification during a build**

When an input file is modified during a build, Bazel might upload invalid
results to the remote cache. You can enable a change detection with
the `--experimental_guard_against_concurrent_changes` flag. There
are no known issues and it will be enabled by default in a future release.
See [issue #3360] for updates. Generally, avoid modifying source files during a
build.

**Environment variables leaking into an action**

An action definition contains environment variables. This can be a problem for
sharing remote cache hits across machines. For example, environments with
different `$PATH` variables won't share cache hits. Only environment variables
explicitly whitelisted via `--action_env` are included in an action
definition. Bazel's Debian/Ubuntu package used to install `/etc/bazel.bazelrc`
with a whitelist of environment variables including `$PATH`. If you are getting
fewer cache hits than expected, check that your environment doesn't have an old
`/etc/bazel.bazelrc` file.

**Bazel does not track tools outside a workspace**

Bazel currently does not track tools outside a workspace. This can be a
problem if, for example, an action uses a compiler from `/usr/bin/`. Then,
two users with different compilers installed will wrongly share cache hits
because the outputs are different but they have the same action hash. See
[issue #4558](https://github.com/bazelbuild/bazel/issues/4558){: .external} for updates.

**Incremental in-memory state is lost when running builds inside docker containers**
Bazel uses server/client architecture even when running in single docker container.
On the server side, Bazel maintains an in-memory state which speeds up builds.
When running builds inside docker containers such as in CI, the in-memory state is lost
and Bazel must rebuild it before using the remote cache.

## External links {:#external-links}

* **Your Build in a Datacenter:** The Bazel team gave a [talk](https://fosdem.org/2018/schedule/event/datacenter_build/){: .external} about remote caching and execution at FOSDEM 2018.

* **Faster Bazel builds with remote caching: a benchmark:** Nicolò Valigi wrote a [blog post](https://nicolovaligi.com/faster-bazel-remote-caching-benchmark.html){: .external}
in which he benchmarks remote caching in Bazel.

* [Adapting Rules for Remote Execution](/remote/rules)
* [Troubleshooting Remote Execution](/remote/sandbox)
* [WebDAV module](https://nginx.org/en/docs/http/ngx_http_dav_module.html){: .external}
* [Docker image](https://hub.docker.com/r/buchgr/bazel-remote-cache/){: .external}
* [bazel-remote](https://github.com/buchgr/bazel-remote/){: .external}
* [Google Cloud Storage](https://cloud.google.com/storage){: .external}
* [Google Cloud Console](https://cloud.google.com/console){: .external}
* [Bucket locations](https://cloud.google.com/storage/docs/bucket-locations){: .external}
* [Hazelcast](https://hazelcast.com){: .external}
* [Apache httpd](http://httpd.apache.org){: .external}
* [AWS S3](https://aws.amazon.com/s3){: .external}
* [issue #3360](https://github.com/bazelbuild/bazel/issues/3360){: .external}
* [gRPC](https://grpc.io/){: .external}
* [gRPC protocol](https://github.com/bazelbuild/remote-apis/blob/main/build/bazel/remote/execution/v2/remote_execution.proto){: .external}
* [Buildbarn](https://github.com/buildbarn){: .external}
* [Buildfarm](https://github.com/bazelbuild/bazel-buildfarm){: .external}
* [BuildGrid](https://gitlab.com/BuildGrid/buildgrid){: .external}
* [issue #4558](https://github.com/bazelbuild/bazel/issues/4558){: .external}
* [Application Authentication](https://cloud.google.com/docs/authentication/production){: .external}
* [TurboCache](https://github.com/allada/turbo-cache){: .external}
