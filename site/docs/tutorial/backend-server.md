---
layout: documentation
title: Tutorial - Build the Backend Server
---

# Tutorial - Build the Backend Server

The backend server is a simple web application that runs on Google App Engine
and responds to incoming HTTP requests from the sample Android and iOS apps.

Here, you'll do the following:

*   Review the source files for the app
*   Update the `WORKSPACE` file
*   Create the `appengine.BUILD` file
*   Create a `BUILD` file
*   Run the build
*   Find the build outputs
*   Deploy to a local development server
*   Deploy to Google App Engine

Bazel provides a set of [App Engine build rules](/docs/be/appengine.html)
written using the [Skylark](/docs/skylark/index.html) framework. You'll use
these in the steps below to build the application.

## Review the source files

The source files for the backend server are located in `$WORKSPACE/backend/`.

The key files and directories are:

<table class="table table-condensed table-striped">
<thead>
<tr>
<td>Name</td>
<td>Location</td>
</tr>
</thead>
<tbody>
<tr>
<td>Source file directory</td>
<td><code>src/main/java/com/google/bazel/example/app/</code></td>
</tr>
<tr>
<td>Web application metadata directory</td>
<td><code>webapp/WEB-INF/</code></td>
</tr>
</tbody>
</table>

## Update the WORKSPACE file

As with the Android app, you must add references to
[external dependencies](http://bazel.io/docs/external.html) to your `WORKSPACE`
file. For the backend server, these are references to the App Engine SDK,
the Java Servlet SDK and other libraries needed to build the App Engine
applications.

### Add a new\_http\_archive rule

When you built the Android app, you added a reference to the location on your
filesystem where you downloaded and installed the Android SDK. For the
backend server, however, you'll give Bazel instructions for downloading the
required App Engine SDK package from a remote server. This is optional. You
can also download and install the SDK manually on your filesystem and reference
it from that location as described in the
[App Engine rule documentation](/docs/be/appengine.html).

Add the following to your `WORKSPACE` file:

```python
new_http_archive(
    name = "appengine-java",
    url = "http://central.maven.org/maven2/com/google/appengine/appengine-java-sdk/1.9.23/appengine-java-sdk-1.9.23.zip",
    sha256 = "05e667036e9ef4f999b829fc08f8e5395b33a5a3c30afa9919213088db2b2e89",
    build_file = "appengine.BUILD",
)
```

The [`new_http_archive`](/docs/be/workspace.html#new_http_archive) rule
instructs Bazel to download a remote archive file, uncompress it and add it to
the virtual `external` package by combining the archive contents with
the referenced `BUILD` file, here `appengine.BUILD`. You'll create this file
below after you finish updating your `WORKSPACE` file.

### Add bind rules

You also need to add some [`bind`](/docs/be/workspace.html#bind) rules
to the file. These provide aliases to targets in the virtual `external` package
so they can be located either either inside or outside the workspace without
changing the App Engine build rules.

Add the following to the `WORKSPACE` file:

```python
bind(
    name = "appengine/java/sdk",
    actual = "@appengine-java//:sdk",
)

bind(
    name = "appengine/java/api",
    actual = "@appengine-java//:api",
)

bind(
    name = "appengine/java/jars",
    actual = "@appengine-java//:jars",
)
```

### Add maven_jar rules

Finally, you need to add some
[`maven_jar`](/docs/be/workspace.html#maven_jar) rules to the file. These
tell Bazel to download `.jar` files from the Maven repository and allow Bazel
to use them as Java dependencies.

Add the following to the `WORKSPACE` file:

```python
maven_jar(
    name = "commons-lang",
    artifact = "commons-lang:commons-lang:2.6",
)

maven_jar(
    name = "javax-servlet-api",
    artifact = "javax.servlet:servlet-api:2.5",
)

bind(
    name = "javax/servlet/api",
    actual = "//tools/build_rules/appengine:javax.servlet.api",
)
```

Now, save and close the file. You can compare your `WORKSPACE` file to the
[completed example](https://github.com/bazelbuild/examples//blob/master/tutorial/WORKSPACE)
in the `master` branch of the GitHub repo.

## Create the appengine.BUILD file

When you added the `new_http_archive` to your `WORKSPACE` above, you specified
the filename `appengine.BUILD` in the `build_file` attribute. The
`appengine.BUILD` file is a special build file that allows Bazel to use the
downloaded App Engine SDK libraries as a Bazel package. Here, you'll create
this file in your top-level workspace directory.

Open your new `appengine.BUILD` file for editing:

```bash
$ vi $WORKSPACE/appengine.BUILD
```

Add the following to the file:

```python
java_import(
    name = "jars",
    jars = glob(["**/*.jar"]),
    visibility = ["//visibility:public"],
)

java_import(
    name = "api",
    jars = ["appengine-java-sdk-1.9.23/lib/impl/appengine-api.jar"],
    visibility = ["//visibility:public"],
    neverlink = 1,
)

filegroup(
    name = "sdk",
    srcs = glob(["appengine-java-sdk-1.9.23/**"]),
    visibility = ["//visibility:public"],
    path = "appengine-java-sdk-1.9.23",
)
```

The [`java_import`](/docs/be/java.html#java_import) rules tell
Bazel how to use the precompiled SDK `.jar` files as libraries. These rules
define the targets that you referenced in the `bind` rules in the `WORKSPACE`
file above.

Save and close the file. The
[completed example](https://github.com/bazelbuild/examples//blob/master/tutorial/appengine.BUILD)
is in the `master` branch of the GitHub repo.

## Create a BUILD file

Now that you have set up the external dependencies, you can go ahead and create
the `BUILD` file for the backend server, as you did previously for the sample
Android and iOS apps.

Open your new `BUILD` file for editing:

```bash
$ vi $WORKSPACE/backend/BUILD
```

### Add a java_binary rule

Add the following to your `BUILD` file:

```python
java_binary(
    name = "app",
    srcs = glob(["src/main/java/**/*.java"]),
    main_class = "does.not.exist",
    deps = [
        "//external:javax/servlet/api",
    ],
)
```

The [`java_binary`](/docs/be/java.html#java_binary) tells Bazel
how to build a Java `.jar` library for your application, plus a wrapper shell
script that launches the application code from the specified main class. Here,
we're using this rule instead of the
[`java_library`](/docs/be/java.html#java_library) because we need
the `.jar` file to contain all the dependencies required to build the final
App Engine `.war` file. For this reason, we specify a bogus class name
for the `main_class` attribute.

### Add an appengine_war rule

Add the following to your `BUILD` file:

```python
load("/tools/build_rules/appengine/appengine", "appengine_war")

appengine_war(
    name = "backend",
    data = [":webapp"],
    data_path = "/backend/webapp",
    jars = [":app_deploy.jar"],
)

filegroup(
    name = "webapp",
    srcs = glob(["webapp/**/*"]),
)
```

The [`appengine_war`](/docs/be/appengine.html#appengine_war)
rule builds the final App Engine `war` file from the library `.jar` file and web
application metadata files in the `webapp` directory.

Save and close the file. Again, the
[completed example](https://github.com/google/bazel-examples/blob/master/tutorial/backend/BUILD)
is in the `master` branch of the GitHub repo.

## Run the build

Make sure that your current working directory is inside your Bazel workspace:

```bash
$ cd $WORKSPACE
```

Now, enter the following to build the sample app:

```bash
$ bazel build //backend:backend
```

Bazel now launches and builds the sample app. During the build process, its
output will appear similar to the following:

```bash
INFO: Found 1 target...
Target //backend:backend up-to-date:
  bazel-bin/backend/backend.war
  bazel-bin/backend/backend.deploy
  bazel-bin/backend/backend
INFO: Elapsed time: 56.867s, Critical Path: 2.72s
```

## Find the build outputs

The `.war` file and other outputs are located in the
`$WORKSPACE/bazel-bin/backend` directory.

## Deploy to a local development server

The `appengine_war` rule generates an upload script that you can use to deploy
your backend server on Google App Engine. Here, you'll start a local App Engine
development server in your environment and deploy your application there.

To deploy the application, enter the following:

```bash
$ bazel-bin/backend/backend --port=12345
```

Your application URL will be `http://localhost:12345`

## Deploy to Google App Engine

You can also deploy the application to the live App Engine serving
environment on Google Cloud Platform. For this scenario, you must first create
a project in the
[Google Developers Console](https://console.developers.google.com).

To deploy the application, enter the following:

```bash
$ $WORKSPACE/bazel-bin/backend/backend.deploy <project-id>
```

The deployment script prompts you to authorize access to Google Cloud Platform.
After you have authorized access the first time, you can deploy the application
using the `bazel` command and the following rule target:

```bash
$ bazel run //backend:backend.deploy <project-id>
```

Your application URL will be `http://<project-id>.appspot.com`.

## What's next

Now let's [review](review.md) the tutorial steps.
