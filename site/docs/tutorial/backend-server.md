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
*   Create a `BUILD` file
*   Run the build
*   Find the build outputs
*   Run the application on a local development server
*   Deploy to Google App Engine

Bazel provides a set of [App Engine build rules](https://github.com/bazelbuild/rules_appengine)
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
[external dependencies](http://bazel.build/docs/external.html) to your `WORKSPACE`
file. For the backend server, these are references to the App Engine SDK,
the Java Servlet SDK and other libraries needed to build the App Engine
applications.

### Add the App Engine rule

When you built the Android app, you added a reference to the location on your
filesystem where you downloaded and installed the Android SDK. For the
backend server, however, you'll give Bazel instructions for downloading the
required App Engine SDK package from a remote server. This is optional. You
can also download and install the SDK manually on your filesystem and reference
it from that location as described in the
[App Engine rule documentation](https://github.com/bazelbuild/rules_appengine).

Add the following to your `WORKSPACE` file:

```python
http_archive(
    name = "io_bazel_rules_appengine",
    sha256 = "f4fb98f31248fca5822a9aec37dc362105e57bc28e17c5611a8b99f1d94b37a4",
    strip_prefix = "rules_appengine-0.0.6",
    url = "https://github.com/bazelbuild/rules_appengine/archive/0.0.6.tar.gz",
)
load("@io_bazel_rules_appengine//appengine:appengine.bzl", "appengine_repositories")
appengine_repositories()
```

[`http_archive`](/docs/be/workspace.html#http_archive) downloads the
AppEngine rules from a GitHub archive. We could also have used
[`git_repository`](/docs/be/workspace.html#git_repository) to fetch the rules
directly from the Git repository.

The next two lines use the `appengine_repositories` function defined in
these rules to download the libraries and SDK needed to build AppEngine
applications.

Now, save and close the file. You can compare your `WORKSPACE` file to the
[completed example](https://github.com/bazelbuild/examples//blob/master/tutorial/WORKSPACE)
in the `master` branch of the GitHub repo.

## Create a BUILD file

Now that you have set up the external dependencies, you can go ahead and create
the `BUILD` file for the backend server, as you did previously for the sample
Android and iOS apps.

Open your new `BUILD` file for editing:

```bash
vi $WORKSPACE/backend/BUILD
```

### Add a java_binary rule

Add the following to your `BUILD` file:

```python
java_binary(
    name = "app",
    srcs = glob(["src/main/java/**/*.java"]),
    main_class = "does.not.exist",
    deps = [
        "@io_bazel_rules_appengine//appengine:javax.servlet.api",
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
load("@io_bazel_rules_appengine//appengine:appengine.bzl", "appengine_war")

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
cd $WORKSPACE
```

Now, enter the following to build the server:

```bash
bazel build //backend:backend
```

Bazel now launches and builds the server files. During the build process, its
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

In particular, the `appengine_war` rule generates scripts that you can use to
run your backend locally or deploy it to Google App Engine:

## Run the application on a local development server

Here, you'll start a local App Engine development server in your environment and
run your application on it.

To run the application, enter the following:

```bash
bazel-bin/backend/backend --port=12345
```

Your application will be available at `http://localhost:12345`

## Deploy to Google App Engine

You can also deploy the application to the live App Engine serving
environment on Google Cloud Platform. For this scenario, you must first create
a new Cloud Platform project and App Engine application using the Google Cloud
Platform Console.
Follow [this link](https://console.cloud.google.com/projectselector/appengine/create?lang=java&st=true)
to perform these actions.

Build the target that allows to deploy to App Engine:

```bash
bazel build --java_toolchain=@io_bazel_rules_appengine//appengine:jdk7 //backend:backend.deploy
```

Then, to deploy the application, enter the following:

```bash
bazel-bin/backend/backend.deploy <project-id>
```

The deployment script prompts you to authorize access to Google Cloud Platform.
After you have authorized access the first time, you can deploy the application
using the `bazel` command and the following rule target:

```bash
bazel run //backend:backend.deploy <project-id>
```

Your application URL will be `http://<project-id>.appspot.com`.

## What's next

Now let's [review](review.md) the tutorial steps.
