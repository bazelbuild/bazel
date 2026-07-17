Project: /_project.yaml
Book: /_book.yaml

# Configuring Bazel CI to Test Rules for Remote Execution

{% include "_buttons.html" %}

This page is for owners and maintainers of Bazel rule repositories. It
describes how to configure the Bazel Continuous Integration (CI) system for
your repository to test your rules for compatibility against a remote execution
scenario. The instructions on this page apply to projects stored in
GitHub repositories.

## Prerequisites {:#prerequisites}

Before completing the steps on this page, ensure the following:

*   Your GitHub repository is part of the
    [Bazel GitHub organization](https://github.com/bazelbuild){: .external}.
*   You have configured Buildkite for your repository as described in
    [Bazel Continuous Integration](https://github.com/bazelbuild/continuous-integration/tree/master/buildkite){: .external}.

## Setting up the Bazel CI for testing {:#bazel-ci-testing}

1.  In your `.bazelci/presubmit.yml` file, do the following:

    a.  Add a config named `rbe_ubuntu1604`.

    b.  In the `rbe_ubuntu1604` config, add the build and test targets you want to test against remote execution.

2.  Add the[`bazel-toolchains`](https://github.com/bazelbuild/bazel-toolchains){: .external}
    GitHub repository to your `WORKSPACE` file, pinned to the
    [latest release](https://releases.bazel.build/bazel-toolchains.html). Also
    add an `rbe_autoconfig` target with name `buildkite_config`. This example
    creates toolchain configuration for remote execution with BuildKite CI
    for `rbe_ubuntu1604`.

```posix-terminal
load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")

rbe_autoconfig(name = "buildkite_config")
```

3.  Send a pull request with your changes to the `presubmit.yml` file. (See
    [example pull request](https://github.com/bazelbuild/rules_rust/commit/db141526d89d00748404856524cedd7db8939c35){: .external}.)

4.  To view build results, click **Details** for the RBE (Ubuntu
    16.04) pull request check in GitHub, as shown in the figure below. This link
    becomes available after the pull request has been merged and the CI tests
    have run. (See
    [example results](https://source.cloud.google.com/results/invocations/375e325c-0a05-47af-87bd-fed1363e0333){: .external}.)

    ![Example results](/docs/images/rbe-ci-1.png "Example results")

5.  (Optional) Set the **bazel test (RBE (Ubuntu 16.04))** check as a test
    required to pass before merging in your branch protection rule. The setting
    is located in GitHub in **Settings > Branches > Branch protection rules**,
    as shown in the following figure.

    ![Branch protection rules settings](/docs/images/rbe-ci-2.png "Branch protection rules")

## Troubleshooting failed builds and tests {:#troubleshooting-failed-builds}

If your build or tests fail, it's likely due to the following:

*   **Required build or test tools are not installed in the default container.**
    Builds using the `rbe_ubuntu1604` config run by default inside an
    [`rbe-ubuntu16-04`](https://console.cloud.google.com/marketplace/details/google/rbe-ubuntu16-04){: .external}
    container, which includes tools common to many Bazel builds. However, if
    your rules require tools not present in the default container, you must
    create a custom container based on the
    [`rbe-ubuntu16-04`](https://console.cloud.google.com/marketplace/details/google/rbe-ubuntu16-04){: .external}
    container and include those tools as described later.

*   **Build or test targets are using rules that are incompatible with remote
    execution.** See
    [Adapting Bazel Rules for Remote Execution](/remote/rules) for
    details about compatibility with remote execution.

## Using a custom container in the rbe_ubuntu1604 CI config {:#custom-container}

The `rbe-ubuntu16-04` container is publicly available at the following URL:

```
http://gcr.io/cloud-marketplace/google/rbe-ubuntu16-04
```

You can pull it directly from Container Registry or build it from source. The
next sections describe both options.

Before you begin, make sure you have installed `gcloud`, `docker`, and `git`.
If you are building the container from source, you must also install the latest
version of Bazel.

### Pulling the rbe-ubuntu16-04 from Container Registry {:#container-registry}

To pull the `rbe-ubuntu16-04` container from Container Registry, run the
following command:

```posix-terminal
gcloud docker -- pull gcr.io/cloud-marketplace/google/rbe-ubuntu16-04@sha256:{{ '<var>' }}sha256-checksum{{ '</var>' }}
```

Replace {{ '<var>' }}sha256-checksum{{ '</var>' }} with the SHA256 checksum value for
[the latest container](https://console.cloud.google.com/gcr/images/cloud-marketplace/GLOBAL/google/rbe-ubuntu16-04){: .external}.

### Building the rbe-ubuntu16-04 container from source {:#container-source}

To build the `rbe-ubuntu16-04` container from source, do the following:

1.  Clone the `bazel-toolchains` repository:

    ```posix-terminal
    git clone https://github.com/bazelbuild/bazel-toolchains
    ```

2.  Set up toolchain container targets and build the container as explained in
    [Toolchain Containers](https://github.com/bazelbuild/bazel-toolchains/tree/master/container){: .external}.

3.  Pull the freshly built container:

    ```posix-terminal
gcloud docker -- pull gcr.io/{{ '<var>' }}project-id{{ '</var>' }}/{{ '<var>' }}custom-container-name{{ '</var>' }}{{ '<var>' }}sha256-checksum{{ '</var>' }}
    ```

### Running the custom container {:#run-custom-container}

To run the custom container, do one of the following:

*   If you pulled the container from Container Registry, run the following
    command:

    ```posix-terminal
    docker run -it gcr.io/cloud-marketplace/google/rbe-ubuntu16-04@sha256:{{ '<var>' }}sha256-checksum{{ '</var>'}}/bin/bash
    ```

    Replace `sha256-checksum` with the SHA256 checksum value for the
    [latest container](https://console.cloud.google.com/gcr/images/cloud-marketplace/GLOBAL/google/rbe-ubuntu16-04){: .external}.

*   If you built the container from source, run the following command:

    ```posix-terminal
    docker run -it gcr.io/{{ '<var>' }}project-id{{ '</var>' }}/{{ '<var>' }}custom-container-name{{ '</var>' }}@sha256:{{ '<var>' }}sha256sum{{ '</var>' }} /bin/bash
    ```

### Adding resources to the custom container {:#add-resources-container}

Use a [`Dockerfile`](https://docs.docker.com/engine/reference/builder/){: .external} or
[`rules_docker`](https://github.com/bazelbuild/rules_docker){: .external} to add resources or
alternate versions of the original resources to the `rbe-ubuntu16-04` container.
If you are new to Docker, read the following:

*   [Docker for beginners](https://github.com/docker/labs/tree/master/beginner){: .external}
*   [Docker Samples](https://docs.docker.com/samples/){: .external}

For example, the following `Dockerfile` snippet installs `{{ '<var>' }}my_tool_package{{ '</var>' }}`:

```
FROM gcr.io/cloud-marketplace/google/rbe-ubuntu16-04@sha256:{{ '<var>' }}sha256-checksum{{ '</var>' }}
RUN apt-get update && yes | apt-get install -y {{ '<var>' }}my_tool_package{{ '</var>' }}
```

### Pushing the custom container to Container Registry {:#push-container-registry}

Once you have customized the container, build the container image and push it to
Container Registry as follows:

1. Build the container image:

    ```posix-terminal
    docker build -t {{ '<var>' }}custom-container-name{{ '</var>' }}.

    docker tag {{ '<var>' }}custom-container-name{{ '</var>' }} gcr.io/{{ '<var>' }}project-id{{ '</var>' }}/{{ '<var>' }}custom-container-name{{ '</var>' }}
    ```

2.  Push the container image to Container Registry:

    ```posix-terminal
    gcloud docker -- push gcr.io/{{ '<var>' }}project-id{{ '</var>' }}/{{ '<var>' }}custom-container-name{{ '</var>' }}
    ```

3.  Navigate to the following URL to verify the container has been pushed:

    https://console.cloud.google.com/gcr/images/{{ '<var>' }}project-id{{ '</var>' }}/GLOBAL/{{ '<var>' }}custom-container-name{{ '</var>' }}

4.  Take note of the SHA256 checksum of your custom container. You will need to
    provide it in your build platform definition later.

5.  Configure the container for public access as described in  publicly
    accessible as explained in
    [Serving images publicly](https://cloud.google.com/container-registry/docs/access-control#serving_images_publicly){: .external}.

    For more information, see
    [Pushing and Pulling Images](https://cloud.google.com/container-registry/docs/pushing-and-pulling){: .external}.


### Specifying the build platform definition {:#platform-definition}

You must include a [Bazel platform](/extending/platforms) configuration in your
custom toolchain configuration, which allows Bazel to select a toolchain
appropriate to the desired hardware/software platform. To generate
automatically a valid platform, you can add  to your `WORKSPACE` an
`rbe_autoconfig` target with name `buildkite_config` which includes additional
attrs to select your custom container. For details on this setup, read
the up-to-date documentation for [`rbe_autoconfig`](https://github.com/bazelbuild/bazel-toolchains/blob/master/rules/rbe_repo.bzl){: .external}.
