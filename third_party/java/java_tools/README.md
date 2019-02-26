# How to upgrade the Java tools version in Bazel

Please first update the java tools and then upgrade the Java tools version
in Bazel. See the required steps below.

First make sure the following environment variables are set accordingly:

BAZEL_WORKSPACE    the root of your local bazel repository
VERSION            the new version of the Java tools; Please check the
previous version in
src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE
for the target "remote_java_tools". For example if the url
ends in "java_tools_pkg-0.4.tar.gz" than the current version
is 0.4 and the new version will be 0.5.
For now only upgrade the minor version.
UPDATE_TOOLS_BRANCH      The name of a new git branch.
UPGRADE_VERSION_BRANCH

## Updating the Java tools

Updating the Java tools version in Bazel consists of three steps:
1. Building the Java tools under third_party/java/java_tools/
2. Archiving the tools
3. Uploading the archive to the cloud

### Full instructions for updating the java tools:

```
$ cd $BAZEL_WORKSPACE
$ git checkout -b $UPDATE_TOOLS_BRANCH
$ third_party/java/java_tools/update_java_tools.sh
$ git add . && git commit -m "Update the Java tools under third_party/java/java_tools/" && git push origin $UPDATE_TOOLS_BRANCH
$ bazel build third_party:java_tools_pkg-gz
$ cp bazel-bin/third_party/java_tools_pkg-gz.tar.gz ~/java_tools_pkg-$VERSION.tar.gz
```

In your browser go to https://pantheon.corp.google.com/storage/browser/bazel-mirror/bazel_java_tools/
and push the "Upload files" button in the upper-left side. Upload ~/java_tools_pkg-$VERSION.tar.gz
and $BAZEL_WORKSPACE/third_party/java/java_tools/java_tools-srcs.zip.

## Upgrade the Java tools version

```
$ cd $BAZEL_WORKSPACE
$ git checkout -b $UPGRADE_VERSION_BRANCH
$ sha256sum ~/java_tools_pkg-$VERSION.tar.gz | awk '{print $1}'
```

Next update the urls and sha256 for the target "remote_java_tools" in
src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE
and also the name, urls and sha256 of all the occurrences in WORKSPACE
(targets jdk_WORKSPACE_files and additional_distfiles).
See for example
[upgrading the java tools version from 0.4 to 0.5](https://github.com/bazelbuild/bazel/pull/7541/commits/93eee0e222df9d8aedd6661ea73311645824f188)

# Status of third_party/java/java_tools/

The following tools were built with bazel $bazel_version at commit $git_head
by running:
$ third_party/java/java_tools/update_java_tools.sh $@