# [MongoDB Bazel Fork](https://bazel.build)

This is the MongoDB internal version of Bazel used for building mongodb/mongo. This is for any modifications that are needed that either wouldn't make sense or would take too long to upstream to Bazel's main repository.

## Building

To build the MongoDB fork of Bazel through the CI pipeline, first clone the branch `release-7.5.0-mongo` then run:

```
evergreen patch -p mongodb-bazel --path mongo/evergreen.yml -y --browse
```

Then select the platforms you would like to build on.

## Deploying

In the repository that you intend to use the Custom Bazel Binaries from this fork, modify `.bazeliskrc` and `.bazelversion` to point at the evergreen patch run that created them. If merging the commits to a mainline branch, a waterfall patch should always be used instead of a user-generated patch.
