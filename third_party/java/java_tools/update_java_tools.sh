#!/bin/bash

# JavaBuilder
echo "Building //src/java_tools/buildjar:JavaBuilder_deploy.jar"
bazel build //src/java_tools/buildjar:JavaBuilder_deploy.jar
cp -f bazel-bin/src/java_tools/buildjar/JavaBuilder_deploy.jar third_party/java/java_tools/JavaBuilder_deploy.jar
echo "Updated third_party/java/java_tools/JavaBuilder_deploy.jar"

# VanillaJavaBuilder
echo "Building //src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar"
bazel build //src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar
cp -f bazel-bin/src/java_tools/buildjar/VanillaJavaBuilder_deploy.jar third_party/java/java_tools/VanillaJavaBuilder_deploy.jar
echo "Updated third_party/java/java_tools/VanillaJavaBuilder_deploy.jar"

