// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.analysis.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.io.MoreFiles;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Function;

public abstract class AbstractMockJavaSupport {

  public static final AbstractMockJavaSupport BAZEL =
      new AbstractMockJavaSupport() {
        @Override
        public void setupRulesJava(
            MockToolsConfig config, Function<String, String> runfilesResolver) throws IOException {
          config.create("rules_java_workspace/WORKSPACE", "workspace(name = 'rules_java')");
          config.create("rules_java_workspace/MODULE.bazel", "module(name = 'rules_java')");
          config.create("rules_java_workspace/java/BUILD");
          config.create("rules_java_workspace/java/common/BUILD");
          config.create("rules_java_workspace/java/private/BUILD");
          config.create("rules_java_workspace/java/toolchains/BUILD");
          config.create("rules_java_workspace/toolchains/BUILD");
          ImmutableList<String> toolsToCopy =
              ImmutableList.of(
                  "java/defs.bzl",
                  "java/java_binary.bzl",
                  "java/java_import.bzl",
                  "java/java_library.bzl",
                  "java/java_plugin.bzl",
                  "java/java_test.bzl",
                  "java/common/java_common.bzl",
                  "java/common/java_info.bzl",
                  "java/common/java_plugin_info.bzl",
                  "java/private/native.bzl",
                  "java/toolchains/java_package_configuration.bzl",
                  "java/toolchains/java_runtime.bzl",
                  "java/toolchains/java_toolchain.bzl",
                  "toolchains/java_toolchain_alias.bzl");
          for (String relativePath : toolsToCopy) {
            Path path = Path.of(runfilesResolver.apply("rules_java/" + relativePath));
            if (Files.exists(path)) {
              config.create(
                  "rules_java_workspace/" + relativePath,
                  MoreFiles.asCharSource(path, UTF_8).read());
            }
          }
          // mocks
          config.create(
              "rules_java_workspace/toolchains/local_java_repository.bzl",
              """
              def local_java_repository(**attrs):
                  pass
              """);
          config.create(
              "rules_java_workspace/toolchains/jdk_build_file.bzl", "JDK_BUILD_TEMPLATE = ''");
          config.create(
              "rules_java_workspace/java/repositories.bzl",
              """
              def rules_java_dependencies():
                  pass

              def rules_java_toolchains():
                  native.register_toolchains("//java/toolchains/runtime:all")
                  native.register_toolchains("//java/toolchains/javac:all")
              """);

          config.create(
              "rules_java_workspace/java/toolchains/runtime/BUILD",
              """
              toolchain_type(name = "toolchain_type")

              toolchain(
                  name = "local_jdk",
                  toolchain = "@bazel_tools//tools/jdk:jdk",
                  toolchain_type = "@rules_java//java/toolchains/runtime:toolchain_type",
              )
              """);
          config.create(
              "rules_java_workspace/java/toolchains/javac/BUILD",
              """
              toolchain_type(name = "toolchain_type")

              toolchain(
                  name = "javac_toolchain",
                  toolchain = "@bazel_tools//tools/jdk:toolchain",
                  toolchain_type = "@rules_java//java/toolchains/javac:toolchain_type",
              )
              """);
        }
      };

  public abstract void setupRulesJava(
      MockToolsConfig mockToolsConfig, Function<String, String> runfilesResolver)
      throws IOException;
}
