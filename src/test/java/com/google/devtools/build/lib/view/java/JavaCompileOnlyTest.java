// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.view.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.CompileOnlyTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests that validate --compile_only behavior. */
@RunWith(JUnit4.class)
public class JavaCompileOnlyTest extends CompileOnlyTestCase {

  @Before
  public void setupStarlarkJavaBinary() throws Exception {
    setBuildLanguageOptions("--experimental_google_legacy_api");
  }

  @Test
  public void testJavaCompileOnly() throws Exception {
    scratch.file(
        "java/main/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_binary")
        java_binary(
            name = "main",
            srcs = ["Main.java"],
            deploy_manifest_lines = [
                "k1: v1",
                "k2: v2",
            ],
            main_class = "main.Main",
            deps = ["//java/hello_library"],
        )
        """);
    scratch.file(
        "java/hello_library/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "hello_library",
            srcs = ["HelloLibrary.java"],
        )
        """);
    scratch.file(
        "java/main/Main.java",
        """
        package main;

        import hello_library.HelloLibrary;

        public class Main {
          public static void main(String[] args) {
            HelloLibrary.funcHelloLibrary();
            System.out.println("Hello, world!");
          }
        }
        """);
    scratch.file(
        "java/hello_library/HelloLibrary.java",
        """
        package hello_library;

        public class HelloLibrary {
          public static void funcHelloLibrary() {
            System.out.println("Hello, library!");
          }
        }
        """);

    ConfiguredTarget target = getConfiguredTarget("//java/main:main");

    // Check that only main.jar would have been compiled and not libhello_library.jar.
    assertThat(getArtifactByExecPathSuffix(target, "/main.jar")).isNotNull();
    assertThat(getArtifactByExecPathSuffix(target, "/libhello_library.jar")).isNull();
  }
}
