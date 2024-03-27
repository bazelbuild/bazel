// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.License;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for {@link License}. */
@RunWith(JUnit4.class)
public class BazelLicensingTests extends BuildViewTestCase {
  @Test
  public void testJavaPluginAllowsOutputLicenseDeclaration() throws Exception {
    scratch.file(
        "ise/BUILD",
        """
        licenses(["restricted"])

        java_library(
            name = "dependency",
            srcs = ["dependency.java"],
        )

        java_plugin(
            name = "plugin",
            srcs = ["plugin.java"],
            output_licenses = ["unencumbered"],
            deps = [":dependency"],
        )
        """);

    scratch.file(
        "gsa/BUILD",
        """
        licenses(["unencumbered"])

        java_library(
            name = "library",
            srcs = ["library.java"],
            plugins = ["//ise:plugin"],
        )
        """);

    assertThat(getConfiguredTarget("//gsa:library")).isNotNull();
    assertNoEvents();
  }
}
