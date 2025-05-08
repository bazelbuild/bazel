// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.view.java;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for java_import. */
@RunWith(JUnit4.class)
public class JavaImportConfiguredTargetTest extends BuildViewTestCase {

  @Before
  public void setCommandLineFlags() throws Exception {
    setBuildLanguageOptions("--experimental_google_legacy_api");
  }

  @Before
  public final void writeBuildFile() throws Exception {
    scratch.overwriteFile(
        "tools/allowlists/java_import_exports/BUILD",
        """
        package_group(
            name = "java_import_exports",
            packages = ["//..."],
        )
        """);
    scratch.overwriteFile(
        "tools/allowlists/java_import_empty_jars/BUILD",
        """
        package_group(
            name = "java_import_empty_jars",
            packages = [],
        )
        """);
  }

  @Test
  public void testRequiresJars() throws Exception {
    checkError(
        "pkg",
        "rule",
        "mandatory attribute 'jars'",
        """
        load("@rules_java//java:defs.bzl", "java_import")
        java_import(name = 'rule')
        """);
  }

  @Test
  public void testDuplicateJars() throws Exception {
    checkError(
        "ji",
        "ji-with-dupe",
        // error:
        "Label '//ji:a.jar' is duplicated in the 'jars' attribute of rule 'ji-with-dupe'",
        // build file
        "load('@rules_java//java:defs.bzl', 'java_import')",
        "filegroup(name='jars', srcs=['a.jar'])",
        "java_import(name = 'ji-with-dupe', jars = ['a.jar', 'a.jar'])");
  }
}
