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
package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.skylark.util.SkylarkUtil;

/**
 * Tests for using bind() with Skylark rules.
 */
public class BindTest extends BuildViewTestCase {

  @Override
  public void setUp() throws Exception {
    super.setUp();
    SkylarkUtil.setup(scratch);
    scratch.file("test/BUILD",
        "load('/rules/java_rules_skylark', 'java_library')",
        "java_library(name = 'giraffe',",
        "    srcs = ['Giraffe.java'],",
        ")",
        "java_library(name = 'safari',",
        "    srcs = ['Safari.java'],",
        "    deps = ['//external:long-horse'],",
        ")");

    scratch.overwriteFile(
        "tools/jdk/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "filegroup(name = 'java', srcs = ['bin/java'])",
        "filegroup(name = 'jar', srcs = ['bin/jar'])",
        "filegroup(name = 'javac', srcs = ['bin/javac'])",
        "filegroup(name = 'jdk')");

    scratch.overwriteFile("WORKSPACE",
        "bind(",
        "    name = 'long-horse',",
        "    actual = '//test:giraffe',",
        ")");
  }

  public void testFilesToBuild() throws Exception {
    invalidatePackages();
    ConfiguredTarget giraffeTarget = getConfiguredTarget("//test:giraffe");
    Artifact giraffeArtifact =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(giraffeTarget), "giraffe.jar");
    ConfiguredTarget safariTarget = getConfiguredTarget("//test:safari");
    Action safariAction = getGeneratingAction(
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(safariTarget), "safari.jar"));
    assertThat(safariAction.getInputs()).contains(giraffeArtifact);
  }

}
