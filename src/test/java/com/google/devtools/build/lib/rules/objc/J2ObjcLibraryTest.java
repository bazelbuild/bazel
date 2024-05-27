// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.packages.util.MockJ2ObjcSupport;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;

/** Setup for unit tests for j2objc transpilation. */
public class J2ObjcLibraryTest extends ObjcRuleTestCase {

  static final ArtifactExpander DUMMY_ARTIFACT_EXPANDER =
      treeArtifact -> {
        SpecialArtifact parent = (SpecialArtifact) treeArtifact;
        return ImmutableSortedSet.of(
            TreeFileArtifact.createTreeOutput(parent, "children1"),
            TreeFileArtifact.createTreeOutput(parent, "children2"));
      };

  /**
   * Creates and injects a j2objc_library target that depends upon the given label, then returns the
   * ConfiguredTarget for the label with the aspects added.
   */
  protected ConfiguredTarget getJ2ObjCAspectConfiguredTarget(String label) throws Exception {
    // Blaze exposes no interface to summon aspects ex nihilo.
    // To get an aspect, you must create a dependent target that requires the aspect.
    scratch.file(
        "java/com/google/dummy/aspect/BUILD",
        "j2objc_library(",
        "    name = 'transpile',",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        "    deps = ['" + label + "'],",
        ")");

    ConfiguredTarget configuredTarget =
        getConfiguredTarget("//java/com/google/dummy/aspect:transpile");
    return getDirectPrerequisite(configuredTarget, label);
  }

  @Before
  public final void setup() throws Exception  {
    scratch.file("java/com/google/dummy/test/test.java");
    scratch.file(
        "java/com/google/dummy/test/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        java_library(
            name = "test",
            srcs = ["test.java"],
        )

        j2objc_library(
            name = "transpile",
            tags = ["__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__"],
            deps = ["test"],
        )
        """);
    MockJ2ObjcSupport.setup(mockToolsConfig);
    MockProtoSupport.setup(mockToolsConfig);

    useConfiguration(
        "--proto_toolchain_for_java=//tools/proto/toolchains:java",
        "--platforms=" + MockObjcSupport.DARWIN_X86_64,
        "--cpu=darwin_x86_64");

    setBuildLanguageOptions("--incompatible_disable_objc_library_transition");

    mockToolsConfig.append(
        "tools/proto/toolchains/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "package(default_visibility=['//visibility:public'])",
        "proto_lang_toolchain(name = 'java', command_line = 'dont_care:$(OUT)')",
        "proto_lang_toolchain(name='java_stubby1_immutable', command_line = 'dont_care:$(OUT)')",
        "proto_lang_toolchain(name='java_stubby3_immutable', command_line = 'dont_care:$(OUT)')",
        "proto_lang_toolchain(name='java_stubby_compatible13_immutable', "
            + "command_line = 'dont_care')");

    invalidatePackages();
  }
}
