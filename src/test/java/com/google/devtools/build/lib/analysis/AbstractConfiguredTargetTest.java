// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link AbstractConfiguredTarget}
 */
@RunWith(JUnit4.class)
public class AbstractConfiguredTargetTest extends BuildViewTestCase {

  @Test
  public void testRunfilesProviderIsNotImportant() throws Exception {
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "java/a",
            "a",
            "java_binary(name='a', srcs=['A.java'], deps=[':b'])",
            "java_library(name='b', srcs=['B.java'])");

    ImmutableSet<Artifact> artifacts =
        TopLevelArtifactHelper.getAllArtifactsToBuild(
                x,
                new TopLevelArtifactContext(
                    /*runTestsExclusively=*/ false,
                    /*expandFilesets=*/ false,
                    /*fullyResolveFilesetSymlinks=*/ false,
                    /*outputGroups=*/ ImmutableSortedSet.<String>of(
                        OutputGroupInfo.DEFAULT, OutputGroupInfo.HIDDEN_TOP_LEVEL)))
            .getImportantArtifacts()
            .toSet();

    assertThat(ActionsTestUtil.baseArtifactNames(artifacts)).doesNotContain("libb.jar");
  }

  @Test
  public void testRunUnderWithExperimental() throws Exception {
    scratch.file("foo/BUILD",
        "sh_test(name = 'test', srcs = ['test.sh'], data = ['test.txt'])");
    scratch.file("experimental/bar/BUILD",
        "sh_binary(name = 'bar', srcs = ['test.sh'])");
    useConfiguration("--run_under=//experimental/bar");
    getConfiguredTarget("//foo:test");
  }
}
