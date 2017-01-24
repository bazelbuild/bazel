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

package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Unit tests for {@link GenRule}.
 */
@RunWith(JUnit4.class)
public class GenruleConfiguredTargetTest extends BuildViewTestCase {
  @Before
  public void createFiles() throws Exception {
    scratch.file("hello/BUILD",
        "genrule(",
        "    name = 'z',",
        "    outs = ['x/y'],",
        "    cmd = 'echo hi > $(@D)/y',",
        ")",
        "genrule(",
        "    name = 'w',",
        "    outs = ['a/b', 'c/d'],",
        "    cmd = 'echo hi | tee $(@D)/a/b $(@D)/c/d',",
        ")");
  }

  @Test
  public void testD() throws Exception {
    ConfiguredTarget z = getConfiguredTarget("//hello:z");
    Artifact y = getOnlyElement(getFilesToBuild(z));
    assertEquals(new PathFragment("hello/x/y"), y.getRootRelativePath());
  }

  @Test
  public void testDMultiOutput() throws Exception {
    ConfiguredTarget z = getConfiguredTarget("//hello:w");
    List<Artifact> files = getFilesToBuild(z).toList();
    assertThat(files).hasSize(2);
    assertEquals(new PathFragment("hello/a/b"), files.get(0).getRootRelativePath());
    assertEquals(new PathFragment("hello/c/d"), files.get(1).getRootRelativePath());
  }
}
