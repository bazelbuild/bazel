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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyValue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import javax.annotation.Nullable;

/**
 * Tests for @{link RepositoryFunction}
 */
@RunWith(JUnit4.class)
public class RepositoryFunctionTest extends BuildViewTestCase {

  /**
   * Exposes RepositoryFunction's protected methods to this class.
   */
  @VisibleForTesting
  static class TestingRepositoryFunction extends RepositoryFunction {
    @Nullable
    @Override
    public SkyValue fetch(Rule rule, Path outputDirectory, SkyFunction.Environment env)
        throws SkyFunctionException, InterruptedException {
      return null;
    }

    @Override
    protected boolean isLocal(Rule rule) {
      return false;
    }

    public static PathFragment getTargetPath(Rule rule, Path workspace)
        throws RepositoryFunctionException {
      return RepositoryFunction.getTargetPath(rule, workspace);
    }

    @Override
    public Class<? extends RuleDefinition> getRuleDefinition() {
      return null;
    }
  }

  @Test
  public void testGetTargetPathRelative() throws Exception {
    Rule rule = scratchRule("external", "z", "local_repository(",
            "    name = 'z',",
            "    path = 'a/b/c',",
            ")");
    assertEquals(rootDirectory.getRelative("a/b/c").asFragment(),
        TestingRepositoryFunction.getTargetPath(rule, rootDirectory));
  }

  @Test
  public void testGetTargetPathAbsolute() throws Exception {
    Rule rule = scratchRule("external", "w", "local_repository(",
        "    name = 'w',",
        "    path = '/a/b/c',",
        ")");
    assertEquals(new PathFragment("/a/b/c"),
        TestingRepositoryFunction.getTargetPath(rule, rootDirectory));
  }

  @Test
  public void testGenerateWorkspaceFile() throws Exception {
    Rule rule = scratchRule("external", "abc", "local_repository(",
        "    name = 'abc',",
        "    path = '/a/b/c',",
        ")");
    TestingRepositoryFunction repositoryFunction = new TestingRepositoryFunction();
    repositoryFunction.createWorkspaceFile(rootDirectory, rule);
    String workspaceContent = new String(
        FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    assertThat(workspaceContent).contains("workspace(name = \"abc\")");
  }
}
