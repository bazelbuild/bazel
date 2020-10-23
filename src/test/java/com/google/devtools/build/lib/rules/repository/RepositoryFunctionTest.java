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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FileValue.RegularFileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
    public RepositoryDirectoryValue.Builder fetch(
        Rule rule,
        Path outputDirectory,
        BlazeDirectories directories,
        SkyFunction.Environment env,
        Map<String, String> markerData,
        SkyKey key)
        throws InterruptedException {
      return null;
    }

    @Override
    protected boolean isLocal(Rule rule) {
      return false;
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
    assertThat(
            TestingRepositoryFunction.getTargetPath(
                TestingRepositoryFunction.getPathAttr(rule), rootDirectory))
        .isEqualTo(rootDirectory.getRelative("a/b/c").asFragment());
  }

  @Test
  public void testGetTargetPathAbsolute() throws Exception {
    Rule rule = scratchRule("external", "w", "local_repository(",
        "    name = 'w',",
        "    path = '/a/b/c',",
        ")");
    assertThat(
            TestingRepositoryFunction.getTargetPath(
                TestingRepositoryFunction.getPathAttr(rule), rootDirectory))
        .isEqualTo(PathFragment.create("/a/b/c"));
  }

  @Test
  public void testGenerateWorkspaceFile() throws Exception {
    Rule rule = scratchRule("external", "abc", "local_repository(",
        "    name = 'abc',",
        "    path = '/a/b/c',",
        ")");
    RepositoryFunction.createWorkspaceFile(rootDirectory, rule.getTargetKind(), rule.getName());
    String workspaceContent = new String(
        FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    assertThat(workspaceContent).contains("workspace(name = \"abc\")");
  }

  private static void assertMarkerFileEscaping(String testCase) {
    String escaped = RepositoryDelegatorFunction.escape(testCase);
    assertThat(RepositoryDelegatorFunction.unescape(escaped)).isEqualTo(testCase);
  }

  @Test
  public void testMarkerFileEscaping() throws Exception {
    assertMarkerFileEscaping(null);
    assertMarkerFileEscaping("\\0");
    assertMarkerFileEscaping("a\\0");
    assertMarkerFileEscaping("a b");
    assertMarkerFileEscaping("a b c");
    assertMarkerFileEscaping("a \\b");
    assertMarkerFileEscaping("a \\nb");
    assertMarkerFileEscaping("a \\\\nb");
    assertMarkerFileEscaping("a \\\nb");
    assertMarkerFileEscaping("a \nb");
  }

  @Test
  public void testFileValueToMarkerValue() throws Exception {
    RootedPath path =
        RootedPath.toRootedPath(Root.fromPath(rootDirectory), scratch.file("foo", "bar"));

    // Digest should be returned if the FileStateValue has it.
    FileStateValue fsv = new RegularFileStateValue(3, new byte[] {1, 2, 3, 4}, null);
    FileValue fv = new RegularFileValue(path, fsv);
    assertThat(RepositoryFunction.fileValueToMarkerValue(fv)).isEqualTo("01020304");

    // Digest should also be returned if the FileStateValue doesn't have it.
    fsv = new RegularFileStateValue(3, null, new FileContentsProxy(100, 200));
    fv = new RegularFileValue(path, fsv);
    String expectedDigest = BaseEncoding.base16().lowerCase().encode(path.asPath().getDigest());
    assertThat(RepositoryFunction.fileValueToMarkerValue(fv)).isEqualTo(expectedDigest);
  }
}
