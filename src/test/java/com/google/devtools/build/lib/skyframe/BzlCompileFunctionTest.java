// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.List;
import net.starlark.java.syntax.LoadStatement;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests of specific functionality of BzlCompileFunction. */
@RunWith(JUnit4.class)
public class BzlCompileFunctionTest extends BuildViewTestCase {

  private class MockFileSystem extends InMemoryFileSystem {
    PathFragment throwIOExceptionFor = null;

    MockFileSystem() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (throwIOExceptionFor != null && path.asFragment().equals(throwIOExceptionFor)) {
        throw new IOException("bork");
      }
      return super.statIfFound(path, followSymlinks);
    }
  }

  private MockFileSystem mockFS;

  @Override
  protected FileSystem createFileSystem() {
    mockFS = new MockFileSystem();
    return mockFS;
  }

  @Test
  public void testIOExceptionOccursDuringReading() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("/workspace/tools/build_rules/BUILD");
    scratch.file(
        "foo/BUILD", //
        "genrule(",
        "    name = 'foo',",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@'",
        ")");
    mockFS.throwIOExceptionFor = PathFragment.create("/workspace/foo/BUILD");
    invalidatePackages(/*alsoConfigs=*/ false); // We don't want to fail early on config creation.

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    Throwable e = errorInfo.getException();
    assertThat(e).isInstanceOf(NoSuchPackageException.class);
    assertThat(e).hasMessageThat().contains("bork");
  }

  @Test
  public void testLoadFromFileInRemoteRepo() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo'",
        ")");
    Path repoPath = scratch.dir("/a_remote_repo");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/foo.bzl", "load(':bar.bzl', 'CONST')");
    scratch.file("/a_remote_repo/remote_pkg/bar.bzl", "CONST = 17");

    invalidatePackages(/*alsoConfigs=*/ false); // Repository shuffling messes with toolchains.
    SkyKey skyKey =
        BzlCompileValue.key(
            Root.fromPath(repoPath),
            Label.parseAbsoluteUnchecked("@a_remote_repo//remote_pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    List<String> loads = getLoads(result.get(skyKey).getAST());
    assertThat(loads).containsExactly(":bar.bzl");
  }

  private static List<String> getLoads(StarlarkFile file) {
    List<String> loads = Lists.newArrayList();
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        loads.add(((LoadStatement) stmt).getImport().getValue());
      }
    }
    return loads;
  }

  @Test
  public void testLoadOfNonexistentFile() throws Exception {
    SkyKey skyKey = BzlCompileValue.key(root, Label.parseAbsoluteUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getError()).contains("cannot load '//pkg:foo.bzl': no such file");
  }
}
