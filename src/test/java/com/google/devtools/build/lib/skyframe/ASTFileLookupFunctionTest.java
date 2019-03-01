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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.syntax.SkylarkImport;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests of specific functionality of ASTFileLookupFunction.
 */
@RunWith(JUnit4.class)
public class ASTFileLookupFunctionTest extends BuildViewTestCase {

  private class MockFileSystem extends InMemoryFileSystem {
    boolean statThrowsIoException;

    @Override
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (statThrowsIoException
          && path.asFragment().getPathString().equals("/workspace/" + preludeLabelRelativePath)) {
        throw new IOException("bork");
      }
      return super.statIfFound(path, followSymlinks);
    }
  }

  private MockFileSystem mockFS;
  String preludeLabelRelativePath =
      getRuleClassProvider().getPreludeLabel().toPathFragment().toString();

  @Override
  protected FileSystem createFileSystem() {
    mockFS = new MockFileSystem();
    return mockFS;
  }

  @Test
  public void testPreludeASTFileIsNotMandatory() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "foo/BUILD", "genrule(name = 'foo',", "  outs = ['out.txt'],", "  cmd = 'echo hello >@')");
    scratch.deleteFile(preludeLabelRelativePath);
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(skyKey).getPackage().containsErrors()).isFalse();
  }

  @Test
  public void testIOExceptionOccursDuringReading() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("/workspace/tools/build_rules/BUILD");
    scratch.file(
        "foo/BUILD", "genrule(name = 'foo',", "  outs = ['out.txt'],", "  cmd = 'echo hello >@')");
    mockFS.statThrowsIoException = true;
    invalidatePackages(/*alsoConfigs=*/false); // We don't want to fail early on config creation.

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    Throwable e = errorInfo.getException();
    assertThat(errorInfo.getRootCauseOfException()).isEqualTo(skyKey);
    assertThat(e).isInstanceOf(NoSuchPackageException.class);
    assertThat(e).hasMessageThat().contains("bork");
  }

  @Test
  public void testLoadFromBuildFileInRemoteRepo() throws Exception {
    scratch.deleteFile(preludeLabelRelativePath);
    scratch.overwriteFile("WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo'",
        ")");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD",
        "load(':ext.bzl', 'CONST')");
    scratch.file("/a_remote_repo/remote_pkg/ext.bzl",
        "CONST = 17");

    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchains.
    SkyKey skyKey =
        ASTFileLookupValue.key(Label.parseAbsoluteUnchecked("@a_remote_repo//remote_pkg:BUILD"));
    EvaluationResult<ASTFileLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    ImmutableList<SkylarkImport> imports = result.get(skyKey).getAST().getImports();
    assertThat(imports).hasSize(1);
    assertThat(imports.get(0).getImportString()).isEqualTo(":ext.bzl");
  }

  @Test
  public void testLoadFromSkylarkFileInRemoteRepo() throws Exception {
    scratch.deleteFile(preludeLabelRelativePath);
    scratch.overwriteFile("WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo'",
        ")");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/ext1.bzl",
        "load(':ext2.bzl', 'CONST')");
    scratch.file("/a_remote_repo/remote_pkg/ext2.bzl",
        "CONST = 17");

    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchains.
    SkyKey skyKey =
        ASTFileLookupValue.key(Label.parseAbsoluteUnchecked("@a_remote_repo//remote_pkg:ext1.bzl"));
    EvaluationResult<ASTFileLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    ImmutableList<SkylarkImport> imports = result.get(skyKey).getAST().getImports();
    assertThat(imports).hasSize(1);
    assertThat(imports.get(0).getImportString()).isEqualTo(":ext2.bzl");
  }

  @Test
  public void testLoadWithNonExistentBuildFile() throws Exception {
    invalidatePackages();
    SkyKey skyKey =
        ASTFileLookupValue.key(Label.parseAbsoluteUnchecked("@a_remote_repo//remote_pkg:BUILD"));
    EvaluationResult<ASTFileLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getErrorMsg())
    .contains("Unable to load package for '@a_remote_repo//remote_pkg:BUILD'");
    assertThat(result.get(skyKey).getErrorMsg())
        .contains("The repository '@a_remote_repo' could not be resolved");
  }
}
