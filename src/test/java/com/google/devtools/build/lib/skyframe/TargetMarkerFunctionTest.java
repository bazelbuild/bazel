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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link TargetMarkerFunction}. Unfortunately, we can't directly test
 * TargetMarkerFunction as it uses PackageValues, and PackageFunction uses legacy stuff that isn't
 * easily mockable. So our testing strategy is to make hacky calls to SkyframeExecutor.
 */
@RunWith(JUnit4.class)
public class TargetMarkerFunctionTest extends BuildViewTestCase {

  private SkyframeExecutor skyframeExecutor;
  private CustomInMemoryFs fs = new CustomInMemoryFs();

  @Before
  public final void setSkyframExecutor() throws Exception  {
    skyframeExecutor = getSkyframeExecutor();
  }

  @Override
  protected FileSystem createFileSystem() {
    return fs;
  }

  private SkyKey skyKey(String labelName) throws Exception {
    return TargetMarkerValue.key(Label.parseAbsolute(labelName, ImmutableMap.of()));
  }

  private Throwable getErrorFromTargetValue(String labelName) throws Exception {
    reporter.removeHandler(failFastHandler);
    SkyKey targetKey = TargetMarkerValue.key(Label.parseAbsolute(labelName, ImmutableMap.of()));
    EvaluationResult<TargetMarkerValue> evaluationResult =
        SkyframeExecutorTestUtils.evaluate(
            skyframeExecutor, targetKey, /*keepGoing=*/ false, reporter);
    Preconditions.checkState(evaluationResult.hasError());
    reporter.addHandler(failFastHandler);
    ErrorInfo errorInfo = evaluationResult.getError(skyKey(labelName));
    // Ensures that TargetFunction rethrows all transitive exceptions.
    assertThat(errorInfo.getRootCauses()).containsExactly(targetKey);
    return errorInfo.getException();
  }

  /** Regression test for b/12545745 */
  @Test
  public void testLabelCrossingSubpackageBoundary() throws Exception {
    scratch.file("a/b/c/foo.sh", "echo 'FOO'");
    scratch.file("a/BUILD", "sh_library(name = 'foo', srcs = ['b/c/foo.sh'])");
    String labelName = "//a:b/c/foo.sh";

    scratch.file("a/b/BUILD");
    ModifiedFileSet subpackageBuildFile =
        ModifiedFileSet.builder().modify(PathFragment.create("a/b/BUILD")).build();
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, subpackageBuildFile, Root.fromPath(rootDirectory));

    NoSuchTargetException exn = (NoSuchTargetException) getErrorFromTargetValue(labelName);
    // In the presence of b/12545745, the error message is different and comes from the
    // PackageFunction.
    assertThat(exn)
        .hasMessageThat()
        .contains("Label '//a:b/c/foo.sh' crosses boundary of subpackage 'a/b'");
  }

  @Test
  public void testNoBuildFileForTargetWithSlash() throws Exception {
    String labelName = "//no/such/package:target/withslash";
    BuildFileNotFoundException exn =
        (BuildFileNotFoundException) getErrorFromTargetValue(labelName);
    assertThat(exn.getPackageId()).isEqualTo(PackageIdentifier.createInMainRepo("no/such/package"));
    String expectedMessage =
        "no such package 'no/such/package': BUILD file not found on "
            + "package path for 'no/such/package'";
    assertThat(exn).hasMessageThat().isEqualTo(expectedMessage);
  }

  @Test
  public void testRuleWithError() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "genrule(name = 'conflict1', cmd = '', srcs = [], outs = ['conflict'])",
        "genrule(name = 'conflict2', cmd = '', srcs = [], outs = ['conflict'])");
    NoSuchTargetException exn = (NoSuchTargetException) getErrorFromTargetValue("@//a:conflict1");
    assertThat(exn)
        .hasMessageThat()
        .contains("Target '//a:conflict1' contains an error and its package is in error");
    assertThat(exn.getLabel().toString()).isEqualTo("//a:conflict1");
    assertThat(exn.hasTarget()).isTrue();
  }

  @Test
  public void testTargetFunctionRethrowsExceptions() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("a/BUILD", "sh_library(name = 'b/c')");
    Path subpackageBuildFile = scratch.file("a/b/BUILD", "sh_library(name = 'c')");
    fs.stubStatIOException(subpackageBuildFile, new IOException("nope"));
    BuildFileNotFoundException exn =
        (BuildFileNotFoundException) getErrorFromTargetValue("//a:b/c");
    assertThat(exn).hasMessageThat().contains("nope");
  }

  private static class CustomInMemoryFs extends InMemoryFileSystem {

    private Map<Path, IOException> stubbedStatExceptions = Maps.newHashMap();

    public CustomInMemoryFs() {
      super(BlazeClock.instance());
    }

    public void stubStatIOException(Path path, IOException stubbedResult) {
      stubbedStatExceptions.put(path, stubbedResult);
    }

    @Override
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (stubbedStatExceptions.containsKey(path)) {
        throw stubbedStatExceptions.get(path);
      }
      return super.statIfFound(path, followSymlinks);
    }
  }
}
