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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests with a custom filesystem layer, for faking things like IOExceptions, on top of
 * the real unix filesystem (so we can execute actions).
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class CustomRealFilesystemBuildIntegrationTestCase
    extends BuildIntegrationTestCase {

  private CustomRealFilesystem customFileSystem = null;

  @Override
  protected boolean realFileSystem() {
    return true;
  }

  @Override
  protected FileSystem createFileSystem() {
    if (customFileSystem == null) {
      customFileSystem = new CustomRealFilesystem();
    }
    return customFileSystem;
  }

  /** Tests that IOExceptions encountered while handling inputs are properly handled. */
  @Test
  public void testIOException() throws Exception {
    Path fooBuildFile = write("foo/BUILD", "sh_binary(name = 'foo', srcs = ['foo.sh'])");
    Path fooShFile = fooBuildFile.getParentDirectory().getRelative("foo.sh");
    customFileSystem.alwaysError(fooShFile);

    RecordingOutErr recOutErr = new RecordingOutErr();
    OutErr origOutErr = this.outErr;
    this.outErr = recOutErr;

    try {
      buildTarget("//foo");
      fail();
    } catch (BuildFailedException e) {
      assertThat(recOutErr.errAsLatin1()).contains("missing input file '//foo:foo.sh");
    } finally {
      this.outErr = origOutErr;
    }
  }

  /**
   * Tests that IOExceptions encountered while handling non-mandatory inputs are properly handled.
   */
  @Test
  public void testIOException_NonMandatoryInputs() throws Exception {
    Path fooBuildFile = write("foo/BUILD", "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    write("foo/foo.cc", "#include \"foo/foo.h\"");
    Path fooHFile = fooBuildFile.getParentDirectory().getRelative("foo.h");
    writeAbsolute(fooHFile, "//thisisacomment");
    customFileSystem.alwaysError(fooHFile);

    RecordingOutErr recOutErr = new RecordingOutErr();
    OutErr origOutErr = this.outErr;
    this.outErr = recOutErr;
    try {
      buildTarget("//foo");
      fail();
    } catch (BuildFailedException e) {
      assertThat(recOutErr.errAsLatin1()).contains("Target //foo:foo failed to build");
    } finally {
      System.out.println(recOutErr.errAsLatin1());
      this.outErr = origOutErr;
    }
  }

  /**
   * Tests that IOExceptions encountered while handling non-mandatory generated inputs are properly
   * handled.
   */
  @Test
  public void testIOException_NonMandatoryGeneratedInputs() throws Exception {
    write(
        "bar/BUILD",
        "cc_library(",
        "   name = 'bar',",
        "   hdrs = ['bar.h'],",
        "   srcs = ['bar.cc'],",
        "   visibility = ['//foo:__pkg__']",
        ")",
        "genrule(name = 'bar-gen', srcs = ['in.txt'], outs = ['bar.h'], "
            + "cmd = \"cp $(SRCS) $(OUTS)\")");
    write("foo/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'], deps = [':mid'])",
        "cc_library(name = 'mid', deps = ['//bar'])");
    write("foo/foo.cc",
        "#include \"bar/bar.h\"",
        "int main() { return f(); }");

    write("bar/bar.cc", "int f() { return 0; }");
    write("bar/in.txt", "int f(); // 0");

    // On an incremental skyframe build, the output file from a genrule is statted 7 times:
    //   1 time in FilesystemValueChecker
    //   1 time in ActionCacheChecker#needToExecute
    //   1 time in GenRuleAction#checkOutputsForDirectories
    //   1 time in SkyframeActionExecutor#checkOutputs
    //   1 time in SkyframeActionExecutor#setOutputsReadOnlyAndExecutable

    int numStatsOnIncrementalBuildWithChange = 5;
    buildTarget("//bar");
    Path barHOutputPath = Iterables.getOnlyElement(getArtifacts("//bar:bar.h")).getPath();
    customFileSystem.alwaysErrorAfter(barHOutputPath, numStatsOnIncrementalBuildWithChange);
    write("bar/in.txt", "int f(); // 1");
    buildTarget("//foo");
    // Check that the expected number of stats were made on the generated file (note that this
    // sufficiently confirms that the genrule was rerun).
    // TODO(bazel-team): This test currently isn't super useful because Skyframe doesn't declare
    // deps on undeclared generated headers (it only does so for source headers). If we change this
    // behavior, we can test it by having the final stat done on the output file (on a clean build)
    // fail, but all the previous ones done (e.g. during the genrule execution) succeed. But for now
    // we just check the exact number of stats done.
    assertThat(customFileSystem.getNumCallsUntilError(barHOutputPath)).isEqualTo(0);
    customFileSystem.alwaysErrorAfter(barHOutputPath, numStatsOnIncrementalBuildWithChange);
    write("bar/in.txt", "int f(); // 2");
    buildTarget("//foo");
    assertThat(customFileSystem.getNumCallsUntilError(barHOutputPath)).isEqualTo(0);
  }

  private class CustomRealFilesystem extends UnixFileSystem {
    private Map<Path, Integer> badPaths = new HashMap<>();

    private CustomRealFilesystem() {
      super(DigestHashFunction.getDefaultUnchecked());
    }

    public void alwaysError(Path path) {
      alwaysErrorAfter(path, 0);
    }

    public void alwaysErrorAfter(Path path, int numCalls) {
      badPaths.put(path, numCalls);
    }

    public int getNumCallsUntilError(Path path) {
      return badPaths.containsKey(path) ? badPaths.get(path) : 0;
    }

    private synchronized void maybeThrowExn(Path path) throws IOException {
      if (badPaths.containsKey(path)) {
        Integer numCallsRemaining = badPaths.get(path);
        if (numCallsRemaining <= 0) {
          throw new IOException("nope");
        } else {
          badPaths.put(path, numCallsRemaining - 1);
        }
      }
    }

    @Override
    protected FileStatus statNullable(Path path, boolean followSymlinks) {
      try {
        maybeThrowExn(path);
      } catch (IOException e) {
        return null;
      }
      return super.statNullable(path, followSymlinks);
    }

    @Override
    protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      maybeThrowExn(path);
      return super.statIfFound(path, followSymlinks);
    }

    @Override
    protected UnixFileStatus statInternal(Path path, boolean followSymlinks) throws IOException {
      maybeThrowExn(path);
      return super.statInternal(path, followSymlinks);
    }
  }
}
