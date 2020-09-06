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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.NotifyingHelper;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests with a custom filesystem layer, for faking things like IOExceptions, on top of
 * the real unix filesystem (so we can execute actions).
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class CustomRealFilesystemBuildIntegrationTest extends BuildIntegrationTestCase {

  private CustomRealFilesystem customFileSystem = null;

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

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    events.assertContainsError("missing input file '//foo:foo.sh': nope");
  }

  /** Tests that IOExceptions encountered while handling inputs are properly handled. */
  @Test
  public void testIOExceptionMidLevel() throws Exception {
    Path fooBuildFile =
        write(
            "foo/BUILD",
            "sh_binary(name = 'foo', srcs = ['foo.sh'])",
            "genrule(name = 'top', srcs = [':foo'], outs = ['out'], cmd = 'touch $@')");
    Path fooShFile = fooBuildFile.getParentDirectory().getRelative("foo.sh");
    customFileSystem.alwaysError(fooShFile);

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:top"));
    events.assertContainsError("//foo:top: missing input file '//foo:foo.sh': nope");
  }

  /**
   * Tests that IOExceptions encountered while handling non-mandatory inputs are properly handled.
   */
  @Test
  public void testIOException_nonMandatoryInputs() throws Exception {
    Path fooBuildFile =
        write("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], hdrs_check = 'loose')");
    write("foo/foo.cc", "#include \"foo/foo.h\"");
    Path fooHFile = fooBuildFile.getParentDirectory().getRelative("foo.h");
    writeAbsolute(fooHFile, "//thisisacomment");
    customFileSystem.alwaysError(fooHFile);

    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    ImmutableList<Cause> rootCauses = e.getRootCauses().toList();
    assertThat(rootCauses).hasSize(1);
    assertThat(rootCauses.get(0).getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:foo"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.BUILD_FAILURE);
    events.assertContainsError("foo/BUILD:1:11: //foo:foo: missing input file 'foo/foo.h': nope");
  }

  /** Tests that IOExceptions encountered when not all discovered deps are done are handled. */
  @Test
  public void testIOException_missingNonMandatoryInput() throws Exception {
    Path fooBuildFile =
        write("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], hdrs_check = 'loose')");
    write("foo/foo.cc", "#include \"foo/error.h\"", "#include \"foo/other.h\"");
    Path errorHFile = fooBuildFile.getParentDirectory().getRelative("error.h");
    Path otherHFile = fooBuildFile.getParentDirectory().getRelative("other.h");
    writeAbsolute(errorHFile, "//thisisacomment");
    writeAbsolute(otherHFile, "//thisisacomment");
    customFileSystem.alwaysError(errorHFile);
    getSkyframeExecutor()
        .getEvaluatorForTesting()
        .injectGraphTransformerForTesting(
            NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  if (key.functionName().equals(FileStateValue.FILE_STATE)
                      && ((RootedPath) key.argument())
                          .getRootRelativePath()
                          .getPathString()
                          .endsWith("foo/other.h")) {
                    try {
                      Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                      throw new IllegalStateException("Should have been interrupted by failure");
                    } catch (InterruptedException e) {
                      Thread.currentThread().interrupt();
                    }
                  }
                }));

    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    ImmutableList<Cause> rootCauses = e.getRootCauses().toList();
    assertThat(rootCauses).hasSize(1);
    assertThat(rootCauses.get(0).getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:foo"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.BUILD_FAILURE);
    events.assertContainsError("foo/BUILD:1:11: //foo:foo: missing input file 'foo/error.h': nope");
  }

  /**
   * Tests that IOExceptions encountered while handling non-mandatory generated inputs are properly
   * handled.
   */
  @Test
  public void testIOException_nonMandatoryGeneratedInputs() throws Exception {
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

  @Test
  public void treeArtifactIOExceptionTopLevel() throws Exception {
    write(
        "foo/tree.bzl",
        "def _tree_impl(ctx):",
        "    tree = ctx.actions.declare_directory('mytree.cc')",
        "    ctx.actions.run_shell(outputs = [tree], command = 'mkdir -p %s && touch %s/one.cc' %"
            + " (tree.path, tree.path))",
        "    return [DefaultInfo(files = depset([tree]))]",
        "",
        "mytree = rule(implementation = _tree_impl)");
    write(
        "foo/BUILD",
        "load('//foo:tree.bzl', 'mytree')",
        "mytree(name = 'tree')",
        "cc_library(name = 'lib', srcs = [':tree'])");
    customFileSystem.errorOnDirectory("mytree");
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//foo:lib"));
    ImmutableList<Cause> rootCauses = e.getRootCauses().toList();
    assertThat(rootCauses).hasSize(1);
    assertThat(rootCauses.get(0).getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:lib"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.BUILD_FAILURE);
    events.assertContainsError(
        "foo/BUILD:3:11: Failed to create output directory for TreeArtifact"
            + " blaze-out/k8-fastbuild/bin/foo/_pic_objs/lib/mytree: nope");
  }

  @Test
  public void treeArtifactIOExceptionMidLevel() throws Exception {
    write(
        "foo/tree.bzl",
        "def _tree_impl(ctx):",
        "    tree = ctx.actions.declare_directory('mytree.cc')",
        "    ctx.actions.run_shell(outputs = [tree], command = 'mkdir -p %s && touch %s/one.cc' %"
            + " (tree.path, tree.path))",
        "    return [DefaultInfo(files = depset([tree]))]",
        "",
        "mytree = rule(implementation = _tree_impl)");
    write(
        "foo/BUILD",
        "load('//foo:tree.bzl', 'mytree')",
        "mytree(name = 'tree')",
        "cc_library(name = 'lib', srcs = [':tree'])",
        "genrule(name = 'top', srcs = [':lib'], outs = ['out'], cmd = 'touch $@')");
    customFileSystem.errorOnDirectory("mytree");
    // Make sure we take default codepath in ActionExecutionFunction.
    addOptions("--experimental_nested_set_as_skykey_threshold=1");
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//foo:top"));
    ImmutableList<Cause> rootCauses = e.getRootCauses().toList();
    assertThat(rootCauses).hasSize(1);
    assertThat(rootCauses.get(0).getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:lib"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.BUILD_FAILURE);
    events.assertContainsError(
        "foo/BUILD:3:11: Failed to create output directory for TreeArtifact"
            + " blaze-out/k8-fastbuild/bin/foo/_pic_objs/lib/mytree: nope");
  }

  private static class CustomRealFilesystem extends UnixFileSystem {
    private Map<Path, Integer> badPaths = new HashMap<>();
    private final Set<String> createDirectoryErrorNames = new HashSet<>();

    private CustomRealFilesystem() {
      super(DigestHashFunction.getDefaultUnchecked());
    }

    void alwaysError(Path path) {
      alwaysErrorAfter(path, 0);
    }

    void alwaysErrorAfter(Path path, int numCalls) {
      badPaths.put(path, numCalls);
    }

    void errorOnDirectory(String baseName) {
      createDirectoryErrorNames.add(baseName);
    }

    int getNumCallsUntilError(Path path) {
      return badPaths.getOrDefault(path, 0);
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

    @Override
    public void createDirectoryAndParents(Path path) throws IOException {
      if (createDirectoryErrorNames.contains(path.getBaseName())) {
        throw new IOException("nope");
      }
      super.createDirectoryAndParents(path);
    }
  }
}
