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
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static com.google.devtools.build.lib.testutil.BlazeTestUtils.createFilesetRule;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.NotifyingHelper;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Integration tests with a custom filesystem layer, for faking things like IOExceptions, on top of
 * the real unix filesystem (so we can execute actions).
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(TestParameterInjector.class)
public class CustomRealFilesystemBuildIntegrationTest extends GoogleBuildIntegrationTestCase {

  private CustomRealFilesystem customFileSystem = null;

  @Override
  protected FileSystem createFileSystem() {
    if (customFileSystem == null) {
      customFileSystem = new CustomRealFilesystem(getDigestHashFunction());
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
    events.assertContainsError("//foo:foo: error reading file '//foo:foo.sh': nope");
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
    events.assertContainsError(
        "Executing genrule //foo:top failed: error reading file '//foo:foo.sh': nope");
  }

  @Test
  public void globDanglingSymlink() throws Exception {
    Path packageDirPath = write("foo/BUILD", "exports_files(glob(['*.txt']))").getParentDirectory();
    write("foo/existing.txt");
    Path badSymlink = packageDirPath.getChild("bad.txt");
    FileSystemUtils.ensureSymbolicLink(badSymlink, "nope");
    customFileSystem.alwaysError(badSymlink);
    TargetParsingException e =
        assertThrows(TargetParsingException.class, () -> buildTarget("//foo:all"));
    assertThat(e).hasMessageThat().contains("no such package 'foo': error globbing [*.txt]: nope");
  }

  /**
   * Tests that IOExceptions encountered while handling non-mandatory inputs are properly handled.
   */
  @Test
  public void testIOException_nonMandatoryInputs() throws Exception {
    addOptions("--features=cc_include_scanning");
    write("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], hdrs_check = 'loose')");
    write("foo/foo.cc", "#include \"foo/foo.h\"");
    Path fooHFile = write("foo/foo.h", "//thisisacomment");
    customFileSystem.alwaysErrorAfter(fooHFile, 1);

    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    events.assertContainsError(
        "foo/BUILD:1:11: Compiling foo/foo.cc failed: include scanning: Include scanning"
            + " IOException: nope");
  }

  @Test
  public void ioExceptionInSkyframeOptionalInput(@TestParameter boolean keepGoing)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    addOptions("--features=cc_include_scanning");
    write("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], hdrs_check = 'loose')");
    Path ccFile = write("foo/foo.cc", "#include \"foo/foo.h\"");
    // Making the destination a symlink keeps the include scanner from populating the syscalls cache
    // before Skyframe gets a chance to stat the bad file.
    ccFile.getParentDirectory().getChild("foo.h").createSymbolicLink(PathFragment.create("bad.h"));
    Path badFile = write("foo/bad.h", "//ok contents");
    customFileSystem.alwaysError(badFile);

    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    events.assertContainsError(
        "foo/BUILD:1:11: Compiling foo/foo.cc failed: error reading file 'foo/foo.h': nope");
  }

  @Test
  public void incrementalNonMandatoryInputIOException(
      @TestParameter boolean keepGoing, @TestParameter({"0", "1"}) int nestedSetOnSkyframe)
      throws Exception {
    addOptions("--features=cc_include_scanning");
    addOptions("--keep_going=" + keepGoing);
    addOptions("--experimental_nested_set_as_skykey_threshold=" + nestedSetOnSkyframe);
    write("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], hdrs_check = 'loose')");
    write("foo/foo.cc", "#include \"foo/foo.h\"");
    Path fooHFile = write("foo/foo.h", "//thisisacomment");
    buildTarget("//foo");
    write("foo/foo.cc", "//no include anymore");
    customFileSystem.alwaysError(fooHFile);
    if (keepGoing) {
      buildTarget("//foo");
    } else {
      // TODO(b/166268889): fix this: this really crashes, not just a bug report!
      assertThrows(RuntimeException.class, () -> buildTarget("//foo"));
    }
  }

  @Test
  public void unusedInputIOExceptionIncremental(@TestParameter boolean keepGoing) throws Exception {
    addOptions("--keep_going=" + keepGoing);
    write(
        "foo/pruning.bzl",
        "def _impl(ctx):",
        "  inputs = ctx.attr.inputs.files",
        "  output = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  unused_file = ctx.actions.declare_file(ctx.label.name + '.unused')",
        "  ctx.actions.run(",
        "    inputs = inputs,",
        "    outputs = [output, unused_file],",
        "    arguments = [output.path, unused_file.path] + [f.path for f in inputs.to_list()],",
        "    executable = ctx.executable.executable,",
        "    unused_inputs_list = unused_file,",
        "  )",
        "  return DefaultInfo(files = depset([output]))",
        "",
        "build_rule = rule(",
        "  attrs = {",
        "    'inputs': attr.label(allow_files = True),",
        "    'executable': attr.label(executable = True, allow_files = True, cfg = 'host'),",
        "  },",
        "  implementation = _impl,",
        ")");
    Path unusedSh =
        write("foo/all_unused.sh", "touch $1", "shift", "unused=$1", "shift", "echo $@ > $unused");
    unusedSh.setExecutable(true);
    write(
        "foo/BUILD",
        "load('//foo:pruning.bzl', 'build_rule')",
        "build_rule(name = 'prune', inputs = ':unused.txt', executable = ':all_unused.sh')");
    Path unusedPath = write("foo/unused.txt");
    buildTarget("//foo:prune");
    customFileSystem.alwaysError(unusedPath);
    if (keepGoing) {
      buildTarget("//foo:prune");
    } else {
      // TODO(b/166268889): fix.
      RuntimeException e = assertThrows(RuntimeException.class, () -> buildTarget("//foo:prune"));
      assertThat(e).hasCauseThat().isInstanceOf(DetailedException.class);
      assertThat(e)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("error reading file '//foo:unused.txt': nope");
      assertThat(((DetailedException) e.getCause()).getDetailedExitCode().getFailureDetail())
          .comparingExpectedFieldsOnly()
          .isEqualTo(
              FailureDetails.FailureDetail.newBuilder()
                  .setExecution(
                      FailureDetails.Execution.newBuilder()
                          .setCode(FailureDetails.Execution.Code.SOURCE_INPUT_IO_EXCEPTION))
                  .build());
    }
  }

  /** Tests that IOExceptions encountered when not all discovered deps are done are handled. */
  @Test
  public void testIOException_missingNonMandatoryInput() throws Exception {
    addOptions("--features=cc_include_scanning");
    Path fooBuildFile =
        write("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], hdrs_check = 'loose')");
    write("foo/foo.cc", "#include \"foo/error.h\"", "#include \"foo/other.h\"");
    Path errorHFile = fooBuildFile.getParentDirectory().getRelative("error.h");
    Path otherHFile = fooBuildFile.getParentDirectory().getRelative("other.h");
    writeAbsolute(errorHFile, "//thisisacomment");
    writeAbsolute(otherHFile, "//thisisacomment");
    customFileSystem.alwaysErrorAfter(errorHFile, 1);
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
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    events.assertContainsError(
        "foo/BUILD:1:11: Compiling foo/foo.cc failed: include scanning: Include scanning"
            + " IOException: nope");
  }

  /**
   * Tests that IOExceptions encountered while handling non-mandatory generated inputs are properly
   * handled.
   */
  @Test
  public void testIOException_nonMandatoryGeneratedInputs() throws Exception {
    addOptions("--features=cc_include_scanning");
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

    // On an incremental skyframe build, the output file from a genrule is accessed 6 times via this
    // test's instrumented methods.
    //   FilesystemValueChecker
    //   ActionCacheChecker#needToExecute
    //   Path#getDigest (from ActionCacheChecker)
    //   Internal remote execution client to check if symlink
    //   SpawnIncludeScanner#shouldparseRemotely
    //   IncludeParser#extractInclusions

    int numAccessesOnIncrementalBuildWithChange = 6;
    buildTarget("//bar");
    Path barHOutputPath = Iterables.getOnlyElement(getArtifacts("//bar:bar.h")).getPath();
    customFileSystem.alwaysErrorAfter(barHOutputPath, numAccessesOnIncrementalBuildWithChange);
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
    customFileSystem.alwaysErrorAfter(barHOutputPath, numAccessesOnIncrementalBuildWithChange);
    write("bar/in.txt", "int f(); // 2");
    buildTarget("//foo");
    assertThat(customFileSystem.getNumCallsUntilError(barHOutputPath)).isEqualTo(0);
  }

  @Test
  public void ioExceptionReadingBuildFileForDiscoveredInput() throws Exception {
    addOptions("--features=cc_include_scanning");
    write("hello/BUILD", "cc_library(name = 'hello', srcs = ['hello.cc'], hdrs_check = 'loose')");
    write("hello/hello.cc", "#include \"hello/subdir/undeclared.h\"");
    Path header = write("hello/subdir/undeclared.h");
    Path buildFile = header.getParentDirectory().getChild("BUILD");
    // Error unfortunately not noticed when we find header directly.
    customFileSystem.alwaysError(buildFile);
    addOptions("--discard_analysis_cache"); // Fall back to action cache on next build.
    buildTarget("//hello:hello");
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//hello:hello"));
    MoreAsserts.assertContainsEventRegex(
        events.collector(),
        "^ERROR.*Compiling hello/hello.cc failed: Unable to resolve hello/subdir/undeclared.h as an"
            + " artifact: no such package 'hello/subdir': IO errors while looking for BUILD file"
            + " reading .*hello/subdir/BUILD: nope");
    assertThat(e.getDetailedExitCode().getFailureDetail())
        .comparingExpectedFieldsOnly()
        .isEqualTo(
            FailureDetails.FailureDetail.newBuilder()
                .setIncludeScanning(
                    FailureDetails.IncludeScanning.newBuilder()
                        .setCode(FailureDetails.IncludeScanning.Code.SYSTEM_PACKAGE_LOAD_FAILURE)
                        .setPackageLoadingCode(
                            FailureDetails.PackageLoading.Code.OTHER_IO_EXCEPTION))
                .build());
  }

  @Test
  public void inconsistentExceptionReadingBuildFileForDiscoveredInput() throws Exception {
    addOptions("--features=cc_include_scanning");
    write("hello/BUILD", "cc_library(name = 'hello', srcs = ['hello.cc'], hdrs_check = 'loose')");
    write("hello/hello.cc", "#include \"hello/subdir/undeclared.h\"");
    write("hello/subdir/undeclared.h");
    addOptions("--discard_analysis_cache"); // Fall back to action cache on next build.
    buildTarget("//hello:hello");
    Path buildFile = write("hello/subdir/BUILD");
    customFileSystem.errorInsideStat(buildFile, 0);
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//hello:hello"));
    MoreAsserts.assertContainsEventRegex(
        events.collector(),
        ".*Compiling hello/hello.cc failed: Unable to resolve hello/subdir/undeclared.h as an"
            + " artifact: Inconsistent filesystem operations. 'stat' said .*/hello/subdir/BUILD is"
            + " a file but then we later encountered error 'nope for .*/hello/subdir/BUILD' which"
            + " indicates that .*/hello/subdir/BUILD is no longer a file.*");
    events.assertContainsError("hello/subdir/BUILD ");
    assertThat(e.getDetailedExitCode().getFailureDetail())
        .comparingExpectedFieldsOnly()
        .isEqualTo(
            FailureDetails.FailureDetail.newBuilder()
                .setIncludeScanning(
                    FailureDetails.IncludeScanning.newBuilder()
                        .setCode(FailureDetails.IncludeScanning.Code.SYSTEM_PACKAGE_LOAD_FAILURE)
                        .setPackageLoadingCode(
                            FailureDetails.PackageLoading.Code
                                .PERSISTENT_INCONSISTENT_FILESYSTEM_ERROR))
                .build());
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
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.BUILD_FAILURE);
    events.assertContainsError(
        "foo/BUILD:3:11: Failed to create output directory for TreeArtifact"
            + " blaze-out/k8-fastbuild/bin/foo/_pic_objs/lib/mytree: nope");
  }

  @Test
  public void filesetIOException() throws Exception {
    write("foo/BUILD");
    Path filePath = write("foo/subdir/file");
    // Violating best practices, this doesn't explicitly list the file underneath //foo that it
    // wants, since then foo would have to expose that file as a target, leading to foo/subdir
    // being listed (and cached in Skyframe) during the analysis phase, not the execution phase.
    write(
        "fileset/BUILD",
        createFilesetRule("fileset", "fs_out", "FilesetEntry (srcdir = '//foo', destdir = 'x')"));
    customFileSystem.alwaysError(filePath.getParentDirectory());
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//fileset"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    events.assertContainsError(
        "Traversing Fileset trees to write manifest fileset/fileset.fileset_manifest failed: Error"
            + " while traversing directory foo/subdir: nope");
  }

  @Test
  public void filesetIOExceptionInBuildFile() throws Exception {
    // Violating best practices, this doesn't explicitly list the file underneath //foo that it
    // wants, since then foo would have to expose that file as a target, leading to foo/subdir
    // being listed (and cached in Skyframe) during the analysis phase, not the execution phase.
    Path packageBuildFile =
        write(
            "fileset/BUILD",
            createFilesetRule("fileset", "fs_out", "FilesetEntry (srcdir = 'foo', destdir = 'x')"));
    Path packageDirectory = packageBuildFile.getParentDirectory();
    Path subdir = packageDirectory.getRelative("foo/bar");
    subdir.createDirectoryAndParents();
    customFileSystem.alwaysError(subdir.getChild("BUILD"));
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//fileset"));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    events.assertContainsError(
        "Traversing Fileset trees to write manifest fileset/fileset.fileset_manifest failed: Error"
            + " while traversing directory fileset/foo/bar: no such package 'fileset/foo/bar': IO"
            + " errors while looking for BUILD file");
  }

  private void runIoExceptionInTopLevelSource() throws Exception {
    write(
        "foo/rule.bzl",
        "def _impl(ctx):",
        "  return [DefaultInfo(files = depset([], transitive = [dep[DefaultInfo].files for dep in"
            + " ctx.attr.srcs]))]",
        "",
        "top_source = rule(",
        "    implementation = _impl,",
        "    attrs = {'srcs': attr.label_list(allow_files = True)}",
        ")");
    Path buildFile =
        write(
            "foo/BUILD",
            "load(':rule.bzl', 'top_source')",
            "top_source(name = 'foo', srcs = ['error.in', 'missing.in'])");
    customFileSystem.alwaysError(buildFile.getParentDirectory().getChild("error.in"));
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
  }

  @Test
  public void ioExceptionInTopLevelSource_keepGoing() throws Exception {
    addOptions("--keep_going");
    runIoExceptionInTopLevelSource();
    events.assertContainsError(
        "foo/BUILD:2:11: //foo:foo: error reading file '//foo:error.in': nope");
    events.assertContainsError("foo/BUILD:2:11: //foo:foo: missing input file '//foo:missing.in'");
    events.assertContainsError("2 input file(s) are in error or do not exist");
  }

  @Test
  public void ioExceptionInTopLevelSource_noKeepGoing() throws Exception {
    runIoExceptionInTopLevelSource();
    MoreAsserts.assertContainsEventRegex(
        events.collector(),
        ".*foo/BUILD:2:11: //foo:foo: (error reading file '//foo:error.in': nope|missing input file"
            + " '//foo:missing.in')");
    MoreAsserts.assertContainsEventRegex(
        events.collector(),
        ".*(1 input file\\(s\\) (are in error|do not exist)|2 input file\\(s\\) are in error or do"
            + " not exist)");
  }

  private void runMissingFileAndIoException() throws Exception {
    Path buildFile =
        write(
            "foo/BUILD",
            "genrule(name = 'foo', srcs = ['error.in', 'missing.in'], outs = ['out'], cmd = 'touch"
                + " $@')");
    customFileSystem.alwaysError(buildFile.getParentDirectory().getChild("error.in"));
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
  }

  @Test
  public void missingFileAndIoException_keepGoing() throws Exception {
    addOptions("--keep_going");
    runMissingFileAndIoException();
    events.assertContainsError(
        "foo/BUILD:1:8: Executing genrule //foo:foo failed: error reading file '//foo:error.in':"
            + " nope");
    events.assertContainsError(
        "foo/BUILD:1:8: Executing genrule //foo:foo failed: missing input file '//foo:missing.in'");
    events.assertContainsError(
        "Executing genrule //foo:foo failed: 2 input file(s) are in error or do not exist");
  }

  @Test
  public void missingFileAndIoException_noKeepGoing() throws Exception {
    runMissingFileAndIoException();
    MoreAsserts.assertContainsEventRegex(
        events.collector(),
        ".*foo/BUILD:1:8: Executing genrule //foo:foo failed: (error reading file '//foo:error.in':"
            + " nope|missing input file '//foo:missing.in')");
    MoreAsserts.assertContainsEventRegex(
        events.collector(),
        ".*(1 input file\\(s\\) (are in error|do not exist)|2 input file\\(s\\) are in error or do"
            + " not exist)");
  }

  private static class CustomRealFilesystem extends UnixFileSystem {
    private final Map<PathFragment, Integer> badPaths = new HashMap<>();
    private final Map<PathFragment, Integer> statBadPaths = new HashMap<>();
    private final Set<String> createDirectoryErrorNames = new HashSet<>();

    private CustomRealFilesystem(DigestHashFunction digestHashFunction) {
      super(digestHashFunction, /*hashAttributeName=*/ "");
    }

    void alwaysError(Path path) {
      alwaysErrorAfter(path, 0);
    }

    void alwaysErrorAfter(Path path, int numCalls) {
      badPaths.put(path.asFragment(), numCalls);
    }

    void errorOnDirectory(String baseName) {
      createDirectoryErrorNames.add(baseName);
    }

    void errorInsideStat(Path path, int numCalls) {
      statBadPaths.put(path.asFragment(), numCalls);
    }

    int getNumCallsUntilError(Path path) {
      return badPaths.getOrDefault(path.asFragment(), 0);
    }

    private static boolean shouldThrowExn(PathFragment path, Map<PathFragment, Integer> paths) {
      if (paths.containsKey(path)) {
        Integer numCallsRemaining = paths.get(path);
        if (numCallsRemaining <= 0) {
          return true;
        } else {
          paths.put(path, numCallsRemaining - 1);
        }
      }
      return false;
    }

    private synchronized void maybeThrowExn(PathFragment path) throws IOException {
      if (shouldThrowExn(path, badPaths)) {
        throw new IOException("nope");
      }
    }

    @Override
    protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
      try {
        maybeThrowExn(path);
      } catch (IOException e) {
        return null;
      }
      return super.statNullable(path, followSymlinks);
    }

    @Override
    protected FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
      maybeThrowExn(path);
      FileStatus fileStatus = super.statIfFound(path, followSymlinks);
      return shouldThrowExn(path, statBadPaths) ? new ThrowingFileStatus(path) : fileStatus;
    }

    @Override
    protected UnixFileStatus statInternal(PathFragment path, boolean followSymlinks)
        throws IOException {
      maybeThrowExn(path);
      return super.statInternal(path, followSymlinks);
    }

    @Override
    protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
        throws IOException {
      maybeThrowExn(path);
      return super.readdir(path, followSymlinks);
    }

    @Override
    public void createDirectoryAndParents(PathFragment path) throws IOException {
      if (createDirectoryErrorNames.contains(path.getBaseName())) {
        throw new IOException("nope");
      }
      super.createDirectoryAndParents(path);
    }

    @Override
    protected PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
      maybeThrowExn(path);
      return super.readSymbolicLinkUnchecked(path);
    }

    @Override
    protected InputStream createFileInputStream(PathFragment path) throws IOException {
      maybeThrowExn(path);
      return super.createFileInputStream(path);
    }

    private static class ThrowingFileStatus implements FileStatus {
      private final PathFragment path;

      ThrowingFileStatus(PathFragment path) {
        this.path = path;
      }

      @Override
      public boolean isFile() {
        return true;
      }

      @Override
      public boolean isDirectory() {
        return false;
      }

      @Override
      public boolean isSymbolicLink() {
        return false;
      }

      @Override
      public boolean isSpecialFile() {
        return false;
      }

      @Override
      public long getSize() throws IOException {
        throw new IOException("nope for " + path);
      }

      @Override
      public long getLastModifiedTime() throws IOException {
        throw new IOException("nope for " + path);
      }

      @Override
      public long getLastChangeTime() throws IOException {
        throw new IOException("nope for " + path);
      }

      @Override
      public long getNodeId() throws IOException {
        throw new IOException("nope for " + path);
      }
    }
  }
}
