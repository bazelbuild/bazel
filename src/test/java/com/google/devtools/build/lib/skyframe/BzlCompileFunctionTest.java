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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests of specific functionality of BzlCompileFunction. */
@RunWith(JUnit4.class)
public class BzlCompileFunctionTest extends BuildViewTestCase {

  private static class MockFileSystem extends InMemoryFileSystem {
    PathFragment throwIOExceptionFor = null;

    MockFileSystem() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
      if (path.equals(throwIOExceptionFor)) {
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
    scratch.file("/workspace/tools/test_build_rules/BUILD");
    scratch.file(
        "foo/BUILD",
        """
        genrule(
            name = "foo",
            outs = ["out.txt"],
            cmd = "echo hello >@",
        )
        """);
    mockFS.throwIOExceptionFor = PathFragment.create("/workspace/foo/BUILD");
    invalidatePackages(/*alsoConfigs=*/ false); // We don't want to fail early on config creation.

    SkyKey skyKey = PackageIdentifier.createInMainRepo("foo");
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
    Path repoPath = scratch.dir("/a_remote_repo");
    scratch.file("/a_remote_repo/REPO.bazel");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/foo.bzl", "load(':bar.bzl', 'CONST')");
    scratch.file("/a_remote_repo/remote_pkg/bar.bzl", "CONST = 17");

    invalidatePackages(/*alsoConfigs=*/ false); // Repository shuffling messes with toolchains.
    SkyKey skyKey =
        BzlCompileValue.key(
            Root.fromPath(repoPath),
            Label.parseCanonicalUnchecked("@a_remote_repo//remote_pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    List<String> loads =
        BzlLoadFunction.getLoadsFromProgram(result.get(skyKey).getProgram()).stream()
            .map(Pair::getFirst)
            .collect(toImmutableList());
    assertThat(loads).containsExactly(":bar.bzl");
  }

  @Test
  public void testLoadOfNonexistentFile() throws Exception {
    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getError()).contains("cannot load '//pkg:foo.bzl': no such file");
  }

  @Test
  public void testBigIntegerLiterals() throws Exception {
    // This test ensures that numerical literals with values that can't be expressed as Java longs
    // can be compiled. Regression test for b/217548647.
    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:bigint.bzl"));
    scratch.file("pkg/BUILD");
    scratch.file(
        "pkg/bigint.bzl",
        String.format(
            "[%s, %s]",
            BigInteger.valueOf(Long.MIN_VALUE).subtract(BigInteger.ONE),
            BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE)));

    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    BzlCompileValue bzlCompileValue = result.get(skyKey);
    assertThat(bzlCompileValue.lookupSuccessful()).isTrue();

    try (Mutability mu = Mutability.create()) {
      Object val =
          Starlark.execFileProgram(
              bzlCompileValue.getProgram(),
              Module.withPredeclared(StarlarkSemantics.DEFAULT, ImmutableMap.of()),
              StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT));
      assertThat(val.toString()).isEqualTo("[-9223372036854775809, 9223372036854775808]");
    }
  }

  @Test
  public void testInvalidUtf8_enforcementOff() throws Exception {
    setBuildLanguageOptions("--noincompatible_enforce_starlark_utf8");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", new byte[] {'#', ' ', (byte) 0x80});

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isTrue();
    assertNoEvents();
  }

  @Test
  public void testInvalidUtf8_enforcementWarning() throws Exception {
    setBuildLanguageOptions("--incompatible_enforce_starlark_utf8=warning");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", new byte[] {'#', ' ', (byte) 0x80});

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isTrue();
    assertContainsEvent(
        "WARNING /workspace/pkg/foo.bzl: not a valid UTF-8 encoded file; this can lead to"
            + " inconsistent behavior and will be disallowed in a future version of Bazel");
  }

  @Test
  public void testInvalidUtf8_enforcementError() throws Exception {
    reporter.removeHandler(failFastHandler);
    setBuildLanguageOptions("--incompatible_enforce_starlark_utf8");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", new byte[] {'#', ' ', (byte) 0x80});

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getError())
        .isEqualTo("compilation of '/workspace/pkg/foo.bzl' failed");
    assertContainsEvent(
        "ERROR /workspace/pkg/foo.bzl: not a valid UTF-8 encoded file; this can lead to"
            + " inconsistent behavior and will be disallowed in a future version of Bazel");
  }

  @Test
  public void testBzlFileSize_withinLimits() throws Exception {
    setBuildLanguageOptions("--max_bzl_file_size=1000", "--soft_max_bzl_file_size=500");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isTrue();
    assertNoEvents();
  }

  @Test
  public void testBzlFileSize_exceedsSoftLimitWithoutOptOut_fails() throws Exception {
    reporter.removeHandler(failFastHandler);
    setBuildLanguageOptions("--max_bzl_file_size=1000", "--soft_max_bzl_file_size=4");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getError())
        .contains("File '//pkg:foo.bzl' size (6 bytes) exceeds the soft maximum size (4 bytes)");
    assertContainsEvent(
        "ERROR /workspace/pkg/foo.bzl: File '//pkg:foo.bzl' size (6 bytes) exceeds the soft"
            + " maximum size (4 bytes). Split this file or move data out of Starlark. To opt out"
            + " up to the hard limit, set `_KNOWN_OVERSIZED_BZL_FILE =");
  }

  @Test
  public void testBzlFileSize_exceedsSoftLimitWithValidOptOut_succeeds() throws Exception {
    setBuildLanguageOptions("--max_bzl_file_size=1000", "--soft_max_bzl_file_size=4");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "_KNOWN_OVERSIZED_BZL_FILE = \"b/564114066\"", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isTrue();
    assertNoEvents();
  }

  @Test
  public void testBzlFileSize_exceedsSoftLimitWithNonStringOptOut_fails() throws Exception {
    reporter.removeHandler(failFastHandler);
    setBuildLanguageOptions("--max_bzl_file_size=1000", "--soft_max_bzl_file_size=4");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "_KNOWN_OVERSIZED_BZL_FILE = True", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertContainsEvent("exceeds the soft maximum size (4 bytes)");
  }

  @Test
  public void testBzlFileSize_exceedsMaxLimitEvenWithSoftOptOut_fails() throws Exception {
    reporter.removeHandler(failFastHandler);
    setBuildLanguageOptions("--max_bzl_file_size=10", "--soft_max_bzl_file_size=4");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "_KNOWN_OVERSIZED_BZL_FILE = \"b/564114066\"", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getError())
        .contains(
            "File '//pkg:foo.bzl' size (48 bytes) exceeds the maximum allowed size (10 bytes)");
    assertContainsEvent(
        "ERROR /workspace/pkg/foo.bzl: File '//pkg:foo.bzl' size (48 bytes) exceeds the maximum"
            + " allowed size (10 bytes).");
  }

  @Test
  public void testSclFileSize_exceedsSoftAndHardLimits() throws Exception {
    reporter.removeHandler(failFastHandler);
    setBuildLanguageOptions("--max_bzl_file_size=1000", "--soft_max_bzl_file_size=4");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.scl", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.scl"));
    EvaluationResult<BzlCompileValue> softResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(softResult.get(skyKey).lookupSuccessful()).isFalse();
    assertContainsEvent(
        "ERROR /workspace/pkg/foo.scl: File '//pkg:foo.scl' size (6 bytes) exceeds the soft"
            + " maximum size (4 bytes).");

    scratch.overwriteFile("pkg/foo.scl", "_KNOWN_OVERSIZED_BZL_FILE = \"b/564114066\"", "x = 1");
    invalidatePackages(/* alsoConfigs= */ false);
    EvaluationResult<BzlCompileValue> optOutResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(optOutResult.get(skyKey).lookupSuccessful()).isTrue();
  }

  @Test
  public void testBzlFileSize_zeroLimitDisabled() throws Exception {
    setBuildLanguageOptions("--max_bzl_file_size=0", "--soft_max_bzl_file_size=0");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isTrue();
    assertNoEvents();
  }

  @Test
  public void testBzlFileSize_allowlistedExempt() throws Exception {
    scratch.file("tools/allowlist/BUILD");
    scratch.file(
        "tools/allowlist/allowlist.scl",
        "ALLOWED = [",
        "    \"other/file.bzl\",",
        "    \"pkg/foo.bzl\",",
        "]");
    setBuildLanguageOptions(
        "--max_bzl_file_size=1000",
        "--soft_max_bzl_file_size=500",
        "--bzl_file_size_limit_allowlist=//tools/allowlist:allowlist.scl");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = '" + "a".repeat(2000) + "'");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isTrue();
    assertNoEvents();
  }

  @Test
  public void testBzlFileSize_allowlistFileModified_invalidatesSkyframeCache() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("tools/allowlist/BUILD");
    scratch.file("tools/allowlist/allowlist.scl", "ALLOWED = [\"other/file.bzl\"]");
    setBuildLanguageOptions(
        "--max_bzl_file_size=1000",
        "--soft_max_bzl_file_size=500",
        "--bzl_file_size_limit_allowlist=//tools/allowlist:allowlist.scl");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = '" + "a".repeat(2000) + "'");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> initialResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(initialResult.get(skyKey).lookupSuccessful()).isFalse();

    // Add pkg/foo.bzl to the allowlist file; Skyframe must invalidate the failed node and succeed.
    scratch.overwriteFile(
        "tools/allowlist/allowlist.scl", "ALLOWED = [\"other/file.bzl\", \"pkg/foo.bzl\"]");
    invalidatePackages(/* alsoConfigs= */ false);
    eventCollector.clear();
    EvaluationResult<BzlCompileValue> allowlistedResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(allowlistedResult.get(skyKey).lookupSuccessful()).isTrue();
    assertNoEvents();

    // Remove pkg/foo.bzl from the allowlist file; Skyframe must invalidate the cached success.
    scratch.overwriteFile("tools/allowlist/allowlist.scl", "ALLOWED = []");
    invalidatePackages(/* alsoConfigs= */ false);
    EvaluationResult<BzlCompileValue> removedResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(removedResult.get(skyKey).lookupSuccessful()).isFalse();
  }

  @Test
  public void testBzlFileSize_allowlistFileOversized_failsWithoutDependencyCycle()
      throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("tools/allowlist/BUILD");
    scratch.file(
        "tools/allowlist/allowlist.scl",
        "# Long comment that makes allowlist.scl itself exceed 15 bytes in file size",
        "ALLOWED = [\"pkg/foo.bzl\"]");
    setBuildLanguageOptions(
        "--max_bzl_file_size=15",
        "--soft_max_bzl_file_size=5",
        "--bzl_file_size_limit_allowlist=//tools/allowlist:allowlist.scl");

    SkyKey allowlistKey =
        BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//tools/allowlist:allowlist.scl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), allowlistKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(allowlistKey).lookupSuccessful()).isFalse();
    assertThat(result.get(allowlistKey).getError())
        .contains("The allowlist file itself cannot be oversized to avoid dependency cycles.");
  }

  @Test
  public void testBzlFileSize_allowlistFileMissingAllowedSymbol_failsCompilation()
      throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("tools/allowlist/BUILD");
    scratch.file("tools/allowlist/allowlist.scl", "NOT_ALLOWED = []");
    setBuildLanguageOptions(
        "--max_bzl_file_size=1000",
        "--soft_max_bzl_file_size=500",
        "--bzl_file_size_limit_allowlist=//tools/allowlist:allowlist.scl");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = '" + "a".repeat(2000) + "'");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(result.get(skyKey).lookupSuccessful()).isFalse();
    assertThat(result.get(skyKey).getError()).contains("does not define an 'ALLOWED' symbol");
  }

  @Test
  public void testBzlFileSize_allowlistChange_onlyInvalidatesOversizedBzlFiles() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("tools/allowlist/BUILD");
    scratch.file("tools/allowlist/allowlist.scl", "ALLOWED = [\"pkg/oversized.bzl\"]");
    setBuildLanguageOptions(
        "--max_bzl_file_size=1000",
        "--soft_max_bzl_file_size=500",
        "--bzl_file_size_limit_allowlist=//tools/allowlist:allowlist.scl");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/normal.bzl", "x = 1"); // normal: size < 500
    scratch.file("pkg/oversized.bzl", "x = '" + "a".repeat(2000) + "'"); // oversized: size > 1000

    SkyKey normalKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:normal.bzl"));
    SkyKey oversizedKey =
        BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:oversized.bzl"));

    // Both compile successfully initially.
    EvaluationResult<BzlCompileValue> initialNormalResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), normalKey, /* keepGoing= */ false, reporter);
    EvaluationResult<BzlCompileValue> initialOversizedResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), oversizedKey, /* keepGoing= */ false, reporter);
    assertThat(initialNormalResult.get(normalKey).lookupSuccessful()).isTrue();
    assertThat(initialOversizedResult.get(oversizedKey).lookupSuccessful()).isTrue();

    // Verify Skyframe dependency graph:
    Label allowlistLabel = Label.parseCanonicalUnchecked("//tools/allowlist:allowlist.scl");
    SkyKey allowlistFileKey =
        FileValue.key(RootedPath.toRootedPath(root, allowlistLabel.toPathFragment()));
    SkyKey allowlistCompileKey = BzlCompileValue.key(root, allowlistLabel);
    Iterable<SkyKey> normalDeps =
        getSkyframeExecutor()
            .getEvaluator()
            .getExistingEntryAtCurrentlyEvaluatingVersion(normalKey)
            .getDirectDeps();
    Iterable<SkyKey> oversizedDeps =
        getSkyframeExecutor()
            .getEvaluator()
            .getExistingEntryAtCurrentlyEvaluatingVersion(oversizedKey)
            .getDirectDeps();

    assertThat(normalDeps).doesNotContain(allowlistFileKey);
    assertThat(normalDeps).doesNotContain(allowlistCompileKey);
    assertThat(oversizedDeps).contains(allowlistFileKey);
    assertThat(oversizedDeps).contains(allowlistCompileKey);

    // Modify allowlist.scl: remove pkg/oversized.bzl
    scratch.overwriteFile("tools/allowlist/allowlist.scl", "ALLOWED = []");
    invalidatePackages(/* alsoConfigs= */ false);

    // Evaluate both again.
    EvaluationResult<BzlCompileValue> secondNormalResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), normalKey, /* keepGoing= */ false, reporter);
    EvaluationResult<BzlCompileValue> secondOversizedResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), oversizedKey, /* keepGoing= */ false, reporter);
    // Oversized bzl was invalidated by the allowlist change and now fails compilation:
    assertThat(secondOversizedResult.get(oversizedKey).lookupSuccessful()).isFalse();
    // Normal bzl has no dependency on allowlist, so it remains valid in Skyframe:
    assertThat(secondNormalResult.get(normalKey).lookupSuccessful()).isTrue();
  }

  @Test
  public void testBzlFileSize_overrideInvalidatesSkyframeCache() throws Exception {
    reporter.removeHandler(failFastHandler);
    setBuildLanguageOptions("--max_bzl_file_size=4", "--soft_max_bzl_file_size=2");

    scratch.file("pkg/BUILD");
    scratch.file("pkg/foo.bzl", "x = 1");

    SkyKey skyKey = BzlCompileValue.key(root, Label.parseCanonicalUnchecked("//pkg:foo.bzl"));
    EvaluationResult<BzlCompileValue> initialResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(initialResult.get(skyKey).lookupSuccessful()).isFalse();

    // Now supply larger limits: Skyframe must invalidate cached failure.
    setBuildLanguageOptions("--max_bzl_file_size=1000", "--soft_max_bzl_file_size=1000");
    EvaluationResult<BzlCompileValue> retryResult =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /* keepGoing= */ false, reporter);
    assertThat(retryResult.get(skyKey).lookupSuccessful()).isTrue();
  }
}
