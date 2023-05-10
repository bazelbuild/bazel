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

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link WorkspaceFileFunction}. */
@RunWith(JUnit4.class)
public class WorkspaceFileFunctionTest extends BuildViewTestCase {
  private static Label getLabelMapping(Package pkg, String name) throws NoSuchTargetException {
    return (Label) ((Rule) pkg.getTarget(name)).getAttr("actual");
  }

  private RootedPath createWorkspaceFile(String... contents) throws IOException {
    Path workspacePath = scratch.overwriteFile("WORKSPACE", contents);
    return RootedPath.toRootedPath(
        Root.fromPath(workspacePath.getParentDirectory()),
        PathFragment.create(workspacePath.getBaseName()));
  }

  private <T extends SkyValue> EvaluationResult<T> eval(SkyKey key)
      throws InterruptedException, AbruptExitException {
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("WORKSPACE")).build(),
            Root.fromPath(rootDirectory));
    return SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
  }

  @Test
  public void testLoadToChunkMapSimple() throws Exception {
    scratch.file("a.bzl", "a = 'a'");
    scratch.file("b.bzl", "b = 'b'");
    scratch.file("BUILD", "");
    RootedPath workspace =
        createWorkspaceFile(
            "workspace(name = 'good')",
            "load('//:a.bzl', 'a')",
            "x = 1  #for chunk break",
            "load('//:b.bzl', 'b')");
    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    assertThat(value1.getLoadToChunkMap()).containsEntry("//:a.bzl", 1);

    SkyKey key2 = WorkspaceFileValue.key(workspace, 2);
    EvaluationResult<WorkspaceFileValue> result2 = eval(key2);
    WorkspaceFileValue value2 = result2.get(key2);
    assertThat(value2.getLoadToChunkMap()).containsEntry("//:a.bzl", 1);
    assertThat(value2.getLoadToChunkMap()).containsEntry("//:b.bzl", 2);
  }

  @Test
  public void testLoadToChunkMapDoesNotOverrideDuplicate() throws Exception {
    scratch.file("a.bzl", "a = 'a'");
    scratch.file("BUILD", "");
    RootedPath workspace =
        createWorkspaceFile(
            "workspace(name = 'good')",
            "load('//:a.bzl', 'a')",
            "x = 1  #for chunk break",
            "load('//:a.bzl', 'a')");
    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    assertThat(value1.getLoadToChunkMap()).containsEntry("//:a.bzl", 1);

    SkyKey key2 = WorkspaceFileValue.key(workspace, 2);
    EvaluationResult<WorkspaceFileValue> result2 = eval(key2);
    WorkspaceFileValue value2 = result2.get(key2);
    assertThat(value2.getLoadToChunkMap()).containsEntry("//:a.bzl", 1);
    assertThat(value2.getLoadToChunkMap()).doesNotContainEntry("//:a.bzl", 2);
  }

  @Test
  public void testRepositoryMappingInChunks() throws Exception {
    scratch.file("b.bzl", "b = 'b'");
    scratch.file("BUILD", "");
    RootedPath workspace =
        createWorkspaceFile(
            "workspace(name = 'good')",
            "local_repository(name = 'a', path = '../a', repo_mapping = {'@x' : '@y'})",
            "load('//:b.bzl', 'b')",
            "local_repository(name = 'b', path = '../b', repo_mapping = {'@x' : '@y'})");
    RepositoryName a = RepositoryName.create("a");
    RepositoryName b = RepositoryName.create("b");
    RepositoryName y = RepositoryName.create("y");
    RepositoryName main = RepositoryName.create("");

    SkyKey key0 = WorkspaceFileValue.key(workspace, 0);
    EvaluationResult<WorkspaceFileValue> result0 = eval(key0);
    WorkspaceFileValue value0 = result0.get(key0);
    assertThat(value0.getRepositoryMapping())
        .containsEntry(a, ImmutableMap.of("x", y, "good", main));

    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    assertThat(value1.getRepositoryMapping())
        .containsEntry(a, ImmutableMap.of("x", y, "good", main));
    assertThat(value1.getRepositoryMapping())
        .containsEntry(b, ImmutableMap.of("x", y, "good", main));
  }

  @Test
  public void testBzlVisibility() throws Exception {
    setBuildLanguageOptions("--experimental_bzl_visibility=true");

    createWorkspaceFile(
        "workspace(name = 'foo')", //
        "load('//pkg:a.bzl', 'a')");
    scratch.file("pkg/BUILD");
    scratch.file(
        "pkg/a.bzl", //
        "visibility('private')");

    reporter.removeHandler(failFastHandler);
    SkyKey key = ExternalPackageFunction.key();
    eval(key);
    // The evaluation result ends up being null, probably due to the test framework swallowing
    // exceptions (similar to b/26382502). So let's just look for the error event instead of
    // asserting on the exception.
    assertContainsEvent(
        "Starlark file //pkg:a.bzl is not visible for loading from package //. Check the file's"
            + " `visibility()` declaration.");
  }

  @Test
  public void testInvalidRepo() throws Exception {
    // Test intentionally introduces errors.
    reporter.removeHandler(failFastHandler);

    createWorkspaceFile("workspace(name = 'foo$')");
    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent("Error in workspace: invalid user-provided repo name 'foo$'");
  }

  @Test
  public void testBindFunction() throws Exception {
    String[] lines = {"bind(name = 'foo/bar',", "actual = '//foo:bar')"};
    createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(getLabelMapping(pkg, "foo/bar")).isEqualTo(Label.parseCanonical("//foo:bar"));
    assertNoEvents();
  }

  @Test
  public void testBindArgsReversed() throws Exception {
    String[] lines = {"bind(actual = '//foo:bar', name = 'foo/bar')"};
    createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(getLabelMapping(pkg, "foo/bar")).isEqualTo(Label.parseCanonical("//foo:bar"));
    assertNoEvents();
  }

  @Test
  public void testNonExternalBinding() throws Exception {
    // Test intentionally introduces errors.
    reporter.removeHandler(failFastHandler);

    // name must be a valid label name.
    String[] lines = {"bind(name = 'foo:bar', actual = '//bar/baz')"};
    createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent("target names may not contain ':'");
  }

  @Test
  public void testWorkspaceFileParsingError() throws Exception {
    // Test intentionally introduces errors.
    reporter.removeHandler(failFastHandler);

    // //external:bar:baz is not a legal package.
    String[] lines = {"bind(name = 'foo/bar', actual = '//external:bar:baz')"};
    createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent("target names may not contain ':'");
  }

  @Test
  public void testRegisterToolchainsInvalidPattern() throws Exception {
    // Test intentionally introduces errors.
    reporter.removeHandler(failFastHandler);

    // //external:bar:baz is not a legal package.
    String[] lines = {"register_toolchains('/:invalid:label:syntax')"};
    createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent("error parsing target pattern");
  }

  @Test
  public void testNoWorkspaceFile() throws Exception {
    // Create and immediately delete to make sure we got the right file.
    RootedPath workspacePath = createWorkspaceFile();
    workspacePath.asPath().delete();

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(pkg.containsErrors()).isFalse();
    assertNoEvents();
  }

  @Test
  public void testListBindFunction() throws Exception {
    String[] lines = {
      "L = ['foo', 'bar']", "bind(name = '%s/%s' % (L[0], L[1]),", "actual = '//foo:bar')"
    };
    createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key();
    EvaluationResult<PackageValue> evaluationResult = eval(key);
    Package pkg = evaluationResult.get(key).getPackage();
    assertThat(getLabelMapping(pkg, "foo/bar")).isEqualTo(Label.parseCanonical("//foo:bar"));
    assertNoEvents();
  }

  @Test
  public void testMangledExternalWorkspaceFileIsIgnored() throws Exception {
    scratch.file("secondary/WORKSPACE", "garbage");
    RootedPath workspace =
        createWorkspaceFile(
            "workspace(name = 'good')",
            "local_repository(name = \"secondary\", path = \"./secondary/\")");

    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    RepositoryName main = RepositoryName.create("");
    RepositoryName secondary = RepositoryName.create("secondary");
    assertThat(value1.getRepositoryMapping())
        .containsEntry(secondary, ImmutableMap.of("good", main));
    assertNoEvents();
  }
  // tests of splitChunks, an internal helper function

  @Test
  public void testChunksNoLoad() {
    assertThat(split(parse("foo_bar = 1"))).isEqualTo("[(assignment)]");
  }

  @Test
  public void testChunksOneLoadAtTop() {
    assertThat(
            split(
                parse(
                    "load('//:foo.bzl', 'bar')", //
                    "foo_bar = 1")))
        .isEqualTo("[(load assignment)]");
  }

  @Test
  public void testChunksOneLoad() {
    assertThat(
            split(
                parse(
                    "foo_bar = 1",
                    //
                    "load('//:foo.bzl', 'bar')")))
        .isEqualTo("[(assignment)][(load)]");
  }

  @Test
  public void testChunksTwoSuccessiveLoads() {
    assertThat(
            split(
                parse(
                    "foo_bar = 1",
                    //
                    "load('//:foo.bzl', 'bar')",
                    "load('//:bar.bzl', 'foo')")))
        .isEqualTo("[(assignment)][(load load)]");
  }

  @Test
  public void testChunksTwoSucessiveLoadsWithNonLoadStatement() {
    assertThat(
            split(
                parse(
                    "foo_bar = 1",
                    //
                    "load('//:foo.bzl', 'bar')",
                    "load('//:bar.bzl', 'foo')",
                    "local_repository(name = 'foobar', path = '/bar/foo')")))
        .isEqualTo("[(assignment)][(load load expression)]");
  }

  @Test
  public void testChunksThreeLoadsThreeSegments() {
    assertThat(
            split(
                parse(
                    "foo_bar = 1",
                    //
                    "load('//:foo.bzl', 'bar')",
                    "load('//:bar.bzl', 'foo')",
                    "local_repository(name = 'foobar', path = '/bar/foo')",
                    //
                    "load('@foobar//:baz.bzl', 'bleh')")))
        .isEqualTo("[(assignment)][(load load expression)][(load)]");
  }

  @Test
  public void testChunksThreeLoadsThreeSegmentsWithContent() {
    assertThat(
            split(
                parse(
                    "foo_bar = 1",
                    //
                    "load('//:foo.bzl', 'bar')",
                    "load('//:bar.bzl', 'foo')",
                    "local_repository(name = 'foobar', path = '/bar/foo')",
                    //
                    "load('@foobar//:baz.bzl', 'bleh')",
                    "bleh()")))
        .isEqualTo("[(assignment)][(load load expression)][(load expression)]");
  }

  @Test
  public void testChunksMaySpanFiles() {
    assertThat(
            split(
                parse(
                    "x = 1", //
                    "load('m', 'y')"),
                parse(
                    "z = 1", //
                    "load('m', 'y2')")))
        .isEqualTo("[(assignment)][(load)(assignment)][(load)]");
  }

  // Returns a string that indicates the breakdown of statements into chunks.
  private static String split(StarlarkFile... files) {
    StringBuilder buf = new StringBuilder();
    for (List<StarlarkFile> chunk : WorkspaceFileFunction.splitChunks(Arrays.asList(files))) {
      buf.append('[');
      for (StarlarkFile partialFile : chunk) {
        buf.append('(');
        String sep = "";
        for (Statement stmt : partialFile.getStatements()) {
          buf.append(sep).append(Ascii.toLowerCase(stmt.kind().toString()));
          sep = " ";
        }
        buf.append(')');
      }
      buf.append(']');
    }
    return buf.toString();
  }

  private static StarlarkFile parse(String... lines) {
    return StarlarkFile.parse(ParserInput.fromLines(lines));
  }
}
