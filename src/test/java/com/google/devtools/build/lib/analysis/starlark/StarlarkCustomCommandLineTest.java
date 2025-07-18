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
package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ArgChunk;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLine.SimpleArgChunk;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.PathMappers;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.starlark.StarlarkCustomCommandLine.VectorArg;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.ParserInput;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link StarlarkCustomCommandLine}. */
@RunWith(TestParameterInjector.class)
public final class StarlarkCustomCommandLineTest {

  private ArtifactRoot derivedRoot;
  private DerivedArtifact artifact1;
  private DerivedArtifact artifact2;
  private DerivedArtifact artifact3;
  private AbstractAction action;

  private final StarlarkCustomCommandLine.Builder builder =
      new StarlarkCustomCommandLine.Builder(StarlarkSemantics.DEFAULT);

  @Before
  public void createArtifacts() throws IOException {
    Path execRoot = new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/execroot");
    derivedRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "bin");

    ArtifactRoot derivedRoot2 =
        ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "bazel-out", "k8-fastbuild", "bin");
    ArtifactRoot derivedRoot3 =
        ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "bazel-out", "k8-opt", "bin");
    artifact1 = (DerivedArtifact) ActionsTestUtil.createArtifact(derivedRoot2, "pkg/artifact1");
    artifact2 = (DerivedArtifact) ActionsTestUtil.createArtifact(derivedRoot3, "pkg/artifact2");
    artifact3 = (DerivedArtifact) ActionsTestUtil.createArtifact(derivedRoot3, "artifact3");
    action =
        new ActionsTestUtil.MockAction(
            ImmutableList.of(artifact1, artifact2), ImmutableSet.of(artifact3)) {
          @Override
          public ImmutableMap<String, String> getExecutionInfo() {
            return ImmutableMap.of(ExecutionRequirements.SUPPORTS_PATH_MAPPING, "");
          }
        };
  }

  @Test
  public void add() throws Exception {
    CommandLine commandLine =
        builder
            .add("one")
            .add("two")
            .add("three")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "one", "two", "three");
  }

  @Test
  public void serializationWithFormattedArgumentsWorks() throws Exception {
    CommandLine original =
        builder.addFormatted("value", "key=%s").build(false, RepositoryMapping.EMPTY);
    CommandLine deserialized = RoundTripping.roundTrip(original);
    verifyCommandLine(deserialized, "key=value");
  }

  @Test
  public void addPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .add(artifact1)
            .add(artifact2)
            .add(artifact3)
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "bazel-out/k8-opt/bin/pkg/artifact2",
        "bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "bazel-out/cfg/bin/pkg/artifact1",
        "bazel-out/cfg/bin/pkg/artifact2",
        "bazel-out/cfg/bin/artifact3");
  }

  @Test
  public void addFormatted() throws Exception {
    CommandLine commandLine =
        builder
            .addFormatted("one", "--arg1=%s")
            .addFormatted("two", "--arg2=%s")
            .addFormatted("three", "--arg3=%s")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg1=one", "--arg2=two", "--arg3=three");
  }

  @Test
  public void addFormattedPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .addFormatted(artifact1, "--arg1=%s")
            .addFormatted(artifact2, "--arg2=%s")
            .addFormatted(artifact3, "--arg3=%s")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg1=bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "--arg2=bazel-out/k8-opt/bin/pkg/artifact2",
        "--arg3=bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "--arg1=bazel-out/cfg/bin/pkg/artifact1",
        "--arg2=bazel-out/cfg/bin/pkg/artifact2",
        "--arg3=bazel-out/cfg/bin/artifact3");
  }

  @Test
  public void argName() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg("one", "two", "three").setArgName("--arg"))
            .add(vectorArg("four").setArgName("--other_arg"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg", "one", "two", "three", "--other_arg", "four");
  }

  @Test
  public void terminateWith() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg("one", "two", "three").setTerminateWith("end1"))
            .add(vectorArg("four").setTerminateWith("end2"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "one", "two", "three", "end1", "four", "end2");
  }

  @Test
  public void formatEach() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg("one", "two", "three").setFormatEach("--arg=%s"))
            .add(vectorArg("four").setFormatEach("--other_arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg=one", "--arg=two", "--arg=three", "--other_arg=four");
  }

  @Test
  public void formatEachPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(artifact1, artifact2, artifact3).setFormatEach("--arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg=bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "--arg=bazel-out/k8-opt/bin/pkg/artifact2",
        "--arg=bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "--arg=bazel-out/cfg/bin/pkg/artifact1",
        "--arg=bazel-out/cfg/bin/pkg/artifact2",
        "--arg=bazel-out/cfg/bin/artifact3");
  }

  @Test
  public void beforeEach() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg("one", "two", "three").setBeforeEach("b4"))
            .add(vectorArg("four").setBeforeEach("and"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "b4", "one", "b4", "two", "b4", "three", "and", "four");
  }

  @Test
  public void beforeEachPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(artifact1, artifact2, artifact3).setBeforeEach("b4"))
            .add(vectorArg(artifact3).setBeforeEach("and"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "b4",
        "bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "b4",
        "bazel-out/k8-opt/bin/pkg/artifact2",
        "b4",
        "bazel-out/k8-opt/bin/artifact3",
        "and",
        "bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "b4",
        "bazel-out/cfg/bin/pkg/artifact1",
        "b4",
        "bazel-out/cfg/bin/pkg/artifact2",
        "b4",
        "bazel-out/cfg/bin/artifact3",
        "and",
        "bazel-out/cfg/bin/artifact3");
  }

  @Test
  public void joinWith() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg("one", "two", "three").setJoinWith("..."))
            .add(vectorArg("four").setJoinWith("n/a"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "one...two...three", "four");
  }

  @Test
  public void joinWithPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(artifact1, artifact2, artifact3).setJoinWith("..."))
            .add(vectorArg(artifact3).setJoinWith("..."))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "bazel-out/k8-fastbuild/bin/pkg/artifact1...bazel-out/k8-opt/bin/pkg/artifact2...bazel-out/k8-opt/bin/artifact3",
        "bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "bazel-out/cfg/bin/pkg/artifact1...bazel-out/cfg/bin/pkg/artifact2...bazel-out/cfg/bin/artifact3",
        "bazel-out/cfg/bin/artifact3");
  }

  @Test
  public void formatJoined() throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg("one", "two", "three").setJoinWith("...").setFormatJoined("--arg=%s"))
            .add(vectorArg("four").setJoinWith("n/a").setFormatJoined("--other_arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg=one...two...three", "--other_arg=four");
  }

  @Test
  public void formatJoinedPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .add(
                vectorArg(artifact1, artifact2, artifact3)
                    .setJoinWith("...")
                    .setFormatJoined("--arg=%s"))
            .add(vectorArg(artifact3).setJoinWith("...").setFormatJoined("--other_arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg=bazel-out/k8-fastbuild/bin/pkg/artifact1...bazel-out/k8-opt/bin/pkg/artifact2...bazel-out/k8-opt/bin/artifact3",
        "--other_arg=bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "--arg=bazel-out/cfg/bin/pkg/artifact1...bazel-out/cfg/bin/pkg/artifact2...bazel-out/cfg/bin/artifact3",
        "--other_arg=bazel-out/cfg/bin/artifact3");
  }

  @Test
  public void emptyVectorArg_omit() throws Exception {
    CommandLine commandLine =
        builder
            .add("before")
            .add(vectorArg().omitIfEmpty(true).setJoinWith(",").setFormatJoined("--empty=%s"))
            .add("after")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "before", "after");
  }

  @Test
  public void emptyVectorArg_noOmit() throws Exception {
    CommandLine commandLine =
        builder
            .add("before")
            .add(vectorArg().omitIfEmpty(false).setJoinWith(",").setFormatJoined("--empty=%s"))
            .add("after")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "before", "--empty=", "after");
  }

  @Test
  public void uniquifyPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .add(
                vectorArg(
                        artifact1,
                        artifact1,
                        artifact2,
                        artifact3,
                        artifact1.getExecPathString(),
                        artifact1.getExecPathString(),
                        artifact2.getExecPathString(),
                        artifact3.getExecPathString())
                    .uniquify(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "bazel-out/k8-opt/bin/pkg/artifact2",
        "bazel-out/k8-opt/bin/artifact3");
    verifyStrippedCommandLine(
        commandLine,
        "bazel-out/cfg/bin/pkg/artifact1",
        "bazel-out/cfg/bin/pkg/artifact2",
        "bazel-out/cfg/bin/artifact3",
        "bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "bazel-out/k8-opt/bin/pkg/artifact2",
        "bazel-out/k8-opt/bin/artifact3");
  }

  @Test
  public void flagPerLine() throws Exception {
    CommandLine commandLine =
        builder
            .recordArgStart()
            .add(vectorArg("is", "line", "one").setArgName("--this"))
            .recordArgStart()
            .add(vectorArg("this", "is", "line", "two").setArgName("--and"))
            .recordArgStart()
            .add("--line_three")
            .add("single_arg")
            .recordArgStart()
            .add(vectorArg("", "line", "four", "has", "no").setTerminateWith("flag"))
            .build(/* flagPerLine= */ true, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--this=is line one",
        "--and=this is line two",
        "--line_three=single_arg",
        "line four has no flag");
  }

  @Test
  public void flagPerLinePathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .recordArgStart()
            .add(vectorArg(artifact1, artifact2, artifact3).setArgName("--this"))
            .recordArgStart()
            .add(vectorArg(artifact3, artifact2, artifact1).setArgName("--and"))
            .recordArgStart()
            .add("--line_three")
            .add("single_arg")
            .recordArgStart()
            .add(vectorArg("", artifact1, artifact2, artifact3).setTerminateWith("flag"))
            .build(/* flagPerLine= */ true, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--this=bazel-out/k8-fastbuild/bin/pkg/artifact1 bazel-out/k8-opt/bin/pkg/artifact2"
            + " bazel-out/k8-opt/bin/artifact3",
        "--and=bazel-out/k8-opt/bin/artifact3 bazel-out/k8-opt/bin/pkg/artifact2"
            + " bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "--line_three=single_arg",
        "bazel-out/k8-fastbuild/bin/pkg/artifact1 bazel-out/k8-opt/bin/pkg/artifact2"
            + " bazel-out/k8-opt/bin/artifact3 flag");
    verifyStrippedCommandLine(
        commandLine,
        "--this=bazel-out/cfg/bin/pkg/artifact1 bazel-out/cfg/bin/pkg/artifact2"
            + " bazel-out/cfg/bin/artifact3",
        "--and=bazel-out/cfg/bin/artifact3 bazel-out/cfg/bin/pkg/artifact2"
            + " bazel-out/cfg/bin/pkg/artifact1",
        "--line_three=single_arg",
        "bazel-out/cfg/bin/pkg/artifact1 bazel-out/cfg/bin/pkg/artifact2"
            + " bazel-out/cfg/bin/artifact3 flag");
  }

  @Test
  public void vectorArgAddToFingerprint_treeArtifactMissingExpansion_fails() {
    SpecialArtifact tree = createTreeArtifact("tree");
    CommandLine commandLine =
        builder
            .add(vectorArg(tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    CommandLineExpansionException e =
        assertThrows(
            CommandLineExpansionException.class,
            () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
    assertThat(e).hasMessageThat().contains("Failed to expand directory <generated file tree>");
  }

  @Test
  public void vectorArgAddToFingerprint_expandFileset_includesInDigest() throws Exception {
    SpecialArtifact fileset = createFileset("fileset");
    CommandLine commandLine =
        builder
            .add(vectorArg(fileset).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    FilesetOutputSymlink symlink1 = createFilesetSymlink("file1");
    FilesetOutputSymlink symlink2 = createFilesetSymlink("file2");
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Fingerprint fingerprint = new Fingerprint();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putFileset(
        fileset, FilesetOutputTree.create(ImmutableList.of(symlink1, symlink2)));
    commandLine.addToFingerprint(
        actionKeyContext, fakeActionInputFileCache, CoreOptions.OutputPathsMode.OFF, fingerprint);

    assertThat(fingerprint.digestAndReset()).isNotEmpty();
  }

  @Test
  public void vectorArgAddToFingerprint_expandTreeArtifact_includesInDigest() throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "child");
    // The files won't be read so MISSING_FILE_MARKER will do
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child, FileArtifactValue.MISSING_FILE_MARKER)
            .build();

    CommandLine commandLine =
        builder
            .add(vectorArg(tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    ActionKeyContext actionKeyContext = new ActionKeyContext();
    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(tree, treeArtifactValue);

    Fingerprint fingerprint = new Fingerprint();
    commandLine.addToFingerprint(
        actionKeyContext, fakeActionInputFileCache, CoreOptions.OutputPathsMode.OFF, fingerprint);
    assertThat(fingerprint.digestAndReset()).isNotEmpty();
  }

  @Test
  public void vectorArgAddToFingerprint_expandFilesetMissingExpansion_fails() {
    SpecialArtifact fileset = createFileset("fileset");
    CommandLine commandLine =
        builder
            .add(vectorArg(fileset).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Fingerprint fingerprint = new Fingerprint();

    assertThrows(
        CommandLineExpansionException.class,
        () ->
            commandLine.addToFingerprint(
                actionKeyContext,
                new FakeActionInputFileCache(),
                CoreOptions.OutputPathsMode.OFF,
                fingerprint));
  }

  @Test
  public void vectorArgArguments_expandsTreeArtifact() throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(tree, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(tree, "child2");
    // The files won't be read so MISSING_FILE_MARKER will do
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child1, FileArtifactValue.MISSING_FILE_MARKER)
            .putChild(child2, FileArtifactValue.MISSING_FILE_MARKER)
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(tree, treeArtifactValue);

    CommandLine commandLine =
        builder
            .add(vectorArg(tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    Iterable<String> arguments = commandLine.arguments(fakeActionInputFileCache, PathMapper.NOOP);
    assertThat(arguments).containsExactly("bin/tree/child1", "bin/tree/child2");
  }

  @Test
  public void vectorArgArguments_expandsFileset() throws Exception {
    SpecialArtifact fileset = createFileset("fileset");
    FilesetOutputSymlink symlink1 = createFilesetSymlink("file1");
    FilesetOutputSymlink symlink2 = createFilesetSymlink("file2");

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putFileset(
        fileset, FilesetOutputTree.create(ImmutableList.of(symlink1, symlink2)));

    CommandLine commandLine =
        builder
            .add(vectorArg(fileset).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    Iterable<String> arguments = commandLine.arguments(fakeActionInputFileCache, PathMapper.NOOP);

    assertThat(arguments).containsExactly("bin/fileset/file1", "bin/fileset/file2");
  }

  @Test
  public void vectorArgArguments_treeArtifactMissingExpansion_fails() {
    SpecialArtifact tree = createTreeArtifact("tree");
    CommandLine commandLine =
        builder
            .add(vectorArg(tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    assertThrows(
        CommandLineExpansionException.class,
        () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
  }

  @Test
  public void vectorArgArguments_manuallyExpandedTreeArtifactMissingExpansion_fails()
      throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    CommandLine commandLine =
        builder
            .add(
                vectorArg(tree)
                    .setExpandDirectories(false)
                    .setLocation(Location.BUILTIN)
                    .setMapEach(
                        (StarlarkFunction)
                            execStarlark(
                                """
                                def map_each(x, expander):
                                  expander.expand(x)
                                map_each
                                """)))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    CommandLineExpansionException e =
        assertThrows(
            CommandLineExpansionException.class,
            () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
    assertThat(e).hasMessageThat().contains("Failed to expand directory <generated file tree>");
  }

  @Test
  public void vectorArgArguments_filesetMissingExpansion_fails() {
    SpecialArtifact fileset = createFileset("fileset");
    CommandLine commandLine =
        builder
            .add(vectorArg(fileset).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    assertThrows(
        CommandLineExpansionException.class,
        () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
  }

  @Test
  public void vectorArgArguments_expandDirectoriesDisabled_manualExpansionReflectedInActionKey(
      @TestParameter boolean useNestedSet) throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(tree, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(tree, "child2");
    // The files won't be read so MISSING_FILE_MARKER will do
    TreeArtifactValue treeArtifactValueBefore =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child1, FileArtifactValue.MISSING_FILE_MARKER)
            .putChild(child2, FileArtifactValue.MISSING_FILE_MARKER)
            .build();
    TreeArtifactValue treeArtifactValueAfter =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child1, FileArtifactValue.MISSING_FILE_MARKER)
            .build();
    var vectorArg =
        useNestedSet
            ? new VectorArg.Builder(
                NestedSetBuilder.wrap(Order.STABLE_ORDER, ImmutableList.of(tree)), Artifact.class)
            : new VectorArg.Builder(Tuple.of(tree));
    CommandLine commandLine =
        builder
            .add(
                vectorArg
                    .setExpandDirectories(false)
                    .setLocation(Location.BUILTIN)
                    .setMapEach(
                        (StarlarkFunction)
                            execStarlark(
                                """
                                def map_each(x, expander):
                                  return [f.path for f in expander.expand(x)]
                                map_each
                                """)))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    var inputMetadataProviderBefore = new FakeActionInputFileCache();
    inputMetadataProviderBefore.putTreeArtifact(tree, treeArtifactValueBefore);
    var argumentsBefore = commandLine.arguments(inputMetadataProviderBefore, PathMapper.NOOP);
    var fingerprintBefore = new Fingerprint();
    commandLine.addToFingerprint(
        new ActionKeyContext(),
        inputMetadataProviderBefore,
        CoreOptions.OutputPathsMode.OFF,
        fingerprintBefore);
    assertThat(argumentsBefore).containsExactly("bin/tree/child1", "bin/tree/child2");

    var inputMetadataProviderAfter = new FakeActionInputFileCache();
    inputMetadataProviderAfter.putTreeArtifact(tree, treeArtifactValueAfter);
    var argumentsAfter = commandLine.arguments(inputMetadataProviderAfter, PathMapper.NOOP);
    var fingerprintAfter = new Fingerprint();
    commandLine.addToFingerprint(
        new ActionKeyContext(),
        inputMetadataProviderAfter,
        CoreOptions.OutputPathsMode.OFF,
        fingerprintAfter);
    assertThat(argumentsAfter).containsExactly("bin/tree/child1");

    assertThat(fingerprintBefore.hexDigestAndReset())
        .isNotEqualTo(fingerprintAfter.hexDigestAndReset());
  }

  private static VectorArg.Builder vectorArg(Object... elems) {
    return new VectorArg.Builder(Tuple.of(elems)).setLocation(Location.BUILTIN);
  }

  private static void verifyCommandLine(CommandLine commandLine, String... expected)
      throws CommandLineExpansionException, InterruptedException {
    verifyCommandLine(PathMapper.NOOP, commandLine, expected);
  }

  private static void verifyCommandLine(
      PathMapper pathMapper, CommandLine commandLine, String... expected)
      throws CommandLineExpansionException, InterruptedException {
    ArgChunk chunk = commandLine.expand(new FakeActionInputFileCache(), pathMapper);
    assertThat(chunk.arguments(pathMapper)).containsExactlyElementsIn(expected).inOrder();
    // Check consistency of the total argument length calculation with SimpleArgChunk, which
    // materializes strings and adds up their lengths.
    assertThat(chunk.totalArgLength(pathMapper))
        .isEqualTo(new SimpleArgChunk(chunk.arguments(pathMapper)).totalArgLength(pathMapper));
  }

  private void verifyStrippedCommandLine(CommandLine commandLine, String... expected)
      throws CommandLineExpansionException, InterruptedException {
    verifyCommandLine(
        PathMappers.create(action, CoreOptions.OutputPathsMode.STRIP, /* isStarlarkAction= */ true),
        commandLine,
        expected);
  }

  private SpecialArtifact createFileset(String relativePath) {
    return createSpecialArtifact(relativePath, SpecialArtifactType.FILESET);
  }

  private FilesetOutputSymlink createFilesetSymlink(String relativePath) {
    return new FilesetOutputSymlink(
        PathFragment.create(relativePath),
        ActionsTestUtil.createArtifact(derivedRoot, "some/target"),
        FileArtifactValue.createForNormalFile(new byte[] {1}, null, 1));
  }

  private SpecialArtifact createTreeArtifact(String relativePath) {
    SpecialArtifact tree = createSpecialArtifact(relativePath, SpecialArtifactType.TREE);
    tree.setGeneratingActionKey(ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0));
    return tree;
  }

  private SpecialArtifact createSpecialArtifact(String relativePath, SpecialArtifactType type) {
    return SpecialArtifact.create(
        derivedRoot,
        derivedRoot.getExecPath().getRelative(relativePath),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        type);
  }

  private static Object execStarlark(String code) throws Exception {
    try (Mutability mutability = Mutability.create("test")) {
      StarlarkThread thread = StarlarkThread.createTransient(mutability, StarlarkSemantics.DEFAULT);
      return Starlark.execFile(
          ParserInput.fromString(code, "test/label.bzl"),
          FileOptions.DEFAULT,
          Module.withPredeclaredAndData(
              StarlarkSemantics.DEFAULT,
              ImmutableMap.of(),
              BazelModuleContext.create(
                  BazelModuleKey.createFakeModuleKeyForTesting(
                      Label.parseCanonicalUnchecked("//test:label")),
                  RepositoryMapping.EMPTY,
                  "test/label.bzl",
                  /* loads= */ ImmutableList.of(),
                  /* bzlTransitiveDigest= */ new byte[0])),
          thread);
    }
  }
}
