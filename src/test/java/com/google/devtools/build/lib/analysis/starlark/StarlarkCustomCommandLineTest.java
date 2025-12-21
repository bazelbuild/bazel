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
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.Arrays;
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
            .add(artifact1.getRoot())
            .add(artifact2.getRoot())
            .add(artifact3.getRoot())
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "bazel-out/k8-opt/bin/pkg/artifact2",
        "bazel-out/k8-opt/bin/artifact3",
        "bazel-out/k8-fastbuild/bin",
        "bazel-out/k8-opt/bin",
        "bazel-out/k8-opt/bin");
    verifyStrippedCommandLine(
        commandLine,
        "bazel-out/cfg/bin/pkg/artifact1",
        "bazel-out/cfg/bin/pkg/artifact2",
        "bazel-out/cfg/bin/artifact3",
        "bazel-out/k8-fastbuild/bin",
        "bazel-out/k8-opt/bin",
        "bazel-out/k8-opt/bin");
  }

  @Test
  public void addFormatted() throws Exception {
    CommandLine commandLine =
        builder
            .addFormatted("one", "--arg1=%s")
            .addFormatted("two", "--arg2=%s")
            .addFormatted("three", "--arg3=%s")
            .addFormatted(artifact1.getRoot(), "--arg1_root=%s")
            .addFormatted(artifact2.getRoot(), "--arg2_root=%s")
            .addFormatted(artifact3.getRoot(), "--arg3_root=%s")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg1=one",
        "--arg2=two",
        "--arg3=three",
        "--arg1_root=bazel-out/k8-fastbuild/bin",
        "--arg2_root=bazel-out/k8-opt/bin",
        "--arg3_root=bazel-out/k8-opt/bin");
  }

  @Test
  public void addFormattedPathMapped() throws Exception {
    CommandLine commandLine =
        builder
            .addFormatted(artifact1, "--arg1=%s")
            .addFormatted(artifact2, "--arg2=%s")
            .addFormatted(artifact3, "--arg3=%s")
            .addFormatted(artifact1.getRoot(), "--arg1_root=%s")
            .addFormatted(artifact2.getRoot(), "--arg2_root=%s")
            .addFormatted(artifact3.getRoot(), "--arg3_root=%s")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg1=bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "--arg2=bazel-out/k8-opt/bin/pkg/artifact2",
        "--arg3=bazel-out/k8-opt/bin/artifact3",
        "--arg1_root=bazel-out/k8-fastbuild/bin",
        "--arg2_root=bazel-out/k8-opt/bin",
        "--arg3_root=bazel-out/k8-opt/bin");
    verifyStrippedCommandLine(
        commandLine,
        "--arg1=bazel-out/cfg/bin/pkg/artifact1",
        "--arg2=bazel-out/cfg/bin/pkg/artifact2",
        "--arg3=bazel-out/cfg/bin/artifact3",
        "--arg1_root=bazel-out/k8-fastbuild/bin",
        "--arg2_root=bazel-out/k8-opt/bin",
        "--arg3_root=bazel-out/k8-opt/bin");
  }

  @Test
  public void argName(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, "one", "two", "three").setArgName("--arg"))
            .add(vectorArg(useNestedSet, "four").setArgName("--other_arg"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg", "one", "two", "three", "--other_arg", "four");
  }

  @Test
  public void terminateWith(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, "one", "two", "three").setTerminateWith("end1"))
            .add(vectorArg(useNestedSet, "four").setTerminateWith("end2"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "one", "two", "three", "end1", "four", "end2");
  }

  @Test
  public void formatEach(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, "one", "two", "three").setFormatEach("--arg=%s"))
            .add(vectorArg(useNestedSet, "four").setFormatEach("--other_arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg=one", "--arg=two", "--arg=three", "--other_arg=four");
  }

  @Test
  public void formatEachPathMapped(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, artifact1, artifact2, artifact3).setFormatEach("--arg=%s"))
            .add(
                vectorArg(useNestedSet, artifact1.getRoot(), artifact2.getRoot())
                    .setFormatEach("--arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg=bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "--arg=bazel-out/k8-opt/bin/pkg/artifact2",
        "--arg=bazel-out/k8-opt/bin/artifact3",
        "--arg=bazel-out/k8-fastbuild/bin",
        "--arg=bazel-out/k8-opt/bin");
    verifyStrippedCommandLine(
        commandLine,
        "--arg=bazel-out/cfg/bin/pkg/artifact1",
        "--arg=bazel-out/cfg/bin/pkg/artifact2",
        "--arg=bazel-out/cfg/bin/artifact3",
        "--arg=bazel-out/k8-fastbuild/bin",
        "--arg=bazel-out/k8-opt/bin");
  }

  @Test
  public void beforeEach(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, "one", "two", "three").setBeforeEach("b4"))
            .add(vectorArg(useNestedSet, "four").setBeforeEach("and"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "b4", "one", "b4", "two", "b4", "three", "and", "four");
  }

  @Test
  public void beforeEachPathMapped(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, artifact1, artifact2, artifact3).setBeforeEach("b4"))
            .add(
                vectorArg(useNestedSet, artifact1.getRoot(), artifact2.getRoot())
                    .setBeforeEach("b4"))
            .add(vectorArg(useNestedSet, artifact3).setBeforeEach("and"))
            .add(vectorArg(useNestedSet, artifact3.getRoot()).setBeforeEach("and"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "b4",
        "bazel-out/k8-fastbuild/bin/pkg/artifact1",
        "b4",
        "bazel-out/k8-opt/bin/pkg/artifact2",
        "b4",
        "bazel-out/k8-opt/bin/artifact3",
        "b4",
        "bazel-out/k8-fastbuild/bin",
        "b4",
        "bazel-out/k8-opt/bin",
        "and",
        "bazel-out/k8-opt/bin/artifact3",
        "and",
        "bazel-out/k8-opt/bin");
    verifyStrippedCommandLine(
        commandLine,
        "b4",
        "bazel-out/cfg/bin/pkg/artifact1",
        "b4",
        "bazel-out/cfg/bin/pkg/artifact2",
        "b4",
        "bazel-out/cfg/bin/artifact3",
        "b4",
        "bazel-out/k8-fastbuild/bin",
        "b4",
        "bazel-out/k8-opt/bin",
        "and",
        "bazel-out/cfg/bin/artifact3",
        "and",
        "bazel-out/k8-opt/bin");
  }

  @Test
  public void joinWith(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, "one", "two", "three").setJoinWith("..."))
            .add(vectorArg(useNestedSet, "four").setJoinWith("n/a"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "one...two...three", "four");
  }

  @Test
  public void joinWithPathMapped(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, artifact1, artifact2, artifact3).setJoinWith("..."))
            .add(
                vectorArg(useNestedSet, artifact1.getRoot(), artifact2.getRoot())
                    .setJoinWith("..."))
            .add(vectorArg(useNestedSet, artifact3).setJoinWith("..."))
            .add(vectorArg(useNestedSet, artifact3.getRoot()).setJoinWith("..."))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "bazel-out/k8-fastbuild/bin/pkg/artifact1...bazel-out/k8-opt/bin/pkg/artifact2...bazel-out/k8-opt/bin/artifact3",
        "bazel-out/k8-fastbuild/bin...bazel-out/k8-opt/bin",
        "bazel-out/k8-opt/bin/artifact3",
        "bazel-out/k8-opt/bin");
    verifyStrippedCommandLine(
        commandLine,
        "bazel-out/cfg/bin/pkg/artifact1...bazel-out/cfg/bin/pkg/artifact2...bazel-out/cfg/bin/artifact3",
        "bazel-out/k8-fastbuild/bin...bazel-out/k8-opt/bin",
        "bazel-out/cfg/bin/artifact3",
        "bazel-out/k8-opt/bin");
  }

  @Test
  public void formatJoined(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(
                vectorArg(useNestedSet, "one", "two", "three")
                    .setJoinWith("...")
                    .setFormatJoined("--arg=%s"))
            .add(
                vectorArg(useNestedSet, "four")
                    .setJoinWith("n/a")
                    .setFormatJoined("--other_arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "--arg=one...two...three", "--other_arg=four");
  }

  @Test
  public void formatJoinedPathMapped(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add(
                vectorArg(useNestedSet, artifact1, artifact2, artifact3)
                    .setJoinWith("...")
                    .setFormatJoined("--arg=%s"))
            .add(
                vectorArg(useNestedSet, artifact1.getRoot(), artifact2.getRoot())
                    .setJoinWith("...")
                    .setFormatJoined("--arg=%s"))
            .add(
                vectorArg(useNestedSet, artifact3)
                    .setJoinWith("...")
                    .setFormatJoined("--other_arg=%s"))
            .add(
                vectorArg(useNestedSet, artifact3.getRoot())
                    .setJoinWith("...")
                    .setFormatJoined("--other_arg=%s"))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--arg=bazel-out/k8-fastbuild/bin/pkg/artifact1...bazel-out/k8-opt/bin/pkg/artifact2...bazel-out/k8-opt/bin/artifact3",
        "--arg=bazel-out/k8-fastbuild/bin...bazel-out/k8-opt/bin",
        "--other_arg=bazel-out/k8-opt/bin/artifact3",
        "--other_arg=bazel-out/k8-opt/bin");
    verifyStrippedCommandLine(
        commandLine,
        "--arg=bazel-out/cfg/bin/pkg/artifact1...bazel-out/cfg/bin/pkg/artifact2...bazel-out/cfg/bin/artifact3",
        "--arg=bazel-out/k8-fastbuild/bin...bazel-out/k8-opt/bin",
        "--other_arg=bazel-out/cfg/bin/artifact3",
        "--other_arg=bazel-out/k8-opt/bin");
  }

  @Test
  public void emptyVectorArg_omit(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add("before")
            .add(
                vectorArg(useNestedSet)
                    .omitIfEmpty(true)
                    .setJoinWith(",")
                    .setFormatJoined("--empty=%s"))
            .add("after")
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    verifyCommandLine(commandLine, "before", "after");
  }

  @Test
  public void emptyVectorArg_noOmit(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .add("before")
            .add(
                vectorArg(useNestedSet)
                    .omitIfEmpty(false)
                    .setJoinWith(",")
                    .setFormatJoined("--empty=%s"))
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
                        // NestedSet doesn't support mixed types.
                        /* useNestedSet= */ false,
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
  public void flagPerLine(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .recordArgStart()
            .add(vectorArg(useNestedSet, "is", "line", "one").setArgName("--this"))
            .recordArgStart()
            .add(vectorArg(useNestedSet, "this", "is", "line", "two").setArgName("--and"))
            .recordArgStart()
            .add("--line_three")
            .add("single_arg")
            .recordArgStart()
            .add(vectorArg(useNestedSet, "", "line", "four", "has", "no").setTerminateWith("flag"))
            .build(/* flagPerLine= */ true, RepositoryMapping.EMPTY);
    verifyCommandLine(
        commandLine,
        "--this=is line one",
        "--and=this is line two",
        "--line_three=single_arg",
        "line four has no flag");
  }

  @Test
  public void flagPerLinePathMapped(@TestParameter boolean useNestedSet) throws Exception {
    CommandLine commandLine =
        builder
            .recordArgStart()
            .add(vectorArg(useNestedSet, artifact1, artifact2, artifact3).setArgName("--this"))
            .recordArgStart()
            .add(vectorArg(useNestedSet, artifact3, artifact2, artifact1).setArgName("--and"))
            .recordArgStart()
            .add("--line_three")
            .add("single_arg")
            .recordArgStart()
            .add(
                vectorArg(/* useNestedSet= */ false, "", artifact1, artifact2, artifact3)
                    .setTerminateWith("flag"))
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
  public void vectorArg_treeArtifactMissingExpansion_fails(@TestParameter boolean useNestedSet) {
    SpecialArtifact tree = createTreeArtifact("tree");
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    CommandLineExpansionException e =
        assertThrows(
            CommandLineExpansionException.class,
            () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
    assertThat(e).hasMessageThat().contains("Failed to expand directory <generated file tree>");
  }

  @Test
  public void vectorArgAddToFingerprint_expandFileset_includesInDigest(
      @TestParameter boolean useNestedSet) throws Exception {
    SpecialArtifact fileset = createFileset("fileset");
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, fileset).setExpandDirectories(true))
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
  public void vectorArgAddToFingerprint_expandTreeArtifact_includesInDigest(
      @TestParameter boolean useNestedSet) throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "child");
    // The files won't be read so MISSING_FILE_MARKER will do
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child, FileArtifactValue.MISSING_FILE_MARKER)
            .build();

    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, tree).setExpandDirectories(true))
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
  public void vectorArg_expandFilesetMissingExpansion_fails(@TestParameter boolean useNestedSet) {
    SpecialArtifact fileset = createFileset("fileset");
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, fileset).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    CommandLineExpansionException e =
        assertThrows(
            CommandLineExpansionException.class,
            () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
    assertThat(e)
        .hasMessageThat()
        .contains("Could not expand fileset: File:[[<execution_root>]bin]fileset");
  }

  @Test
  public void vectorArgArguments_expandsTreeArtifact(@TestParameter boolean useNestedSet)
      throws Exception {
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
            .add(vectorArg(useNestedSet, tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    Iterable<String> arguments = commandLine.arguments(fakeActionInputFileCache, PathMapper.NOOP);
    assertThat(arguments).containsExactly("bin/tree/child1", "bin/tree/child2");
  }

  @Test
  public void vectorArgArguments_expandsFileset(@TestParameter boolean useNestedSet)
      throws Exception {
    SpecialArtifact fileset = createFileset("fileset");
    FilesetOutputSymlink symlink1 = createFilesetSymlink("file1");
    FilesetOutputSymlink symlink2 = createFilesetSymlink("file2");

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putFileset(
        fileset, FilesetOutputTree.create(ImmutableList.of(symlink1, symlink2)));

    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, fileset).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);
    Iterable<String> arguments = commandLine.arguments(fakeActionInputFileCache, PathMapper.NOOP);

    assertThat(arguments).containsExactly("bin/fileset/file1", "bin/fileset/file2");
  }

  @Test
  public void vectorArgArguments_treeArtifactMissingExpansion_fails(
      @TestParameter boolean useNestedSet) {
    SpecialArtifact tree = createTreeArtifact("tree");
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, tree).setExpandDirectories(true))
            .build(/* flagPerLine= */ false, RepositoryMapping.EMPTY);

    assertThrows(
        CommandLineExpansionException.class,
        () -> commandLine.arguments(new FakeActionInputFileCache(), PathMapper.NOOP));
  }

  @Test
  public void vectorArgArguments_manuallyExpandedTreeArtifactMissingExpansion_fails(
      @TestParameter boolean useNestedSet) throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    CommandLine commandLine =
        builder
            .add(
                vectorArg(useNestedSet, tree)
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
  public void vectorArgArguments_filesetMissingExpansion_fails(
      @TestParameter boolean useNestedSet) {
    SpecialArtifact fileset = createFileset("fileset");
    CommandLine commandLine =
        builder
            .add(vectorArg(useNestedSet, fileset).setExpandDirectories(true))
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
    CommandLine commandLine =
        builder
            .add(
                vectorArg(useNestedSet, tree)
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

  @Test
  public void
      vectorArgArguments_expandDirectoriesDisabled_noMapEach_expansionDoesNotAffectActionKey(
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
    CommandLine commandLine =
        builder
            .add(
                vectorArg(useNestedSet, tree)
                    .setExpandDirectories(false)
                    .setLocation(Location.BUILTIN))
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
    assertThat(argumentsBefore).containsExactly("bin/tree");

    var inputMetadataProviderAfter = new FakeActionInputFileCache();
    inputMetadataProviderAfter.putTreeArtifact(tree, treeArtifactValueAfter);
    var argumentsAfter = commandLine.arguments(inputMetadataProviderAfter, PathMapper.NOOP);
    var fingerprintAfter = new Fingerprint();
    commandLine.addToFingerprint(
        new ActionKeyContext(),
        inputMetadataProviderAfter,
        CoreOptions.OutputPathsMode.OFF,
        fingerprintAfter);
    assertThat(argumentsAfter).containsExactly("bin/tree");

    assertThat(fingerprintBefore.hexDigestAndReset())
        .isEqualTo(fingerprintAfter.hexDigestAndReset());
  }

  @Test
  public void
      vectorArgArguments_expandDirectoriesDisabled_noManualExpansion_expansionDoesNotAffectActionKey(
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
    CommandLine commandLine =
        builder
            .add(
                vectorArg(useNestedSet, tree)
                    .setExpandDirectories(false)
                    .setLocation(Location.BUILTIN)
                    .setMapEach(
                        (StarlarkFunction)
                            execStarlark(
                                """
                                def map_each(x):
                                  return x.path
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
    assertThat(argumentsBefore).containsExactly("bin/tree");

    var inputMetadataProviderAfter = new FakeActionInputFileCache();
    inputMetadataProviderAfter.putTreeArtifact(tree, treeArtifactValueAfter);
    var argumentsAfter = commandLine.arguments(inputMetadataProviderAfter, PathMapper.NOOP);
    var fingerprintAfter = new Fingerprint();
    commandLine.addToFingerprint(
        new ActionKeyContext(),
        inputMetadataProviderAfter,
        CoreOptions.OutputPathsMode.OFF,
        fingerprintAfter);
    assertThat(argumentsAfter).containsExactly("bin/tree");

    assertThat(fingerprintBefore.hexDigestAndReset())
        .isEqualTo(fingerprintAfter.hexDigestAndReset());
  }

  private static VectorArg.Builder vectorArg(boolean useNestedSet, Object... elems) {
    if (useNestedSet) {
      Class<?> commonType;
      if (Arrays.stream(elems).allMatch(FileApi.class::isInstance)) {
        commonType = FileApi.class;
      } else if (Arrays.stream(elems).allMatch(FileRootApi.class::isInstance)) {
        commonType = FileRootApi.class;
      } else if (Arrays.stream(elems).allMatch(String.class::isInstance)) {
        commonType = String.class;
      } else {
        throw new IllegalArgumentException("Unsupported element types");
      }
      return new VectorArg.Builder(
              NestedSetBuilder.wrap(Order.STABLE_ORDER, Arrays.asList(elems)), commonType)
          .setLocation(Location.BUILTIN);
    } else {
      return new VectorArg.Builder(Tuple.of(elems)).setLocation(Location.BUILTIN);
    }
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
                  /* bzlTransitiveDigest= */ new byte[0],
                  /* docCommentsMap= */ ImmutableMap.of(),
                  /* unusedDocCommentLines= */ ImmutableList.of())),
          thread);
    }
  }
}
