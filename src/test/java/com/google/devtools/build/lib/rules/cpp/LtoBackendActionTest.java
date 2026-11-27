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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.ensureMemoizedIsInitializedIsSet;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.AbstractCommandLine;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.skyframe.serialization.ArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.rules.cpp.LtoBackendAction}. */
@RunWith(JUnit4.class)
public class LtoBackendActionTest extends BuildViewTestCase {
  private Artifact bitcode1Artifact;
  private Artifact bitcode2Artifact;
  private Artifact index1Artifact;
  private Artifact index2Artifact;
  private Artifact imports1Artifact;
  private Artifact imports2Artifact;
  private Artifact.DerivedArtifact destinationArtifact;
  private BitcodeFiles allBitcodeFiles;
  private AnalysisTestUtil.CollectingAnalysisEnvironment collectingAnalysisEnvironment;
  private ActionExecutionContext context;

  @Before
  public final void createArtifacts() throws Exception {
    collectingAnalysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(getTestAnalysisEnvironment());
    bitcode1Artifact = getSourceArtifact("bitcode1.o");
    bitcode2Artifact = getSourceArtifact("bitcode2.o");
    index1Artifact = getSourceArtifact("bitcode1.thinlto.bc");
    index2Artifact = getSourceArtifact("bitcode2.thinlto.bc");
    scratch.file("bitcode1.imports");
    scratch.file("bitcode2.imports", "bitcode1.o");
    imports1Artifact = getSourceArtifact("bitcode1.imports");
    imports2Artifact = getSourceArtifact("bitcode2.imports");
    destinationArtifact = getBinArtifactWithNoOwner("output");
    allBitcodeFiles =
        new BitcodeFiles(
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode1Artifact, bitcode2Artifact));
  }

  @Before
  public final void createExecutorAndContext() throws Exception {
    Executor executor = new TestExecutorBuilder(fileSystem, directories).build();
    context =
        new ActionExecutionContext(
            executor,
            /* inputMetadataProvider= */ null,
            ActionInputPrefetcher.NONE,
            actionKeyContext,
            /* outputMetadataStore= */ null,
            /* rewindingEnabled= */ false,
            LostInputsCheck.NONE,
            new FileOutErr(),
            new StoredEventHandler(),
            /* clientEnv= */ ImmutableMap.of(),
            /* actionFileSystem= */ null,
            DiscoveredModulesPruner.DEFAULT,
            SyscallCache.NO_CACHE,
            ThreadStateReceiver.NULL_INSTANCE);
  }

  @Test
  public void testEmptyImports() throws Exception {
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode1Artifact, index1Artifact),
            allBitcodeFiles,
            imports1Artifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));

    collectingAnalysisEnvironment.registerAction(action);
    assertThat(action.getOwner().getLabel())
        .isEqualTo(ActionsTestUtil.NULL_ACTION_OWNER.getLabel());
    assertThat(action.getInputs().toList()).containsExactly(bitcode1Artifact, index1Artifact);
    assertThat(action.getOutputs()).containsExactly(destinationArtifact);
    assertThat(action.getSpawnForTesting().getLocalResources())
        .isEqualTo(AbstractAction.DEFAULT_RESOURCE_SET);
    assertThat(action.getArguments()).containsExactly("/bin/clang");
    assertThat(action.getProgressMessage()).isEqualTo("LTO Backend Compile output");
    assertThat(action.inputsKnown()).isFalse();

    // Discover inputs, which should not add any inputs since bitcode1.imports is empty.
    action.discoverInputs(context);
    assertThat(action.inputsKnown()).isTrue();
    assertThat(action.getInputs().toList()).containsExactly(bitcode1Artifact, index1Artifact);
  }

  @Test
  public void testNonEmptyImports() throws Exception {
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode2Artifact, index2Artifact),
            allBitcodeFiles,
            imports2Artifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));
    collectingAnalysisEnvironment.registerAction(action);
    assertThat(action.getOwner().getLabel())
        .isEqualTo(ActionsTestUtil.NULL_ACTION_OWNER.getLabel());
    assertThat(action.getInputs().toList()).containsExactly(bitcode2Artifact, index2Artifact);
    assertThat(action.getOutputs()).containsExactly(destinationArtifact);
    assertThat(action.getSpawnForTesting().getLocalResources())
        .isEqualTo(AbstractAction.DEFAULT_RESOURCE_SET);
    assertThat(action.getArguments()).containsExactly("/bin/clang");
    assertThat(action.getProgressMessage()).isEqualTo("LTO Backend Compile output");
    assertThat(action.inputsKnown()).isFalse();

    // Discover inputs, which should add bitcode1.o which is listed in bitcode2.imports.
    action.discoverInputs(context);
    assertThat(action.inputsKnown()).isTrue();
    assertThat(action.getInputs().toList())
        .containsExactly(bitcode1Artifact, bitcode2Artifact, index2Artifact);
  }

  @Test
  public void testGetInputFilesForExtraAction() throws Exception {
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode2Artifact, index2Artifact),
            allBitcodeFiles,
            imports2Artifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));
    collectingAnalysisEnvironment.registerAction(action);

    NestedSet<Artifact> extraActionInputs = action.getInputFilesForExtraAction(context);
    assertThat(extraActionInputs.toList()).containsExactly(bitcode1Artifact);
  }

  private enum KeyAttributes {
    EXECUTABLE,
    IMPORTS_INFO,
    INPUT,
    FIXED_ENVIRONMENT
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact artifactA = getSourceArtifact("a");
    final Artifact artifactB = getSourceArtifact("b");
    final Artifact artifactAimports = getSourceArtifact("a.imports");
    final Artifact artifactBimports = getSourceArtifact("b.imports");

    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) {
            PathFragment executable =
                attributesToFlip.contains(KeyAttributes.EXECUTABLE)
                    ? artifactA.getExecPath()
                    : artifactB.getExecPath();

            Artifact imports;
            if (attributesToFlip.contains(KeyAttributes.IMPORTS_INFO)) {
              imports = artifactAimports;
            } else {
              imports = artifactBimports;
            }

            Artifact input;
            if (attributesToFlip.contains(KeyAttributes.INPUT)) {
              input = artifactA;
            } else {
              input = artifactB;
            }

            Map<String, String> env = new HashMap<>();
            if (attributesToFlip.contains(KeyAttributes.FIXED_ENVIRONMENT)) {
              env.put("foo", "bar");
            }

            SpawnAction action =
                LtoBackendAction.create(
                    ActionsTestUtil.NULL_ACTION_OWNER,
                    targetConfig,
                    NestedSetBuilder.create(Order.STABLE_ORDER, imports, input),
                    new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER)),
                    imports,
                    ImmutableSet.of(destinationArtifact),
                    CommandLines.builder().addSingleArgument(executable).build(),
                    ActionEnvironment.create(ImmutableMap.copyOf(env)));
            collectingAnalysisEnvironment.registerAction(action);
            return action;
          }
        },
        actionKeyContext);
  }

  @Test
  public void discoverInputs_missingInputErrorMessage() throws Exception {
    FileSystemUtils.writeIsoLatin1(imports1Artifact.getPath(), "file1.o", "file2.o", "file3.o");

    Artifact index1Artifact = getSourceArtifact("file2.o");
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, imports1Artifact, index1Artifact),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, index1Artifact)),
            imports1Artifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));

    ActionExecutionException e =
        assertThrows(ActionExecutionException.class, () -> action.discoverInputs(context));

    assertThat(e).hasMessageThat().endsWith("(first 10): file1.o, file3.o");
  }

  @Test
  public void discoverInputs_cacheNotFoundException_throwsRemoteCacheEvictedCode()
      throws Exception {
    FileSystem customFs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public synchronized InputStream getInputStream(PathFragment path) throws IOException {
            if (path.getBaseName().endsWith(".imports")) {
              throw new CacheNotFoundException(Digest.getDefaultInstance(), path);
            }
            return super.getInputStream(path);
          }
        };
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(customFs.getPath("/")));
    Artifact importsArtifact = ActionsTestUtil.createArtifact(root, "bitcode1.imports");
    Artifact index1Artifact = getSourceArtifact("file2.o");
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, importsArtifact, index1Artifact),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, index1Artifact)),
            importsArtifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));

    ActionExecutionException e =
        assertThrows(ActionExecutionException.class, () -> action.discoverInputs(context));

    assertThat(e.getExitCode()).isEqualTo(ExitCode.REMOTE_CACHE_EVICTED);
  }

  @Test
  public void discoverInputs_bulkTransferException_throwsRemoteCacheEvictedCode() throws Exception {
    FileSystem customFs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public synchronized InputStream getInputStream(PathFragment path) throws IOException {
            if (path.getBaseName().endsWith(".imports")) {
              throw new BulkTransferException(
                  new CacheNotFoundException(Digest.getDefaultInstance(), path));
            }
            return super.getInputStream(path);
          }
        };
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(customFs.getPath("/")));
    Artifact importsArtifact = ActionsTestUtil.createArtifact(root, "bitcode1.imports");
    Artifact index1Artifact = getSourceArtifact("file2.o");
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, importsArtifact, index1Artifact),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, index1Artifact)),
            importsArtifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));

    ActionExecutionException e =
        assertThrows(ActionExecutionException.class, () -> action.discoverInputs(context));

    assertThat(e.getExitCode()).isEqualTo(ExitCode.REMOTE_CACHE_EVICTED);
  }

  @Test
  public void serializationRoundTrip_resetsInputs() throws Exception {
    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(
                Order.STABLE_ORDER, bitcode2Artifact, index2Artifact, imports2Artifact),
            allBitcodeFiles,
            imports2Artifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder()
                .addSingleArgument(scratch.file("/bin/clang").asFragment())
                .build(),
            ActionEnvironment.create(ImmutableMap.of()));

    destinationArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    ensureMemoizedIsInitializedIsSet(action);

    String originalStructure = dumpStructureWithEquivalenceReduction(action);

    ImmutableList<Artifact> originalInputs = action.getInputs().toList();
    var unused = action.discoverInputs(context);
    assertThat(action.getInputs().toList()).isNotEqualTo(originalInputs);

    new SerializationTester(action)
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .addCodec(ArrayCodec.forComponentType(Artifact.class))
        .setVerificationFunction(
            (unusedInput, deserialized) ->
                assertThat(dumpStructureWithEquivalenceReduction(deserialized))
                    .isEqualTo(originalStructure))
        .addDependencies(getCommonSerializationDependencies())
        .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
        .runTests();
  }

  @Test
  public void describeKey_includesLtoFields() {
    LtoBackendAction actionWithImports =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, index1Artifact, imports1Artifact),
            allBitcodeFiles,
            imports1Artifact,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder().addSingleArgument("clang").build(),
            ActionEnvironment.create(ImmutableMap.of("FOO", "bar")));
    LtoBackendAction actionWithoutImports =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, index2Artifact),
            null,
            null,
            ImmutableSet.of(destinationArtifact),
            CommandLines.builder().addSingleArgument("clang").build(),
            ActionEnvironment.create(ImmutableMap.of("FOO", "bar")));

    String descWithImports = actionWithImports.describeKey();
    String descWithoutImports = actionWithoutImports.describeKey();
    assertThat(descWithImports).contains("Mnemonic: CcLtoBackendCompile");
    assertThat(descWithImports).contains("Mandatory input: " + index1Artifact.getExecPathString());
    assertThat(descWithImports).contains("Imports: " + imports1Artifact.getExecPathString());
    assertThat(descWithImports).contains("Bitcode files digest:");
    assertThat(descWithoutImports).doesNotContain("Imports:");
    assertThat(descWithoutImports)
        .contains("Mandatory input: " + index2Artifact.getExecPathString());
    assertThat(descWithImports).isNotEqualTo(descWithoutImports);
  }

  private static CommandLines createLtoCommandLines(
      Artifact index, Artifact output, Artifact bitcode) {
    return CommandLines.builder()
        .addSingleArgument(PathFragment.create("/bin/clang"))
        .addCommandLine(
            new AbstractCommandLine() {
              @Override
              public ImmutableList<String> arguments() {
                return arguments(/* inputMetadataProvider= */ null, PathMapper.NOOP);
              }

              @Override
              public ImmutableList<String> arguments(
                  InputMetadataProvider inputMetadataProvider, PathMapper pathMapper) {
                return ImmutableList.of(
                    "-fthinlto-index="
                        + (index == null ? "/dev/null" : pathMapper.getMappedExecPathString(index)),
                    "-o",
                    pathMapper.getMappedExecPathString(output),
                    "-c",
                    pathMapper.getMappedExecPathString(bitcode));
              }
            })
        .build();
  }

  @Test
  public void ltoBackendPathMapper_stripsTargetLtoPrefixAndPreservesConfigOnBitcode()
      throws Exception {
    useConfiguration("--experimental_output_paths=off");

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact indexA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact importsA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputA = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    Artifact indexB =
        getBinArtifactWithNoOwner("pkg/test_b.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact importsB =
        getBinArtifactWithNoOwner("pkg/test_b.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputB = getBinArtifactWithNoOwner("pkg/test_b.lto-obj/pkg/_objs/lib/shared.pic.o");

    BitcodeFiles sharedBitcodePool =
        new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode));

    LtoBackendAction actionA =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, indexA, importsA),
            sharedBitcodePool,
            importsA,
            ImmutableSet.of(outputA),
            createLtoCommandLines(indexA, outputA, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    LtoBackendAction actionB =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, indexB, importsB),
            sharedBitcodePool,
            importsB,
            ImmutableSet.of(outputB),
            createLtoCommandLines(indexB, outputB, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    String binRoot = outputA.getRoot().getExecPathString();
    assertThat(actionA.getArguments())
        .containsExactly(
            "/bin/clang",
            "-fthinlto-index=" + binRoot + "/_lto/pkg/_objs/lib/shared.pic.o.thinlto.bc",
            "-o",
            binRoot + "/_lto-obj/pkg/_objs/lib/shared.pic.o",
            "-c",
            sharedBitcode.getExecPathString())
        .inOrder();
    assertThat(actionA.getArguments()).isEqualTo(actionB.getArguments());
    assertThat(actionA.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isEqualTo(actionB.getKey(actionKeyContext, /* inputMetadataProvider= */ null));

    Spawn spawnA = actionA.getSpawn(context);
    PathMapper mapperA = spawnA.getPathMapper();
    assertThat(mapperA.isNoop()).isFalse();
    assertThat(mapperA.map(indexA.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    assertThat(mapperA.map(importsA.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/pkg/_objs/lib/shared.pic.o.imports");
    assertThat(mapperA.map(outputA.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto-obj/pkg/_objs/lib/shared.pic.o");
    assertThat(mapperA.map(sharedBitcode.getExecPath())).isEqualTo(sharedBitcode.getExecPath());

    PathMapper mapperA2 = actionA.getPathMapper();
    PathMapper mapperB = actionB.getPathMapper();
    assertThat(mapperA.cacheKey()).isEqualTo(mapperA2.cacheKey());
    assertThat(mapperA.cacheKey()).isNotEqualTo(mapperB.cacheKey());
    assertThat(mapperA.cacheKey()).isNotInstanceOf(Integer.class);
  }

  @Test
  public void ltoBackendPathMapper_disabledWhenDwoOutputPresent() throws Exception {
    useConfiguration();

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact index =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact imports =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");
    Artifact outputDwo =
        getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.dwo");

    LtoBackendAction actionWithDwo =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj, outputDwo),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    assertThat(actionWithDwo.getPathMapper().isNoop()).isTrue();
    assertThat(actionWithDwo.getExecutionInfo()).doesNotContainKey("supports-path-mapping");
    assertThat(actionWithDwo.getArguments())
        .containsExactly(
            "/bin/clang",
            "-fthinlto-index=" + index.getExecPathString(),
            "-o",
            outputObj.getExecPathString(),
            "-c",
            sharedBitcode.getExecPathString())
        .inOrder();

    useConfiguration("--modify_execution_info=CcLtoBackendCompile=+supports-path-mapping");
    LtoBackendAction forcedWithDwo =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj, outputDwo),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);
    assertThat(forcedWithDwo.getPathMapper().isNoop()).isTrue();
    assertThat(forcedWithDwo.getArguments()).contains(outputObj.getExecPathString());
  }

  @Test
  public void ltoBackendPathMapper_disabledViaModifyExecutionInfo() throws Exception {
    useConfiguration("--modify_execution_info=CcLtoBackendCompile=-supports-path-mapping");

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact index =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact imports =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    LtoBackendAction action =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    assertThat(action.getPathMapper().isNoop()).isTrue();
    assertThat(action.getArguments()).contains("-o");
    assertThat(action.getArguments()).contains(outputObj.getExecPathString());
  }

  @Test
  public void ltoBackendPathMapper_nullConfigurationUsesActionOwnerConfig() throws Exception {
    scratch.file(
        "ownerpkg/BUILD",
        """
        filegroup(
            name = "fg",
            srcs = [],
        )
        """);
    useConfiguration();
    var enabledOwner = getRuleContext(getConfiguredTarget("//ownerpkg:fg")).getActionOwner();

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact index =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact imports =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    LtoBackendAction enabledTemplateAction =
        LtoBackendAction.create(
            enabledOwner,
            /* configuration= */ null,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);
    assertThat(enabledTemplateAction.getPathMapper().isNoop()).isFalse();

    useConfiguration("--modify_execution_info=CcLtoBackendCompile=-supports-path-mapping");
    var disabledOwner = getRuleContext(getConfiguredTarget("//ownerpkg:fg")).getActionOwner();

    LtoBackendAction disabledTemplateAction =
        LtoBackendAction.create(
            disabledOwner,
            /* configuration= */ null,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);
    assertThat(disabledTemplateAction.getPathMapper().isNoop()).isTrue();

    LtoBackendAction nullConfigDisabled =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            /* configuration= */ null,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()));
    assertThat(nullConfigDisabled.getPathMapper().isNoop()).isTrue();
    assertThat(nullConfigDisabled.describeKey()).doesNotContain("PathMapper GUID:");

    LtoBackendAction nullConfigEnabled =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            /* configuration= */ null,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, index, imports),
            new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode)),
            imports,
            ImmutableSet.of(outputObj),
            createLtoCommandLines(index, outputObj, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);
    assertThat(nullConfigEnabled.getPathMapper().isNoop()).isFalse();
  }

  @Test
  public void ltoBackendPathMapper_preservesBitcodeUnderPackageNamedDotLto() throws Exception {
    useConfiguration();

    Artifact sharedBitcode =
        getBinArtifactWithNoOwner("foo.lto/bar.lto-obj/_objs/lib/shared.pic.o");
    String bitcodeExecPath = sharedBitcode.getExecPathString();
    Artifact indexA =
        getBinArtifactWithNoOwner(
            "foo.lto/bar.lto-obj/test_a.lto/" + bitcodeExecPath + ".thinlto.bc");
    Artifact importsA =
        getBinArtifactWithNoOwner("foo.lto/bar.lto-obj/test_a.lto/" + bitcodeExecPath + ".imports");
    Artifact outputA =
        getBinArtifactWithNoOwner("foo.lto/bar.lto-obj/test_a.lto-obj/" + bitcodeExecPath);

    Artifact indexB =
        getBinArtifactWithNoOwner("other.lto/test_b.lto/" + bitcodeExecPath + ".thinlto.bc");
    Artifact importsB =
        getBinArtifactWithNoOwner("other.lto/test_b.lto/" + bitcodeExecPath + ".imports");
    Artifact outputB = getBinArtifactWithNoOwner("other.lto/test_b.lto-obj/" + bitcodeExecPath);

    BitcodeFiles sharedBitcodePool =
        new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode));

    LtoBackendAction actionA =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, indexA, importsA),
            sharedBitcodePool,
            importsA,
            ImmutableSet.of(outputA),
            createLtoCommandLines(indexA, outputA, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    LtoBackendAction actionB =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode, indexB, importsB),
            sharedBitcodePool,
            importsB,
            ImmutableSet.of(outputB),
            createLtoCommandLines(indexB, outputB, sharedBitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    PathMapper mapperA = actionA.getPathMapper();
    assertThat(mapperA.isNoop()).isFalse();
    assertThat(mapperA.map(sharedBitcode.getExecPath())).isEqualTo(sharedBitcode.getExecPath());
    assertThat(actionA.getArguments()).contains(bitcodeExecPath);
    assertThat(actionA.getArguments()).isEqualTo(actionB.getArguments());
    assertThat(actionA.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isEqualTo(actionB.getKey(actionKeyContext, /* inputMetadataProvider= */ null));
  }

  @Test
  public void ltoBackendPathMapper_preservesBitcodeUnderTargetLtoSubpackage() throws Exception {
    useConfiguration();

    Artifact ltoSubpkgBitcode = getBinArtifactWithNoOwner("pkg/target.lto/lib/_objs/foo/bar.pic.o");
    String ltoSubpkgBitcodeExec = ltoSubpkgBitcode.getExecPathString();
    Artifact ltoObjSubpkgBitcode =
        getBinArtifactWithNoOwner("pkg/target.lto-obj/lib/_objs/baz/qux.pic.o");
    String ltoObjSubpkgBitcodeExec = ltoObjSubpkgBitcode.getExecPathString();

    Artifact indexLtoSubpkg =
        getBinArtifactWithNoOwner("pkg/target.lto/" + ltoSubpkgBitcodeExec + ".thinlto.bc");
    Artifact importsLtoSubpkg =
        getBinArtifactWithNoOwner("pkg/target.lto/" + ltoSubpkgBitcodeExec + ".imports");
    Artifact outputLtoSubpkg =
        getBinArtifactWithNoOwner("pkg/target.lto-obj/" + ltoSubpkgBitcodeExec);

    LtoBackendAction ltoSubpkgAction =
        createSingleLtoCompileAction(
            ltoSubpkgBitcode, indexLtoSubpkg, importsLtoSubpkg, outputLtoSubpkg);

    String binRoot = outputLtoSubpkg.getRoot().getExecPathString();
    PathMapper ltoSubpkgMapper = ltoSubpkgAction.getPathMapper();
    assertThat(ltoSubpkgMapper.isNoop()).isFalse();
    assertThat(ltoSubpkgMapper.map(ltoSubpkgBitcode.getExecPath()))
        .isEqualTo(ltoSubpkgBitcode.getExecPath());
    assertThat(ltoSubpkgMapper.map(ltoObjSubpkgBitcode.getExecPath()))
        .isEqualTo(ltoObjSubpkgBitcode.getExecPath());
    assertThat(ltoSubpkgMapper.map(indexLtoSubpkg.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/" + ltoSubpkgBitcodeExec + ".thinlto.bc");
    assertThat(ltoSubpkgMapper.map(outputLtoSubpkg.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto-obj/" + ltoSubpkgBitcodeExec);
    assertThat(ltoSubpkgAction.getArguments())
        .containsExactly(
            "/bin/clang",
            "-fthinlto-index=" + binRoot + "/_lto/" + ltoSubpkgBitcodeExec + ".thinlto.bc",
            "-o",
            binRoot + "/_lto-obj/" + ltoSubpkgBitcodeExec,
            "-c",
            ltoSubpkgBitcodeExec)
        .inOrder();
    assertThat(ltoSubpkgAction.getSpawn(context).getArguments())
        .isEqualTo(ltoSubpkgAction.getArguments());

    Artifact indexLtoObjSubpkg =
        getBinArtifactWithNoOwner("pkg/target.lto/" + ltoObjSubpkgBitcodeExec + ".thinlto.bc");
    Artifact importsLtoObjSubpkg =
        getBinArtifactWithNoOwner("pkg/target.lto/" + ltoObjSubpkgBitcodeExec + ".imports");
    Artifact outputLtoObjSubpkg =
        getBinArtifactWithNoOwner("pkg/target.lto-obj/" + ltoObjSubpkgBitcodeExec);
    LtoBackendAction ltoObjSubpkgAction =
        createSingleLtoCompileAction(
            ltoObjSubpkgBitcode, indexLtoObjSubpkg, importsLtoObjSubpkg, outputLtoObjSubpkg);

    PathMapper ltoObjSubpkgMapper = ltoObjSubpkgAction.getPathMapper();
    assertThat(ltoObjSubpkgMapper.map(ltoSubpkgBitcode.getExecPath()))
        .isEqualTo(ltoSubpkgBitcode.getExecPath());
    assertThat(ltoObjSubpkgMapper.map(ltoObjSubpkgBitcode.getExecPath()))
        .isEqualTo(ltoObjSubpkgBitcode.getExecPath());
    assertThat(ltoObjSubpkgAction.getArguments())
        .containsExactly(
            "/bin/clang",
            "-fthinlto-index=" + binRoot + "/_lto/" + ltoObjSubpkgBitcodeExec + ".thinlto.bc",
            "-o",
            binRoot + "/_lto-obj/" + ltoObjSubpkgBitcodeExec,
            "-c",
            ltoObjSubpkgBitcodeExec)
        .inOrder();

    Artifact treeDummyImports = getBinArtifactWithNoOwner("pkg/target.lto/" + ltoSubpkgBitcodeExec);
    Artifact treeDummyOutput =
        getBinArtifactWithNoOwner("pkg/target.lto-obj/" + ltoSubpkgBitcodeExec);
    PathMapper treeDummyMapper =
        LtoBackendAction.LtoBackendPathMapper.tryCreate(treeDummyImports, treeDummyOutput);
    assertThat(treeDummyMapper.map(treeDummyImports.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/" + ltoSubpkgBitcodeExec);
    assertThat(treeDummyMapper.map(treeDummyOutput.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto-obj/" + ltoSubpkgBitcodeExec);
    assertThat(treeDummyMapper.map(indexLtoSubpkg.getExecPath()))
        .isEqualTo(indexLtoSubpkg.getExecPath());
  }

  @Test
  public void ltoBackendPathMapper_noopForSharedNonLtoBackend() throws Exception {
    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact sharedNonLtoObj =
        getBinArtifactWithNoOwner("shared.nonlto-obj/" + sharedBitcode.getExecPathString());

    for (boolean forcePathMapping : new boolean[] {false, true}) {
      if (forcePathMapping) {
        useConfiguration("--modify_execution_info=CcLtoBackendCompile=+supports-path-mapping");
      } else {
        useConfiguration();
      }

      LtoBackendAction sharedNonLtoAction =
          LtoBackendAction.create(
              ActionsTestUtil.NULL_ACTION_OWNER,
              targetConfig,
              NestedSetBuilder.create(Order.STABLE_ORDER, sharedBitcode),
              /* allBitcodeFiles= */ null,
              /* importsFile= */ null,
              ImmutableSet.of(sharedNonLtoObj),
              createLtoCommandLines(/* index= */ null, sharedNonLtoObj, sharedBitcode),
              ActionEnvironment.create(ImmutableMap.of()),
              /* enablePathMapping= */ true);

      if (!forcePathMapping) {
        assertThat(sharedNonLtoAction.getExecutionInfo())
            .doesNotContainKey("supports-path-mapping");
      }
      assertThat(sharedNonLtoAction.getPathMapper().isNoop()).isTrue();
      assertThat(sharedNonLtoAction.describeKey()).doesNotContain("PathMapper GUID:");
      assertThat(sharedNonLtoAction.getArguments()).contains(sharedNonLtoObj.getExecPathString());
    }
  }

  private LtoBackendAction createSingleLtoCompileAction(
      Artifact bitcode, Artifact index, Artifact imports, Artifact output) {
    return LtoBackendAction.create(
        ActionsTestUtil.NULL_ACTION_OWNER,
        targetConfig,
        NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, index, imports),
        new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, bitcode)),
        imports,
        ImmutableSet.of(output),
        createLtoCommandLines(index, output, bitcode),
        ActionEnvironment.create(ImmutableMap.of()),
        /* enablePathMapping= */ true);
  }

  @Test
  public void ltoBackendPathMapper_getSpawnUsesLtoBackendPathMapperEvenUnderStripMode()
      throws Exception {
    useConfiguration("--experimental_output_paths=strip");

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact indexA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact importsA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputA = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    LtoBackendAction action =
        createSingleLtoCompileAction(sharedBitcode, indexA, importsA, outputA);

    Spawn spawn = action.getSpawn(context);

    assertThat(action.describeKey())
        .contains("PathMapper GUID: " + LtoBackendAction.LtoBackendPathMapper.GUID);
    assertThat(spawn.getPathMapper().isNoop()).isFalse();
    assertThat(spawn.getOutputFiles()).containsExactly(outputA);
    assertThat(spawn.getArguments()).isEqualTo(action.getArguments());
    assertThat(spawn.getArguments()).contains(sharedBitcode.getExecPathString());
  }

  @Test
  public void ltoBackendPathMapper_noopMapperDisablesOutputPathsModeAndDoesNotFallBackToStripping()
      throws Exception {
    useConfiguration("--experimental_output_paths=strip");

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact index =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact sourceImports = getSourceArtifact("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    LtoBackendAction action =
        createSingleLtoCompileAction(sharedBitcode, index, sourceImports, outputObj);

    assertThat(action.getPathMapper().isNoop()).isTrue();
    assertThat(action.describeKey()).doesNotContain("PathMapper GUID:");
    assertThat(action.getArguments())
        .containsExactly(
            "/bin/clang",
            "-fthinlto-index=" + index.getExecPathString(),
            "-o",
            outputObj.getExecPathString(),
            "-c",
            sharedBitcode.getExecPathString())
        .inOrder();

    Spawn spawn = action.getSpawn(context);
    assertThat(spawn.getPathMapper().isNoop()).isTrue();
    assertThat(spawn.getArguments()).isEqualTo(action.getArguments());
  }

  @Test
  public void ltoBackendPathMapper_disabledByLocalOrNoSandboxNoRemoteExecutionInfo()
      throws Exception {
    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact index =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact imports =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    useConfiguration("--modify_execution_info=CcLtoBackendCompile=+local");
    LtoBackendAction localAction =
        createSingleLtoCompileAction(sharedBitcode, index, imports, outputObj);
    assertThat(localAction.getPathMapper().isNoop()).isTrue();
    assertThat(localAction.describeKey()).doesNotContain("PathMapper GUID:");
    assertThat(localAction.getArguments()).contains(outputObj.getExecPathString());

    useConfiguration(
        "--modify_execution_info=CcLtoBackendCompile=+no-sandbox",
        "--modify_execution_info=CcLtoBackendCompile=+no-remote");
    LtoBackendAction noSandboxNoRemoteAction =
        createSingleLtoCompileAction(sharedBitcode, index, imports, outputObj);
    assertThat(noSandboxNoRemoteAction.getPathMapper().isNoop()).isTrue();
    assertThat(noSandboxNoRemoteAction.getArguments()).contains(outputObj.getExecPathString());

    useConfiguration("--modify_execution_info=CcLtoBackendCompile=+no-sandbox");
    LtoBackendAction noSandboxOnlyAction =
        createSingleLtoCompileAction(sharedBitcode, index, imports, outputObj);
    assertThat(noSandboxOnlyAction.getPathMapper().isNoop()).isFalse();
  }

  @Test
  public void ltoBackendPathMapper_stripsSingleLtoDirectoryWhenSeparateObjDirDisabled()
      throws Exception {
    useConfiguration();

    Artifact sharedBitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    String bitcodeExecPath = sharedBitcode.getExecPathString();
    Artifact indexA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/" + bitcodeExecPath + ".thinlto.bc");
    Artifact importsA = getBinArtifactWithNoOwner("pkg/test_a.lto/" + bitcodeExecPath + ".imports");
    Artifact outputA = getBinArtifactWithNoOwner("pkg/test_a.lto/" + bitcodeExecPath);

    Artifact indexB =
        getBinArtifactWithNoOwner("pkg/test_b.lto/" + bitcodeExecPath + ".thinlto.bc");
    Artifact importsB = getBinArtifactWithNoOwner("pkg/test_b.lto/" + bitcodeExecPath + ".imports");
    Artifact outputB = getBinArtifactWithNoOwner("pkg/test_b.lto/" + bitcodeExecPath);

    LtoBackendAction actionA =
        createSingleLtoCompileAction(sharedBitcode, indexA, importsA, outputA);
    LtoBackendAction actionB =
        createSingleLtoCompileAction(sharedBitcode, indexB, importsB, outputB);

    String binRoot = outputA.getRoot().getExecPathString();
    PathMapper mapperA = actionA.getPathMapper();
    assertThat(mapperA.isNoop()).isFalse();
    assertThat(mapperA.map(indexA.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/" + bitcodeExecPath + ".thinlto.bc");
    assertThat(mapperA.map(outputA.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/" + bitcodeExecPath);
    assertThat(mapperA.map(sharedBitcode.getExecPath())).isEqualTo(sharedBitcode.getExecPath());
    assertThat(actionA.getArguments()).isEqualTo(actionB.getArguments());
    assertThat(actionA.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isEqualTo(actionB.getKey(actionKeyContext, /* inputMetadataProvider= */ null));

    Artifact dotLtoBitcode = getBinArtifactWithNoOwner("foo.lto/bar.lto/_objs/lib/shared.pic.o");
    String dotLtoBitcodePath = dotLtoBitcode.getExecPathString();
    Artifact indexC =
        getBinArtifactWithNoOwner("pkg.lto/test_c.lto/" + dotLtoBitcodePath + ".thinlto.bc");
    Artifact importsC =
        getBinArtifactWithNoOwner("pkg.lto/test_c.lto/" + dotLtoBitcodePath + ".imports");
    Artifact outputC = getBinArtifactWithNoOwner("pkg.lto/test_c.lto/" + dotLtoBitcodePath);

    Artifact indexD =
        getBinArtifactWithNoOwner("other.lto/test_d.lto/" + dotLtoBitcodePath + ".thinlto.bc");
    Artifact importsD =
        getBinArtifactWithNoOwner("other.lto/test_d.lto/" + dotLtoBitcodePath + ".imports");
    Artifact outputD = getBinArtifactWithNoOwner("other.lto/test_d.lto/" + dotLtoBitcodePath);

    LtoBackendAction actionC =
        createSingleLtoCompileAction(dotLtoBitcode, indexC, importsC, outputC);
    LtoBackendAction actionD =
        createSingleLtoCompileAction(dotLtoBitcode, indexD, importsD, outputD);

    PathMapper mapperC = actionC.getPathMapper();
    assertThat(mapperC.isNoop()).isFalse();
    assertThat(mapperC.map(outputC.getExecPath()).getPathString())
        .isEqualTo(binRoot + "/_lto/" + dotLtoBitcodePath);
    assertThat(actionC.getArguments()).isEqualTo(actionD.getArguments());
    assertThat(actionC.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isEqualTo(actionD.getKey(actionKeyContext, /* inputMetadataProvider= */ null));
  }

  @Test
  public void ltoBackendPathMapper_randomizedPropertyFuzzInvariants() throws Exception {
    useConfiguration();

    String[] pkgTokens = {"alpha", "beta.lto", "gamma.lto-obj", "_lto_dir", "sub"};
    String[] targetTokens = {"bin_a", "test_b", "srv.lto", "helper_1"};
    Random rng = new Random(0x5170BACEL);

    for (int iter = 0; iter < 250; iter++) {
      int libDepth = 1 + rng.nextInt(4);
      StringBuilder libPkg = new StringBuilder();
      for (int d = 0; d < libDepth; d++) {
        if (d > 0) {
          libPkg.append('/');
        }
        libPkg.append(pkgTokens[rng.nextInt(pkgTokens.length)]);
      }
      Artifact bitcode =
          getBinArtifactWithNoOwner(
              libPkg + "/_objs/lib" + (iter % 7) + "/mod" + (iter % 11) + ".pic.o");
      String bitcodeExec = bitcode.getExecPathString();
      BitcodeFiles pool = new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, bitcode));

      String pkgA =
          pkgTokens[rng.nextInt(pkgTokens.length)] + "/" + pkgTokens[rng.nextInt(pkgTokens.length)];
      String pkgB = pkgTokens[rng.nextInt(pkgTokens.length)];
      String tgtA = targetTokens[rng.nextInt(targetTokens.length)] + "_a";
      String tgtB = targetTokens[rng.nextInt(targetTokens.length)] + "_b";

      Artifact indexA =
          getBinArtifactWithNoOwner(pkgA + "/" + tgtA + ".lto/" + bitcodeExec + ".thinlto.bc");
      Artifact importsA =
          getBinArtifactWithNoOwner(pkgA + "/" + tgtA + ".lto/" + bitcodeExec + ".imports");
      Artifact outA = getBinArtifactWithNoOwner(pkgA + "/" + tgtA + ".lto-obj/" + bitcodeExec);

      Artifact indexB =
          getBinArtifactWithNoOwner(pkgB + "/" + tgtB + ".lto/" + bitcodeExec + ".thinlto.bc");
      Artifact importsB =
          getBinArtifactWithNoOwner(pkgB + "/" + tgtB + ".lto/" + bitcodeExec + ".imports");
      Artifact outB = getBinArtifactWithNoOwner(pkgB + "/" + tgtB + ".lto-obj/" + bitcodeExec);

      LtoBackendAction actionA =
          LtoBackendAction.create(
              ActionsTestUtil.NULL_ACTION_OWNER,
              targetConfig,
              NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, indexA, importsA),
              pool,
              importsA,
              ImmutableSet.of(outA),
              createLtoCommandLines(indexA, outA, bitcode),
              ActionEnvironment.create(ImmutableMap.of()),
              /* enablePathMapping= */ true);

      LtoBackendAction actionB =
          LtoBackendAction.create(
              ActionsTestUtil.NULL_ACTION_OWNER,
              targetConfig,
              NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, indexB, importsB),
              pool,
              importsB,
              ImmutableSet.of(outB),
              createLtoCommandLines(indexB, outB, bitcode),
              ActionEnvironment.create(ImmutableMap.of()),
              /* enablePathMapping= */ true);

      PathMapper mapperA = actionA.getPathMapper();
      PathMapper mapperB = actionB.getPathMapper();
      assertThat(mapperA.isNoop()).isFalse();
      assertThat(mapperB.isNoop()).isFalse();
      assertThat(mapperA.map(bitcode.getExecPath())).isEqualTo(bitcode.getExecPath());
      assertThat(mapperB.map(bitcode.getExecPath())).isEqualTo(bitcode.getExecPath());
      assertThat(mapperA.map(indexA.getExecPath())).isEqualTo(mapperB.map(indexB.getExecPath()));
      assertThat(mapperA.map(outA.getExecPath())).isEqualTo(mapperB.map(outB.getExecPath()));
      assertThat(actionA.getArguments()).isEqualTo(actionB.getArguments());
      assertThat(actionA.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
          .isEqualTo(actionB.getKey(actionKeyContext, /* inputMetadataProvider= */ null));
    }
  }

  @Test
  public void ltoBackendPathMapper_computeKeyDistinguishesMandatoryInputAndImportsPaths()
      throws Exception {
    useConfiguration("--experimental_output_paths=strip");

    Artifact bitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact extraInputA = getBinArtifactWithNoOwner("pkg/_objs/lib/extra_a.pic.o");
    Artifact extraInputB = getBinArtifactWithNoOwner("pkg/_objs/lib/extra_b.pic.o");
    Artifact indexA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact importsA =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact importsOther =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/other.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");

    BitcodeFiles sharedPool =
        new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, bitcode));
    CommandLines fixedCmd = createLtoCommandLines(indexA, outputObj, bitcode);

    LtoBackendAction withExtraA =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, importsA, extraInputA),
            sharedPool,
            importsA,
            ImmutableSet.of(outputObj),
            fixedCmd,
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);
    LtoBackendAction withExtraB =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, importsA, extraInputB),
            sharedPool,
            importsA,
            ImmutableSet.of(outputObj),
            fixedCmd,
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);

    assertThat(withExtraA.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isNotEqualTo(withExtraB.getKey(actionKeyContext, /* inputMetadataProvider= */ null));

    LtoBackendAction noopImportsA =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, extraInputA),
            sharedPool,
            importsA,
            ImmutableSet.of(outputObj),
            fixedCmd,
            ActionEnvironment.create(ImmutableMap.of()));
    LtoBackendAction noopImportsOther =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, extraInputA),
            sharedPool,
            importsOther,
            ImmutableSet.of(outputObj),
            fixedCmd,
            ActionEnvironment.create(ImmutableMap.of()));
    assertThat(noopImportsA.getPathMapper().isNoop()).isTrue();
    assertThat(noopImportsA.getExecutionInfo()).doesNotContainKey("supports-path-mapping");
    assertThat(noopImportsA.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isNotEqualTo(noopImportsOther.getKey(actionKeyContext, /* inputMetadataProvider= */ null));
  }

  private static CcToolchainFeatures.FeatureConfiguration featureConfigForPathMapping(
      ImmutableSet<String> requestedFeatures, ImmutableList<CcToolchainFeatures.Feature> enabled) {
    return new CcToolchainFeatures.FeatureConfiguration(
        requestedFeatures,
        enabled,
        ImmutableSet.of(),
        ImmutableMap.of(),
        PathFragment.EMPTY_FRAGMENT) {
      @Override
      public String getToolPathForAction(String actionName) {
        return "/bin/clang";
      }
    };
  }

  @Test
  public void ltoBackendArtifacts_featureGateAndDisableOverride() throws Exception {
    useConfiguration("--experimental_output_paths=strip");

    Artifact bitcode = getBinArtifactWithNoOwner("pkg/_objs/lib/shared.pic.o");
    Artifact index =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.thinlto.bc");
    Artifact imports =
        getBinArtifactWithNoOwner("pkg/test_a.lto/pkg/_objs/lib/shared.pic.o.imports");
    Artifact outputObj = getBinArtifactWithNoOwner("pkg/test_a.lto-obj/pkg/_objs/lib/shared.pic.o");
    NestedSet<Artifact> inputs =
        NestedSetBuilder.create(Order.STABLE_ORDER, bitcode, index, imports);
    BitcodeFiles bitcodeFiles =
        new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, bitcode));

    LtoBackendAction defaultAction =
        LtoBackendArtifacts.createLtoBackendActionForStarlark(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            featureConfigForPathMapping(ImmutableSet.of(), ImmutableList.of()),
            CcToolchainVariables.builder().build(),
            /* usePic= */ true,
            inputs,
            bitcodeFiles,
            imports,
            ImmutableSet.of(outputObj),
            ActionEnvironment.create(ImmutableMap.of()));
    assertThat(defaultAction.getPathMapper().isNoop()).isTrue();
    assertThat(defaultAction.getExecutionInfo()).doesNotContainKey("supports-path-mapping");

    LtoBackendAction enabledAction =
        LtoBackendArtifacts.createLtoBackendActionForStarlark(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            featureConfigForPathMapping(
                ImmutableSet.of(CppRuleClasses.LTO_BACKEND_PATH_MAPPING), ImmutableList.of()),
            CcToolchainVariables.builder().build(),
            /* usePic= */ true,
            inputs,
            bitcodeFiles,
            imports,
            ImmutableSet.of(outputObj),
            ActionEnvironment.create(ImmutableMap.of()));
    assertThat(enabledAction.getPathMapper().isNoop()).isFalse();
    assertThat(enabledAction.getExecutionInfo()).containsKey("supports-path-mapping");

    CcToolchainFeatures.Feature disableFeature =
        new CcToolchainFeatures.Feature(
            CppRuleClasses.DISABLE_LTO_BACKEND_PATH_MAPPING,
            ImmutableList.of(),
            ImmutableList.of(),
            /* enabled= */ true,
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    LtoBackendAction toolchainDisabledAction =
        LtoBackendArtifacts.createLtoBackendActionForStarlark(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            featureConfigForPathMapping(
                ImmutableSet.of(CppRuleClasses.LTO_BACKEND_PATH_MAPPING),
                ImmutableList.of(disableFeature)),
            CcToolchainVariables.builder().build(),
            /* usePic= */ true,
            inputs,
            bitcodeFiles,
            imports,
            ImmutableSet.of(outputObj),
            ActionEnvironment.create(ImmutableMap.of()));
    assertThat(toolchainDisabledAction.getPathMapper().isNoop()).isTrue();
    assertThat(toolchainDisabledAction.getExecutionInfo())
        .doesNotContainKey("supports-path-mapping");
  }

  @Test
  public void describeKey_withPathMapping_usesMappedPathsAndGuid() throws Exception {
    Artifact indexA =
        getBinArtifactWithNoOwner(
            "pkg/bin_a.lto/blaze-out/k8-fastbuild/bin/lib/foo.pic.o.thinlto.bc");
    Artifact importsA =
        getBinArtifactWithNoOwner("pkg/bin_a.lto/blaze-out/k8-fastbuild/bin/lib/foo.pic.o.imports");
    Artifact outputA =
        getBinArtifactWithNoOwner("pkg/bin_a.lto-obj/blaze-out/k8-fastbuild/bin/lib/foo.pic.o");
    Artifact bitcode = getBinArtifactWithNoOwner("lib/foo.pic.o");
    BitcodeFiles bitcodeFiles =
        new BitcodeFiles(NestedSetBuilder.create(Order.STABLE_ORDER, bitcode));

    LtoBackendAction mappedAction =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, indexA, importsA, bitcode),
            bitcodeFiles,
            importsA,
            ImmutableSet.of(outputA),
            createLtoCommandLines(indexA, outputA, bitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ true);
    LtoBackendAction unmappedAction =
        LtoBackendAction.create(
            ActionsTestUtil.NULL_ACTION_OWNER,
            targetConfig,
            NestedSetBuilder.create(Order.STABLE_ORDER, indexA, importsA, bitcode),
            bitcodeFiles,
            importsA,
            ImmutableSet.of(outputA),
            createLtoCommandLines(indexA, outputA, bitcode),
            ActionEnvironment.create(ImmutableMap.of()),
            /* enablePathMapping= */ false);

    String mappedDesc = mappedAction.describeKey();
    String unmappedDesc = unmappedAction.describeKey();
    assertThat(mappedDesc)
        .contains("PathMapper GUID: " + LtoBackendAction.LtoBackendPathMapper.GUID);
    assertThat(mappedDesc).contains("/_lto/blaze-out/k8-fastbuild/bin/lib/foo.pic.o.thinlto.bc");
    assertThat(mappedDesc).contains("/_lto/blaze-out/k8-fastbuild/bin/lib/foo.pic.o.imports");
    assertThat(mappedDesc).contains("/_lto-obj/blaze-out/k8-fastbuild/bin/lib/foo.pic.o");
    assertThat(mappedDesc).doesNotContain("pkg/bin_a.lto/");
    assertThat(unmappedDesc).doesNotContain("PathMapper GUID:");
    assertThat(unmappedDesc).contains("Imports: " + importsA.getExecPathString());
  }
}
