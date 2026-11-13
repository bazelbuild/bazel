// Copyright 2023 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;
import static com.google.devtools.build.lib.testutil.TestConstants.PRODUCT_NAME;
import static com.google.devtools.build.lib.testutil.TestConstants.WORKSPACE_NAME;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.SortedMap;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link CompactSpawnLogContext}. */
@RunWith(TestParameterInjector.class)
public final class CompactSpawnLogContextTest extends SpawnLogContextTestBase {
  private final Path logPath = fs.getPath("/log");

  @Test
  public void testTransitiveNestedSet(@TestParameter InputsMode inputsMode) throws Exception {
    Artifact file1 = ActionsTestUtil.createArtifact(rootDir, "file1");
    Artifact file2 = ActionsTestUtil.createArtifact(rootDir, "file2");
    Artifact file3 = ActionsTestUtil.createArtifact(rootDir, "file3");

    writeFile(file1, "abc");
    writeFile(file2, "def");
    writeFile(file3, "ghi");

    NestedSet<ActionInput> inputs =
        NestedSetBuilder.<ActionInput>stableOrder()
            .add(file1)
            .addTransitive(
                NestedSetBuilder.<ActionInput>stableOrder().add(file2).add(file3).build())
            .build();

    assertThat(inputs.getLeaves()).hasSize(1);
    assertThat(inputs.getNonLeaves()).hasSize(1);

    SpawnBuilder spawn = defaultSpawnBuilder().withInputs(inputs);
    if (inputsMode.isTool()) {
      spawn = spawn.withTools(inputs);
    }

    SpawnLogContext context = createSpawnLogContext();

    context.logSpawn(
        spawn.build(),
        createInputMetadataProvider(file1, file2, file3),
        createInputMap(file1, file2, file3),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    closeAndAssertLog(
        context,
        defaultSpawnExecBuilder()
            .addInputs(
                File.newBuilder()
                    .setPath("file1")
                    .setDigest(getDigest("abc"))
                    .setIsTool(inputsMode.isTool()))
            .addInputs(
                File.newBuilder()
                    .setPath("file2")
                    .setDigest(getDigest("def"))
                    .setIsTool(inputsMode.isTool()))
            .addInputs(
                File.newBuilder()
                    .setPath("file3")
                    .setDigest(getDigest("ghi"))
                    .setIsTool(inputsMode.isTool()))
            .build());
  }

  @Test
  public void testSymlinkAction() throws IOException, InterruptedException {
    Artifact source = ActionsTestUtil.createArtifact(rootDir, "source");
    Artifact target = ActionsTestUtil.createArtifact(rootDir, "target");
    ActionOwner owner =
        ActionOwner.createDummy(
            Label.parseCanonicalUnchecked("//pkg:symlink"),
            new Location("dummy-file", 0, 0),
            "some_rule",
            "configurationMnemonic",
            /* configurationChecksum= */ "configurationChecksum",
            new BuildConfigurationEvent(
                BuildEventStreamProtos.BuildEventId.getDefaultInstance(),
                BuildEventStreamProtos.BuildEvent.getDefaultInstance()),
            /* isToolConfiguration= */ false,
            /* executionPlatform= */ null,
            /* aspectDescriptors= */ ImmutableList.of(),
            /* execProperties= */ ImmutableMap.of());
    SymlinkAction symlinkAction =
        SymlinkAction.toArtifact(owner, source, target, "Creating symlink");

    SpawnLogContext context = createSpawnLogContext();
    context.logSymlinkAction(symlinkAction);

    var entries = closeAndReadCompactLog(context);
    assertThat(entries)
        .containsExactly(
            Protos.ExecLogEntry.newBuilder()
                .setInvocation(
                    Protos.ExecLogEntry.Invocation.newBuilder()
                        .setHashFunctionName("SHA-256")
                        .setWorkspaceRunfilesDirectory(TestConstants.WORKSPACE_NAME)
                        .setId("00000000-0000-0000-0000-000000000000"))
                .build(),
            Protos.ExecLogEntry.newBuilder()
                .setSymlinkAction(
                    Protos.ExecLogEntry.SymlinkAction.newBuilder()
                        .setInputPath("source")
                        .setOutputPath("target")
                        .setMnemonic("Symlink")
                        .setTargetLabel("//pkg:symlink"))
                .build());
  }

  @Test
  public void testRunfilesTreeReusedForTool() throws Exception {
    Artifact tool = ActionsTestUtil.createArtifact(rootDir, "data.txt");
    writeFile(tool, "abc");
    Artifact toolRunfiles = ActionsTestUtil.createRunfilesArtifact(outputDir, "tool.runfiles");

    PathFragment runfilesRoot = outputDir.getExecPath().getRelative("foo.runfiles");
    RunfilesTree runfilesTree = createRunfilesTree(runfilesRoot, tool);

    Artifact firstInput = ActionsTestUtil.createArtifact(rootDir, "first_input");
    writeFile(firstInput, "def");
    Artifact secondInput = ActionsTestUtil.createArtifact(rootDir, "second_input");
    writeFile(secondInput, "ghi");

    Spawn firstSpawn =
        defaultSpawnBuilder().withTool(toolRunfiles).withInputs(firstInput, toolRunfiles).build();
    Spawn secondSpawn =
        defaultSpawnBuilder().withTool(toolRunfiles).withInputs(secondInput, toolRunfiles).build();

    SpawnLogContext context = createSpawnLogContext();

    context.logSpawn(
        firstSpawn,
        createInputMetadataProvider(runfilesTree, toolRunfiles, firstInput),
        createInputMap(runfilesTree, firstInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());
    context.logSpawn(
        secondSpawn,
        createInputMetadataProvider(runfilesTree, toolRunfiles, secondInput),
        createInputMap(runfilesTree, secondInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    var entries = closeAndReadCompactLog(context);
    assertThat(entries.stream().filter(Protos.ExecLogEntry::hasRunfilesTree)).hasSize(1);

    closeAndAssertLog(
        context,
        defaultSpawnExecBuilder()
            .addInputs(
                File.newBuilder()
                    .setPath(
                        PRODUCT_NAME
                            + "-out/k8-fastbuild/bin/foo.runfiles/"
                            + WORKSPACE_NAME
                            + "/data.txt")
                    .setDigest(getDigest("abc"))
                    .setIsTool(true))
            .addInputs(
                File.newBuilder()
                    .setPath("first_input")
                    .setDigest(getDigest("def"))
                    .setIsTool(false))
            .build(),
        defaultSpawnExecBuilder()
            .addInputs(
                File.newBuilder()
                    .setPath(
                        PRODUCT_NAME
                            + "-out/k8-fastbuild/bin/foo.runfiles/"
                            + WORKSPACE_NAME
                            + "/data.txt")
                    .setDigest(getDigest("abc"))
                    .setIsTool(true))
            .addInputs(
                File.newBuilder()
                    .setPath("second_input")
                    .setDigest(getDigest("ghi"))
                    .setIsTool(false))
            .build());
  }

  @Test
  public void testUnmaterializedEmptyTreeInput() throws Exception {
    TreeArtifact treeInput =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputDir, "tree");

    // Deliberately don't create the directory: an empty tree artifact may not be materialized on
    // disk, so its (empty) contents are only known to in-memory metadata.
    assertThat(treeInput.getPath().exists()).isFalse();

    TreeArtifactValue treeMetadata = TreeArtifactValue.newBuilder(treeInput).build();
    FakeActionInputFileCache inputMetadataProvider = new FakeActionInputFileCache();
    inputMetadataProvider.put(treeInput, treeMetadata.getMetadata());
    inputMetadataProvider.putTreeArtifact(treeInput, treeMetadata);

    SpawnLogContext context = createSpawnLogContext();

    context.logSpawn(
        defaultSpawnBuilder().withInputs(treeInput).build(),
        inputMetadataProvider,
        // SpawnInputExpander keeps empty tree artifacts in the input map.
        ImmutableSortedMap.of(treeInput.getExecPath(), treeInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    closeAndAssertLog(context, defaultSpawnExecBuilder().build());
  }

  @Test
  public void testUnreadableOutputs(@TestParameter OutputsMode outputsMode) throws Exception {
    Artifact readableFile = ActionsTestUtil.createArtifact(outputDir, "readable");
    Artifact unreadableFile = ActionsTestUtil.createArtifact(outputDir, "unreadable");
    Artifact unreadableFileDir =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputDir, "unreadableFileDir");

    writeFile(readableFile.getPath(), "xyz");
    // Make the files unreadable.
    writeFile(unreadableFile.getPath(), "abc");
    unreadableFile.getPath().setReadable(false);
    writeFile(unreadableFileDir.getPath().getChild("file"), "def");
    unreadableFileDir.getPath().getChild("file").setReadable(false);

    SpawnBuilder spawn =
        defaultSpawnBuilder().withOutputs(readableFile, unreadableFile, unreadableFileDir);

    SpawnLogContext context = createSpawnLogContext();

    context.logSpawn(
        spawn.build(),
        createInputMetadataProvider(),
        createInputMap(),
        outputsMode.getActionFileSystem(fs),
        defaultTimeout(),
        defaultSpawnResult());

    closeAndAssertLog(
        context,
        defaultSpawnExecBuilder()
            .addActualOutputs(
                File.newBuilder()
                    .setPath(PRODUCT_NAME + "-out/k8-fastbuild/bin/readable")
                    .setDigest(getDigest("xyz"))
                    .setIsTool(false))
            .addListedOutputs(PRODUCT_NAME + "-out/k8-fastbuild/bin/readable")
            .addListedOutputs(PRODUCT_NAME + "-out/k8-fastbuild/bin/unreadable")
            .addListedOutputs(PRODUCT_NAME + "-out/k8-fastbuild/bin/unreadableFileDir")
            .build());

    assertContainsEvent(
        storedEventHandler.getEvents(),
        "The compact execution log is incomplete because some outputs could not be read. Refer"
            + " to the server log file for details.");
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getPosts()).isEmpty();
  }

  @Test
  public void testMnemonicFilter() throws Exception {
    SpawnBuilder spawn1 = defaultSpawnBuilder().withMnemonic("Mnemonic1");
    SpawnBuilder spawn2 = defaultSpawnBuilder().withMnemonic("Mnemonic2");

    SpawnLogContext context =
        createSpawnLogContext(spawn -> spawn.getMnemonic().equals("Mnemonic1"));

    context.logSpawn(
        spawn1.build(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResult());
    context.logSpawn(
        spawn2.build(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    closeAndAssertLog(context, defaultSpawnExecBuilder().setMnemonic("Mnemonic1").build());
  }

  @Test
  public void testExpandParamFiles(@TestParameter boolean expandParamFiles) throws Exception {
    ParamFileActionInput paramFileInput =
        new ParamFileActionInput(
            PathFragment.create("foo.params"),
            "@foo.params",
            ImmutableList.of("a", "b", "c"),
            ParameterFileType.UNQUOTED);

    SpawnBuilder spawn =
        new SpawnBuilder("cmd", "--opt", "@foo.params", "--last").withInputs(paramFileInput);

    SpawnLogContext context =
        createSpawnLogContext(
            ImmutableMap.of(), /* logSpawnPredicate= */ s -> true, expandParamFiles);

    context.logSpawn(
        spawn.build(),
        // ParamFileActionInputs appear in the input map but not in the metadata provider.
        createInputMetadataProvider(),
        createInputMap(paramFileInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    ImmutableList<String> expectedArgs =
        expandParamFiles
            ? ImmutableList.of("cmd", "--opt", "a", "b", "c", "--last")
            : ImmutableList.of("cmd", "--opt", "@foo.params", "--last");
    closeAndAssertLog(
        context,
        defaultSpawnExecBuilder()
            .clearCommandArgs()
            .addAllCommandArgs(expectedArgs)
            // The param file is still logged as an input either way.
            .addInputs(File.newBuilder().setPath("foo.params").setDigest(getDigest("a\nb\nc\n")))
            .build());
  }

  @Test
  public void testStreaming() throws Exception {
    Artifact file = ActionsTestUtil.createArtifact(rootDir, "file");
    writeFile(file, "abc");

    SpawnBuilder spawn = defaultSpawnBuilder().withInput(file);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    BufferedOutputStream out = new BufferedOutputStream(baos);

    SpawnLogContext context =
        new CompactSpawnLogContext(
            out,
            "stream",
            execRoot.asFragment(),
            TestConstants.WORKSPACE_NAME,
            Options.getDefaults(RemoteOptions.class),
            DigestHashFunction.SHA256,
            SyscallCache.NO_CACHE,
            new CompressionServiceImpl(),
            UUID.fromString("00000000-0000-0000-0000-000000000000"),
            storedEventHandler,
            /* logSpawnPredicate= */ s -> true,
            /* expandParamFiles= */ false);

    context.logSpawn(
        spawn.build(),
        createInputMetadataProvider(file),
        createInputMap(file),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    context.close();

    ArrayList<SpawnExec> actual = new ArrayList<>();
    try (InputStream in = new ByteArrayInputStream(baos.toByteArray());
        SpawnLogReconstructor reconstructor = new SpawnLogReconstructor(in)) {
      SpawnExec ex;
      while ((ex = reconstructor.read()) != null) {
        actual.add(ex);
      }
    }

    assertThat(actual)
        .containsExactly(
            defaultSpawnExecBuilder()
                .addInputs(File.newBuilder().setPath("file").setDigest(getDigest("abc")))
                .build());
  }

  @Test
  public void testConcurrentLogging() throws Exception {
    // Logs spawns that share a deeply nested graph of input sets from many threads at once, and
    // checks that the resulting log is well-formed and describes the same spawns as a log produced
    // by a single thread.
    int numLayers = 4;
    int setsPerLayer = 6;
    int numSpawns = 48;
    int numThreads = 16;
    int numRounds = 20;

    ArrayList<Artifact> artifacts = new ArrayList<>();

    // Inputs of every kind that is logged as a separate entry.
    Artifact sourceDir = ActionsTestUtil.createArtifact(rootDir, "source_dir");
    writeFile(sourceDir.getPath().getRelative("file1"), "abc");
    writeFile(sourceDir.getPath().getRelative("subdir/file2"), "def");
    artifacts.add(sourceDir);
    TreeArtifact tree = ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputDir, "tree");
    writeFile(tree.getPath().getRelative("child1"), "ghi");
    writeFile(tree.getPath().getRelative("subdir/child2"), "jkl");
    artifacts.add(tree);
    Artifact symlink = ActionsTestUtil.createUnresolvedSymlinkArtifact(outputDir, "symlink");
    symlink.getPath().getParentDirectory().createDirectoryAndParents();
    symlink.getPath().createSymbolicLink(PathFragment.create("/some/path"));
    artifacts.add(symlink);

    // A tool with a runfiles tree, which is shared between spawns.
    Artifact runfilesFile = createFileArtifact("runfiles_file", artifacts);
    Artifact runfilesSymlinkTarget = createFileArtifact("runfiles_symlink_target", artifacts);
    Artifact runfilesRootSymlinkTarget =
        createFileArtifact("runfiles_root_symlink_target", artifacts);
    Artifact toolRunfiles = ActionsTestUtil.createRunfilesArtifact(outputDir, "tool.runfiles");
    artifacts.add(toolRunfiles);
    RunfilesTree runfilesTree =
        createRunfilesTree(
            outputDir.getExecPath().getRelative("tool.runfiles"),
            ImmutableMap.of("symlink", runfilesSymlinkTarget),
            ImmutableMap.of("root_symlink", runfilesRootSymlinkTarget),
            runfilesFile);

    // Every shared nested set directly contains a marker file that no other set contains, which
    // makes it possible to recognize its entry in the log.
    ArrayList<String> markerPaths = new ArrayList<>();
    Artifact toolsMarker = createFileArtifact("tools_marker", artifacts);
    markerPaths.add(toolsMarker.getExecPathString());
    NestedSet<ActionInput> tools =
        NestedSetBuilder.<ActionInput>stableOrder().add(toolsMarker).add(toolRunfiles).build();

    // Layers of nested sets, each of which contains multiple sets of the previous layer, so that
    // the sets are shared both between spawns and between other sets.
    ArrayList<List<NestedSet<ActionInput>>> layers = new ArrayList<>();
    for (int layer = 0; layer < numLayers; layer++) {
      ArrayList<NestedSet<ActionInput>> sets = new ArrayList<>();
      for (int i = 0; i < setsPerLayer; i++) {
        String name = "layer%d_set%d".formatted(layer, i);
        Artifact marker = createFileArtifact(name + "_marker", artifacts);
        markerPaths.add(marker.getExecPathString());
        NestedSetBuilder<ActionInput> builder =
            NestedSetBuilder.<ActionInput>stableOrder()
                .add(marker)
                .add(createFileArtifact(name + "_file", artifacts));
        if (layer == 0) {
          switch (i) {
            case 0 -> builder.add(sourceDir);
            case 1 -> builder.add(tree);
            case 2 -> builder.add(symlink);
            default -> {}
          }
        } else {
          for (int offset : new int[] {0, 1, 3}) {
            builder.addTransitive(layers.get(layer - 1).get((i + offset) % setsPerLayer));
          }
        }
        sets.add(builder.build());
      }
      layers.add(sets);
    }

    ArrayList<Spawn> spawns = new ArrayList<>();
    for (int i = 0; i < numSpawns; i++) {
      NestedSetBuilder<ActionInput> inputs =
          NestedSetBuilder.<ActionInput>stableOrder()
              .add(createFileArtifact("spawn%d_input".formatted(i), artifacts))
              .addTransitive(layers.get(numLayers - 1).get(i % setsPerLayer))
              .addTransitive(layers.get(i % numLayers).get((5 * i + 1) % setsPerLayer))
              .addTransitive(tools);
      if (i % 4 == 0) {
        // A param file is never shared between spawns.
        inputs.add(
            new ParamFileActionInput(
                PathFragment.create("spawn%d.params".formatted(i)),
                "@spawn%d.params".formatted(i),
                ImmutableList.of("a", "b", "c"),
                ParameterFileType.UNQUOTED));
      }
      Artifact output = ActionsTestUtil.createArtifact(outputDir, "spawn%d_output".formatted(i));
      writeFile(output, "output of spawn " + i);
      spawns.add(
          new SpawnBuilder("cmd", "--spawn=" + i)
              .withInputs(inputs.build())
              .withTools(tools)
              .withOutputs(output)
              .build());
    }

    InputMetadataProvider inputMetadataProvider =
        createInputMetadataProvider(runfilesTree, artifacts.toArray(new Artifact[0]));
    // The compact log is produced from the inputs of the spawn rather than from the input map.
    SortedMap<PathFragment, ActionInput> inputMap = createInputMap();

    // Log all spawns from a single thread to obtain the expected result.
    SpawnLogContext sequentialContext = createSpawnLogContext();
    for (Spawn spawn : spawns) {
      sequentialContext.logSpawn(
          spawn, inputMetadataProvider, inputMap, fs, defaultTimeout(), defaultSpawnResult());
    }
    ImmutableList<ExecLogEntry> sequentialEntries = closeAndReadCompactLog(sequentialContext);
    assertWellFormed(sequentialEntries, numSpawns, markerPaths);
    // Every kind of entry that may reference or be referenced by another entry is logged.
    assertThat(sequentialEntries.stream().map(ExecLogEntry::getTypeCase).collect(toImmutableSet()))
        .containsExactly(
            ExecLogEntry.TypeCase.INVOCATION,
            ExecLogEntry.TypeCase.FILE,
            ExecLogEntry.TypeCase.DIRECTORY,
            ExecLogEntry.TypeCase.UNRESOLVED_SYMLINK,
            ExecLogEntry.TypeCase.RUNFILES_TREE,
            ExecLogEntry.TypeCase.INPUT_SET,
            ExecLogEntry.TypeCase.SYMLINK_ENTRY_SET,
            ExecLogEntry.TypeCase.SPAWN);
    ImmutableList<SpawnExec> expected = reconstructLog();
    assertThat(expected).hasSize(numSpawns);
    assertThat(expected.get(0).getInputsCount()).isGreaterThan(2 * numLayers);

    ExecutorService executor = Executors.newFixedThreadPool(numThreads);
    try {
      for (int round = 0; round < numRounds; round++) {
        ArrayList<Spawn> shuffledSpawns = new ArrayList<>(spawns);
        Collections.shuffle(shuffledSpawns, new Random(round));

        SpawnLogContext context = createSpawnLogContext();
        AtomicInteger nextSpawn = new AtomicInteger();
        CountDownLatch start = new CountDownLatch(1);
        ArrayList<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < numThreads; i++) {
          futures.add(
              executor.submit(
                  () -> {
                    start.await();
                    int index;
                    while ((index = nextSpawn.getAndIncrement()) < numSpawns) {
                      context.logSpawn(
                          shuffledSpawns.get(index),
                          inputMetadataProvider,
                          inputMap,
                          fs,
                          defaultTimeout(),
                          defaultSpawnResult());
                    }
                    return null;
                  }));
        }
        start.countDown();
        for (Future<?> future : futures) {
          future.get();
        }

        ImmutableList<ExecLogEntry> entries = closeAndReadCompactLog(context);
        assertWellFormed(entries, numSpawns, markerPaths);
        assertThat(entries).hasSize(sequentialEntries.size());
        assertThat(reconstructLog()).containsExactlyElementsIn(expected);
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testSlowEntryIsComputedOnceWithoutBlockingUnrelatedSpawns() throws Exception {
    Artifact slowFile = ActionsTestUtil.createArtifact(rootDir, "slow_file");
    writeFile(slowFile, "abc");
    Artifact slowSetFile = ActionsTestUtil.createArtifact(rootDir, "slow_set_file");
    writeFile(slowSetFile, "def");
    NestedSet<ActionInput> slowSet =
        NestedSetBuilder.<ActionInput>stableOrder().add(slowFile).add(slowSetFile).build();

    // The slow file has no metadata, so its digest is obtained from the filesystem, which doesn't
    // return from the first attempt until told to.
    BlockingFileSystem blockingFs = new BlockingFileSystem(fs, slowFile.getPath());

    Spawn firstSlowSpawn = spawnWithInputs("first_slow_input", slowSet);
    Spawn secondSlowSpawn = spawnWithInputs("second_slow_input", slowSet);
    Spawn unrelatedSpawn =
        spawnWithInputs(
            "unrelated_input",
            NestedSetBuilder.<ActionInput>stableOrder()
                .add(createFileArtifact("unrelated_file1", new ArrayList<>()))
                .add(createFileArtifact("unrelated_file2", new ArrayList<>()))
                .build());

    SpawnLogContext context = createSpawnLogContext();

    ExecutorService executor = Executors.newCachedThreadPool();
    try {
      Future<?> firstSlow = executor.submit(() -> logSpawn(context, firstSlowSpawn, blockingFs));
      blockingFs.awaitBlocked();

      // A spawn that doesn't have the slow file as an input is logged while the entry for the slow
      // file is being computed. This times out if entries are computed under a lock.
      executor.submit(() -> logSpawn(context, unrelatedSpawn, blockingFs)).get(60, SECONDS);

      // A spawn that also has the slow file as an input waits for the entry to be logged instead
      // of computing it again, so it can't complete yet.
      Future<?> secondSlow = executor.submit(() -> logSpawn(context, secondSlowSpawn, blockingFs));
      assertThrows(TimeoutException.class, () -> secondSlow.get(200, MILLISECONDS));

      blockingFs.unblock();
      firstSlow.get();
      secondSlow.get();
    } finally {
      executor.shutdownNow();
    }

    assertThat(blockingFs.getThreadCount()).isEqualTo(1);

    ImmutableList<ExecLogEntry> entries = closeAndReadCompactLog(context);
    assertWellFormed(entries, /* expectedSpawns= */ 3, ImmutableList.of("slow_file"));
    assertThat(entries.stream().filter(ExecLogEntry::hasFile)).hasSize(7);
    assertThat(entries.stream().filter(ExecLogEntry::hasInputSet)).hasSize(5);
    assertThat(reconstructLog())
        .containsExactly(
            defaultSpawnExecBuilder()
                .addInputs(fileWithContents("first_slow_input"))
                .addInputs(File.newBuilder().setPath("slow_file").setDigest(getDigest("abc")))
                .addInputs(File.newBuilder().setPath("slow_set_file").setDigest(getDigest("def")))
                .build(),
            defaultSpawnExecBuilder()
                .addInputs(fileWithContents("second_slow_input"))
                .addInputs(File.newBuilder().setPath("slow_file").setDigest(getDigest("abc")))
                .addInputs(File.newBuilder().setPath("slow_set_file").setDigest(getDigest("def")))
                .build(),
            defaultSpawnExecBuilder()
                .addInputs(fileWithContents("unrelated_file1"))
                .addInputs(fileWithContents("unrelated_file2"))
                .addInputs(fileWithContents("unrelated_input"))
                .build());
  }

  @Test
  public void testEntryIsComputedAgainAfterFailureInAnotherThread() throws Exception {
    Artifact failingFile = ActionsTestUtil.createArtifact(rootDir, "failing_file");
    writeFile(failingFile, "abc");
    NestedSet<ActionInput> failingSet =
        NestedSetBuilder.<ActionInput>stableOrder()
            .add(failingFile)
            .add(createFileArtifact("failing_set_file", new ArrayList<>()))
            .build();

    // The failing file has no metadata, so its digest is obtained from the filesystem, which fails
    // the first attempt once told to.
    BlockingFileSystem blockingFs =
        new BlockingFileSystem(fs, failingFile.getPath(), new IOException("injected failure"));

    Spawn failingSpawn = spawnWithInputs("failing_input", failingSet);
    Spawn succeedingSpawn = spawnWithInputs("succeeding_input", failingSet);

    SpawnLogContext context = createSpawnLogContext();

    ExecutorService executor = Executors.newCachedThreadPool();
    try {
      Future<?> failing = executor.submit(() -> logSpawn(context, failingSpawn, blockingFs));
      blockingFs.awaitBlocked();

      // The second spawn waits for the entry that is being computed for the first spawn.
      Future<?> succeeding =
          executor.submit(() -> logSpawn(context, succeedingSpawn, blockingFs));
      assertThrows(TimeoutException.class, () -> succeeding.get(200, MILLISECONDS));

      blockingFs.unblock();
      ExecutionException e = assertThrows(ExecutionException.class, failing::get);
      assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
      assertThat(e).hasCauseThat().hasMessageThat().isEqualTo("injected failure");
      // The failure isn't reported for the second spawn, for which the entry is computed again.
      succeeding.get();
    } finally {
      executor.shutdownNow();
    }

    assertThat(blockingFs.getThreadCount()).isEqualTo(2);

    assertWellFormed(
        closeAndReadCompactLog(context), /* expectedSpawns= */ 1, ImmutableList.of("failing_file"));
    assertThat(reconstructLog())
        .containsExactly(
            defaultSpawnExecBuilder()
                .addInputs(File.newBuilder().setPath("failing_file").setDigest(getDigest("abc")))
                .addInputs(fileWithContents("failing_set_file"))
                .addInputs(fileWithContents("succeeding_input"))
                .build());
  }

  /**
   * A filesystem on which the first attempt to stat a given path blocks until {@link #unblock} is
   * called, and then optionally fails.
   */
  private static final class BlockingFileSystem extends DelegateFileSystem {
    private final PathFragment blockingPath;
    @Nullable private final IOException failure;
    private final CountDownLatch blocked = new CountDownLatch(1);
    private final CountDownLatch unblocked = new CountDownLatch(1);
    private final Set<Thread> threads = ConcurrentHashMap.newKeySet();

    BlockingFileSystem(FileSystem delegateFs, Path blockingPath) {
      this(delegateFs, blockingPath, /* failure= */ null);
    }

    BlockingFileSystem(FileSystem delegateFs, Path blockingPath, @Nullable IOException failure) {
      super(delegateFs);
      this.blockingPath = blockingPath.asFragment();
      this.failure = failure;
    }

    @Override
    public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
      if (path.equals(blockingPath) && threads.add(Thread.currentThread()) && threads.size() == 1) {
        blocked.countDown();
        try {
          unblocked.await();
        } catch (InterruptedException e) {
          throw new InterruptedIOException();
        }
        if (failure != null) {
          throw failure;
        }
      }
      return super.stat(path, followSymlinks);
    }

    /** Waits until a thread is blocked in an attempt to stat the path. */
    void awaitBlocked() throws InterruptedException {
      blocked.await();
    }

    void unblock() {
      unblocked.countDown();
    }

    /** Returns the number of threads that have attempted to stat the path. */
    int getThreadCount() {
      return threads.size();
    }
  }

  /**
   * Returns a spawn whose inputs are a newly created file with the given name and the given nested
   * set, which is nested in the inputs of the spawn.
   */
  private Spawn spawnWithInputs(String uniqueInputName, NestedSet<ActionInput> sharedInputs)
      throws IOException {
    return defaultSpawnBuilder()
        .withInputs(
            NestedSetBuilder.<ActionInput>stableOrder()
                .add(createFileArtifact(uniqueInputName, new ArrayList<>()))
                .addTransitive(sharedInputs)
                .build())
        .build();
  }

  /**
   * Logs a spawn whose inputs have no metadata, so that their digests are obtained from the given
   * filesystem.
   */
  @Nullable
  private static Void logSpawn(SpawnLogContext context, Spawn spawn, FileSystem fileSystem)
      throws Exception {
    context.logSpawn(
        spawn,
        createInputMetadataProvider(),
        createInputMap(),
        fileSystem,
        defaultTimeout(),
        defaultSpawnResult());
    return null;
  }

  private File fileWithContents(String name) {
    return File.newBuilder().setPath(name).setDigest(getDigest("contents of " + name)).build();
  }

  private Artifact createFileArtifact(String name, List<Artifact> artifacts) throws IOException {
    Artifact artifact = ActionsTestUtil.createArtifact(rootDir, name);
    writeFile(artifact, "contents of " + name);
    artifacts.add(artifact);
    return artifact;
  }

  /**
   * Asserts that a compact log satisfies the constraints of the format as well as those assumed by
   * {@link SpawnLogReconstructor}.
   *
   * <p>In particular, asserts that an entry is only referenced by later entries, that every entry
   * with an ID is referenced, and that an entry that is deduplicated by the path of its input has
   * been logged only once.
   *
   * @param entries the entries in the order in which they appear in the log
   * @param expectedSpawns the expected number of spawn entries
   * @param markerPaths paths of files that are each expected to be directly contained in exactly
   *     one input set
   */
  private static void assertWellFormed(
      List<ExecLogEntry> entries, int expectedSpawns, List<String> markerPaths) {
    References references = new References();
    HashMap<String, Integer> idByPath = new HashMap<>();
    HashMap<Integer, Integer> inputSetsByInputId = new HashMap<>();
    int lastId = 0;
    int spawns = 0;

    assertThat(entries.get(0).hasInvocation()).isTrue();
    assertThat(entries.get(0).getId()).isEqualTo(0);

    for (ExecLogEntry entry : entries.subList(1, entries.size())) {
      String path = null;
      switch (entry.getTypeCase()) {
        case FILE -> path = entry.getFile().getPath();
        case DIRECTORY -> path = entry.getDirectory().getPath();
        case UNRESOLVED_SYMLINK -> path = entry.getUnresolvedSymlink().getPath();
        case RUNFILES_TREE -> {
          ExecLogEntry.RunfilesTree runfilesTree = entry.getRunfilesTree();
          path = runfilesTree.getPath();
          references.toInputSet(runfilesTree.getInputSetId());
          references.toSymlinkEntrySet(runfilesTree.getSymlinksId());
          references.toSymlinkEntrySet(runfilesTree.getRootSymlinksId());
        }
        case INPUT_SET -> {
          ExecLogEntry.InputSet inputSet = entry.getInputSet();
          inputSet.getInputIdsList().forEach(references::toInput);
          inputSet.getTransitiveSetIdsList().forEach(references::toInputSet);
          for (int id : new HashSet<>(inputSet.getInputIdsList())) {
            inputSetsByInputId.merge(id, 1, Integer::sum);
          }
        }
        case SYMLINK_ENTRY_SET -> {
          ExecLogEntry.SymlinkEntrySet symlinkEntrySet = entry.getSymlinkEntrySet();
          symlinkEntrySet.getDirectEntriesMap().values().forEach(references::toInput);
          symlinkEntrySet.getTransitiveSetIdsList().forEach(references::toSymlinkEntrySet);
        }
        case SPAWN -> {
          ExecLogEntry.Spawn spawn = entry.getSpawn();
          spawns++;
          assertThat(entry.getId()).isEqualTo(0);
          references.toInputSet(spawn.getInputSetId());
          references.toInputSet(spawn.getToolSetId());
          for (ExecLogEntry.Output output : spawn.getOutputsList()) {
            assertThat(output.hasOutputId()).isTrue();
            references.toInput(output.getOutputId());
          }
        }
        default ->
            // This includes a second invocation entry.
            throw new AssertionError("unexpected entry: " + entry);
      }

      if (entry.getTypeCase() != ExecLogEntry.TypeCase.SPAWN) {
        // IDs are consecutive in the order in which entries appear in the log, which implies that
        // they are unique.
        assertThat(entry.getId()).isEqualTo(++lastId);
        references.typeById.put(entry.getId(), entry.getTypeCase());
      }
      if (path != null) {
        // No test logs an input that isn't deduplicated by its path more than once.
        assertWithMessage("entry logged more than once for path %s", path)
            .that(idByPath.put(path, entry.getId()))
            .isNull();
      }
    }

    assertThat(spawns).isEqualTo(expectedSpawns);
    // Every entry with an ID is referenced, as no test fails to log a spawn after logging some of
    // its inputs. An unreferenced entry would be left behind by a thread that computed an entry
    // which another thread computed as well.
    assertThat(references.referencedIds).containsExactlyElementsIn(references.typeById.keySet());
    for (String markerPath : markerPaths) {
      assertThat(idByPath).containsKey(markerPath);
      assertWithMessage("number of input sets directly containing %s", markerPath)
          .that(inputSetsByInputId.get(idByPath.get(markerPath)))
          .isEqualTo(1);
    }
  }

  /** Asserts that entries are only referenced after they appear in the log. */
  private static final class References {
    // The type of every entry with an ID that has appeared in the log so far.
    final HashMap<Integer, ExecLogEntry.TypeCase> typeById = new HashMap<>();
    final HashSet<Integer> referencedIds = new HashSet<>();

    /** Asserts that the ID is 0 or references an earlier file, directory or symlink entry. */
    void toInput(int id) {
      to(
          id,
          ImmutableSet.of(
              ExecLogEntry.TypeCase.FILE,
              ExecLogEntry.TypeCase.DIRECTORY,
              ExecLogEntry.TypeCase.UNRESOLVED_SYMLINK,
              ExecLogEntry.TypeCase.RUNFILES_TREE));
    }

    /** Asserts that the ID is 0 or references an earlier input set entry. */
    void toInputSet(int id) {
      to(id, ImmutableSet.of(ExecLogEntry.TypeCase.INPUT_SET));
    }

    /** Asserts that the ID is 0 or references an earlier symlink entry set entry. */
    void toSymlinkEntrySet(int id) {
      to(id, ImmutableSet.of(ExecLogEntry.TypeCase.SYMLINK_ENTRY_SET));
    }

    private void to(int id, ImmutableSet<ExecLogEntry.TypeCase> expectedTypes) {
      if (id == 0) {
        return;
      }
      assertWithMessage("entry %s is referenced before it appears in the log", id)
          .that(typeById)
          .containsKey(id);
      assertThat(typeById.get(id)).isIn(expectedTypes);
      referencedIds.add(id);
    }
  }

  @Override
  protected SpawnLogContext createSpawnLogContext(ImmutableMap<String, String> platformProperties)
      throws IOException, InterruptedException {
    return createSpawnLogContext(platformProperties, /* logSpawnPredicate= */ spawn -> true);
  }

  SpawnLogContext createSpawnLogContext(Predicate<Spawn> logSpawnPredicate)
      throws IOException, InterruptedException {
    return createSpawnLogContext(ImmutableMap.of(), logSpawnPredicate);
  }

  SpawnLogContext createSpawnLogContext(
      ImmutableMap<String, String> platformProperties, Predicate<Spawn> logSpawnPredicate)
      throws IOException, InterruptedException {
    return createSpawnLogContext(
        platformProperties, logSpawnPredicate, /* expandParamFiles= */ false);
  }

  SpawnLogContext createSpawnLogContext(
      ImmutableMap<String, String> platformProperties,
      Predicate<Spawn> logSpawnPredicate,
      boolean expandParamFiles)
      throws IOException, InterruptedException {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.setRemoteDefaultExecPropertiesField(platformProperties.entrySet().asList());

    return new CompactSpawnLogContext(
        new BufferedOutputStream(logPath.getOutputStream()),
        logPath.toString(),
        execRoot.asFragment(),
        TestConstants.WORKSPACE_NAME,
        remoteOptions,
        DigestHashFunction.SHA256,
        SyscallCache.NO_CACHE,
        new CompressionServiceImpl(),
        UUID.fromString("00000000-0000-0000-0000-000000000000"),
        storedEventHandler,
        logSpawnPredicate,
        expandParamFiles);
  }

  @Override
  protected void closeAndAssertLog(SpawnLogContext context, SpawnExec... expected)
      throws IOException {
    context.close();

    assertThat(reconstructLog()).containsExactlyElementsIn(expected).inOrder();
  }

  private ImmutableList<SpawnExec> reconstructLog() throws IOException {
    ImmutableList.Builder<SpawnExec> spawnExecs = ImmutableList.builder();
    try (SpawnLogReconstructor reconstructor =
        new SpawnLogReconstructor(logPath.getInputStream())) {
      SpawnExec ex;
      while ((ex = reconstructor.read()) != null) {
        spawnExecs.add(ex);
      }
    }
    return spawnExecs.build();
  }

  private ImmutableList<Protos.ExecLogEntry> closeAndReadCompactLog(SpawnLogContext context)
      throws IOException {
    context.close();

    ImmutableList.Builder<Protos.ExecLogEntry> entries = ImmutableList.builder();
    try (InputStream in = logPath.getInputStream();
        ZstdInputStream zstdIn = new ZstdInputStream(in)) {
      Protos.ExecLogEntry entry;
      while ((entry = Protos.ExecLogEntry.parseDelimitedFrom(zstdIn)) != null) {
        entries.add(entry);
      }
    }
    return entries.build();
  }
}
