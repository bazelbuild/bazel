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

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.exec.SpawnLogContext.millisToProto;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.verify;

import com.google.common.base.Utf8;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.Protos.EnvironmentVariable;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.Platform;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.SortedMap;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for {@link SpawnLogContext}. */
@RunWith(TestParameterInjector.class)
public final class SpawnLogContextTest {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Mock private MessageOutputStream<SpawnExec> outputStream;

  private final DigestHashFunction digestHashFunction = DigestHashFunction.SHA256;
  private final FileSystem fs = new InMemoryFileSystem(digestHashFunction);
  private final Path execRoot = fs.getPath("/execroot");
  private final ArtifactRoot rootDir = ArtifactRoot.asSourceRoot(Root.fromPath(execRoot));
  private final ArtifactRoot outputDir =
      ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out");

  // A fake action filesystem that provides a fast digest, but refuses to compute it from the
  // file contents (which won't be available when building without the bytes).
  private static final class FakeActionFileSystem extends DelegateFileSystem {
    FakeActionFileSystem(FileSystem delegateFs) {
      super(delegateFs);
    }

    @Override
    protected byte[] getFastDigest(PathFragment path) throws IOException {
      return super.getDigest(path);
    }

    @Override
    protected byte[] getDigest(PathFragment path) throws IOException {
      throw new UnsupportedOperationException();
    }
  }

  /** Test parameter determining whether the spawn inputs are also tool inputs. */
  enum InputsMode {
    TOOLS,
    NON_TOOLS;

    boolean isTool() {
      return this == TOOLS;
    }
  }

  /** Test parameter determining whether to emulate building with or without the bytes. */
  enum OutputsMode {
    WITH_BYTES,
    WITHOUT_BYTES;

    FileSystem getActionFileSystem(FileSystem fs) {
      return this == WITHOUT_BYTES ? new FakeActionFileSystem(fs) : fs;
    }
  }

  /** Test parameter determining whether an input/output directory should be empty. */
  enum DirContents {
    EMPTY,
    NON_EMPTY;

    boolean isEmpty() {
      return this == EMPTY;
    }
  }

  @Test
  public void testFileInput(@TestParameter InputsMode inputsMode) throws Exception {
    Artifact fileInput = ActionsTestUtil.createArtifact(rootDir, "file");

    writeFile(fileInput, "abc");

    SpawnBuilder spawn = defaultSpawnBuilder().withInputs(fileInput);
    if (inputsMode.isTool()) {
      spawn.withTools(fileInput);
    }

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn.build(),
        createInputMetadataProvider(fileInput),
        createInputMap(fileInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addInputs(
                    File.newBuilder()
                        .setPath("file")
                        .setDigest(getDigest("abc"))
                        .setIsTool(inputsMode.isTool()))
                .build());
  }

  @Test
  public void testDirectoryInput(
      @TestParameter InputsMode inputsMode, @TestParameter DirContents dirContents)
      throws Exception {
    Artifact dirInput = ActionsTestUtil.createArtifact(rootDir, "dir");

    dirInput.getPath().createDirectoryAndParents();
    if (!dirContents.isEmpty()) {
      writeFile(dirInput.getPath().getChild("file"), "abc");
    }

    SpawnBuilder spawn = defaultSpawnBuilder().withInputs(dirInput);
    if (inputsMode.equals(InputsMode.TOOLS)) {
      spawn.withTools(dirInput);
    }

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn.build(),
        createInputMetadataProvider(dirInput),
        createInputMap(dirInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    // TODO(tjgq): Propagate tool bit to files inside source directories.
    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addAllInputs(
                    dirContents.isEmpty()
                        ? ImmutableList.of()
                        : ImmutableList.of(
                            File.newBuilder()
                                .setPath("dir/file")
                                .setDigest(getDigest("abc"))
                                .build()))
                .build());
  }

  @Test
  public void testTreeInput(
      @TestParameter InputsMode inputsMode, @TestParameter DirContents dirContents)
      throws Exception {
    SpecialArtifact treeInput =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputDir, "tree");

    treeInput.getPath().createDirectoryAndParents();
    if (!dirContents.isEmpty()) {
      writeFile(treeInput.getPath().getChild("child"), "abc");
    }

    SpawnBuilder spawn = defaultSpawnBuilder().withInputs(treeInput);
    if (inputsMode.isTool()) {
      spawn.withTools(treeInput);
    }

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn.build(),
        createInputMetadataProvider(treeInput),
        createInputMap(treeInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addAllInputs(
                    dirContents.isEmpty()
                        ? ImmutableList.of()
                        : ImmutableList.of(
                            File.newBuilder()
                                .setPath("out/tree/child")
                                .setDigest(getDigest("abc"))
                                .setIsTool(inputsMode.isTool())
                                .build()))
                .build());
  }

  @Test
  public void testRunfilesInput() throws Exception {
    Artifact runfilesInput = ActionsTestUtil.createArtifact(rootDir, "data.txt");

    writeFile(runfilesInput, "abc");

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        // In reality, the spawn would have a RunfilesSupplier and a runfiles middleman input.
        defaultSpawn(),
        createInputMetadataProvider(runfilesInput),
        /* inputMap= */ ImmutableSortedMap.of(
            outputDir.getExecPath().getRelative("foo.runfiles/data.txt"), runfilesInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    // TODO(tjgq): The path should be foo.runfiles/data.txt.
    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addInputs(File.newBuilder().setPath("data.txt").setDigest(getDigest("abc")))
                .build());
  }

  @Test
  public void testAbsolutePathInput() throws Exception {
    // Only observed to occur for source files inside a fileset.
    Path absolutePath = fs.getPath("/some/file.txt");
    ActionInput absolutePathInput = ActionInputHelper.fromPath(absolutePath.asFragment());

    writeFile(absolutePath, "abc");

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        new StaticInputMetadataProvider(
            ImmutableMap.of(absolutePathInput, FileArtifactValue.createForTesting(absolutePath))),
        /* inputMap= */ ImmutableSortedMap.of(absolutePath.asFragment(), absolutePathInput),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addInputs(File.newBuilder().setPath("/some/file.txt").setDigest(getDigest("abc")))
                .build());
  }

  @Test
  public void testEmptyInput() throws Exception {
    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        /* inputMap= */ ImmutableSortedMap.of(
            outputDir.getExecPath().getRelative("__init__.py"), VirtualActionInput.EMPTY_MARKER),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    // TODO(tjgq): It would make more sense to report an empty file.
    verify(outputStream).write(defaultSpawnExec());
  }

  @Test
  public void testFileOutput(@TestParameter OutputsMode outputsMode) throws Exception {
    Artifact fileOutput = ActionsTestUtil.createArtifact(outputDir, "file");

    writeFile(fileOutput, "abc");

    Spawn spawn = defaultSpawnBuilder().withOutputs(fileOutput).build();

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn,
        createInputMetadataProvider(),
        createInputMap(),
        outputsMode.getActionFileSystem(fs),
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addListedOutputs("out/file")
                .addActualOutputs(File.newBuilder().setPath("out/file").setDigest(getDigest("abc")))
                .build());
  }

  @Test
  public void testDirectoryOutput(
      @TestParameter OutputsMode outputsMode, @TestParameter DirContents dirContents)
      throws Exception {
    Artifact dirOutput = ActionsTestUtil.createArtifact(outputDir, "dir");

    dirOutput.getPath().createDirectoryAndParents();
    if (!dirContents.isEmpty()) {
      writeFile(dirOutput.getPath().getChild("file"), "abc");
    }

    SpawnBuilder spawn = defaultSpawnBuilder().withOutputs(dirOutput);

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn.build(),
        createInputMetadataProvider(),
        createInputMap(),
        outputsMode.getActionFileSystem(fs),
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addListedOutputs("out/dir")
                .addAllActualOutputs(
                    dirContents.isEmpty()
                        ? ImmutableList.of()
                        : ImmutableList.of(
                            File.newBuilder()
                                .setPath("out/dir/file")
                                .setDigest(getDigest("abc"))
                                .build()))
                .build());
  }

  @Test
  public void testTreeOutput(
      @TestParameter OutputsMode outputsMode, @TestParameter DirContents dirContents)
      throws Exception {
    SpecialArtifact treeOutput =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputDir, "tree");

    treeOutput.getPath().createDirectoryAndParents();
    if (!dirContents.isEmpty()) {
      writeFile(treeOutput.getPath().getChild("child"), "abc");
    }

    Spawn spawn = defaultSpawnBuilder().withOutputs(treeOutput).build();

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn,
        createInputMetadataProvider(),
        createInputMap(),
        outputsMode.getActionFileSystem(fs),
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addListedOutputs("out/tree")
                .addAllActualOutputs(
                    dirContents.isEmpty()
                        ? ImmutableList.of()
                        : ImmutableList.of(
                            File.newBuilder()
                                .setPath("out/tree/child")
                                .setDigest(getDigest("abc"))
                                .build()))
                .build());
  }

  @Test
  public void testEnvironment() throws Exception {
    Spawn spawn =
        defaultSpawnBuilder().withEnvironment("SPAM", "eggs").withEnvironment("FOO", "bar").build();

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn,
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .addEnvironmentVariables(
                    EnvironmentVariable.newBuilder().setName("FOO").setValue("bar"))
                .addEnvironmentVariables(
                    EnvironmentVariable.newBuilder().setName("SPAM").setValue("eggs"))
                .build());
  }

  @Test
  public void testDefaultPlatformProperties() throws Exception {
    SpawnLogContext spawnLogContext = createSpawnLogContext(ImmutableMap.of("a", "1", "b", "2"));

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .setPlatform(
                    Platform.newBuilder()
                        .addProperties(Platform.Property.newBuilder().setName("a").setValue("1"))
                        .addProperties(Platform.Property.newBuilder().setName("b").setValue("2"))
                        .build())
                .build());
  }

  @Test
  public void testSpawnPlatformProperties() throws Exception {
    Spawn spawn =
        defaultSpawnBuilder().withExecProperties(ImmutableMap.of("a", "3", "c", "4")).build();

    SpawnLogContext spawnLogContext = createSpawnLogContext(ImmutableMap.of("a", "1", "b", "2"));

    spawnLogContext.logSpawn(
        spawn,
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    // The spawn properties should override the default properties.
    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .setPlatform(
                    Platform.newBuilder()
                        .addProperties(Platform.Property.newBuilder().setName("a").setValue("3"))
                        .addProperties(Platform.Property.newBuilder().setName("b").setValue("2"))
                        .addProperties(Platform.Property.newBuilder().setName("c").setValue("4"))
                        .build())
                .build());
  }

  @Test
  public void testExecutionInfo(
      @TestParameter({"no-remote", "no-cache", "no-remote-cache"}) String requirement)
      throws Exception {
    Spawn spawn = defaultSpawnBuilder().withExecutionInfo(requirement, "").build();

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        spawn,
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .setRemotable(!requirement.equals("no-remote"))
                .setCacheable(!requirement.equals("no-cache"))
                .setRemoteCacheable(
                    !requirement.equals("no-cache")
                        && !requirement.equals("no-remote")
                        && !requirement.equals("no-remote-cache"))
                .build());
  }

  @Test
  public void testCacheHit() throws Exception {
    SpawnLogContext spawnLogContext = createSpawnLogContext();

    SpawnResult result = defaultSpawnResultBuilder().setCacheHit(true).build();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        result);

    verify(outputStream).write(defaultSpawnExecBuilder().setCacheHit(true).build());
  }

  @Test
  public void testDigest() throws Exception {
    SpawnLogContext spawnLogContext = createSpawnLogContext();

    SpawnResult result = defaultSpawnResultBuilder().setDigest(getDigest("abc")).build();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        result);

    verify(outputStream).write(defaultSpawnExecBuilder().setDigest(getDigest("abc")).build());
  }

  @Test
  public void testTimeout() throws Exception {
    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        /* timeout= */ Duration.ofSeconds(42),
        defaultSpawnResult());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder().setTimeoutMillis(Duration.ofSeconds(42).toMillis()).build());
  }

  @Test
  public void testSpawnMetrics() throws Exception {
    SpawnMetrics metrics = SpawnMetrics.Builder.forLocalExec().setTotalTimeInMs(1).build();

    SpawnLogContext spawnLogContext = createSpawnLogContext();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        defaultSpawnResultBuilder().setSpawnMetrics(metrics).build());

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .setMetrics(Protos.SpawnMetrics.newBuilder().setTotalTime(millisToProto(1)))
                .build());
  }

  @Test
  public void testStatus() throws Exception {
    SpawnLogContext spawnLogContext = createSpawnLogContext();

    // SpawnResult requires a non-zero exit code and non-null failure details when the status isn't
    // successful.
    SpawnResult result =
        defaultSpawnResultBuilder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(37)
            .setFailureDetail(
                FailureDetail.newBuilder()
                    .setMessage("oops")
                    .setCrash(Crash.getDefaultInstance())
                    .build())
            .build();

    spawnLogContext.logSpawn(
        defaultSpawn(),
        createInputMetadataProvider(),
        createInputMap(),
        fs,
        defaultTimeout(),
        result);

    verify(outputStream)
        .write(
            defaultSpawnExecBuilder()
                .setExitCode(37)
                .setStatus(Status.NON_ZERO_EXIT.toString())
                .build());
  }

  private static Duration defaultTimeout() {
    return Duration.ZERO;
  }

  private static SpawnBuilder defaultSpawnBuilder() {
    return new SpawnBuilder("cmd", "--opt");
  }

  private static Spawn defaultSpawn() {
    return defaultSpawnBuilder().build();
  }

  private static SpawnResult.Builder defaultSpawnResultBuilder() {
    return new SpawnResult.Builder().setRunnerName("runner").setStatus(Status.SUCCESS);
  }

  private static SpawnResult defaultSpawnResult() {
    return defaultSpawnResultBuilder().build();
  }

  private static SpawnExec.Builder defaultSpawnExecBuilder() {
    return SpawnExec.newBuilder()
        .addCommandArgs("cmd")
        .addCommandArgs("--opt")
        .setRunner("runner")
        .setRemotable(true)
        .setCacheable(true)
        .setRemoteCacheable(true)
        .setMnemonic("Mnemonic")
        .setTargetLabel("//dummy:label")
        .setMetrics(Protos.SpawnMetrics.getDefaultInstance());
  }

  private static SpawnExec defaultSpawnExec() {
    return defaultSpawnExecBuilder().build();
  }

  private static InputMetadataProvider createInputMetadataProvider(Artifact... artifacts)
      throws Exception {
    ImmutableMap.Builder<ActionInput, FileArtifactValue> builder = ImmutableMap.builder();
    for (Artifact artifact : artifacts) {
      if (artifact.isTreeArtifact()) {
        // Emulate ActionInputMap: add both tree and children.
        TreeArtifactValue treeMetadata = createTreeArtifactValue(artifact);
        builder.put(artifact, treeMetadata.getMetadata());
        for (Map.Entry<TreeFileArtifact, FileArtifactValue> entry :
            treeMetadata.getChildValues().entrySet()) {
          builder.put(entry.getKey(), entry.getValue());
        }
      } else {
        builder.put(artifact, FileArtifactValue.createForTesting(artifact));
      }
    }
    return new StaticInputMetadataProvider(builder.buildOrThrow());
  }

  private static SortedMap<PathFragment, ActionInput> createInputMap(Artifact... artifacts)
      throws Exception {
    ImmutableSortedMap.Builder<PathFragment, ActionInput> builder =
        ImmutableSortedMap.naturalOrder();
    for (Artifact artifact : artifacts) {
      if (artifact.isTreeArtifact()) {
        // Emulate SpawnInputExpander: expand to children, preserve if empty.
        TreeArtifactValue treeMetadata = createTreeArtifactValue(artifact);
        if (treeMetadata.getChildren().isEmpty()) {
          builder.put(artifact.getExecPath(), artifact);
        } else {
          for (TreeFileArtifact child : treeMetadata.getChildren()) {
            builder.put(child.getExecPath(), child);
          }
        }
      } else {
        builder.put(artifact.getExecPath(), artifact);
      }
    }
    return builder.buildOrThrow();
  }

  private static TreeArtifactValue createTreeArtifactValue(Artifact tree) throws Exception {
    checkState(tree.isTreeArtifact());
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder((SpecialArtifact) tree);
    TreeArtifactValue.visitTree(
        tree.getPath(),
        (parentRelativePath, type) -> {
          if (type.equals(Dirent.Type.DIRECTORY)) {
            return;
          }
          TreeFileArtifact child =
              TreeFileArtifact.createTreeOutput((SpecialArtifact) tree, parentRelativePath);
          builder.putChild(child, FileArtifactValue.createForTesting(child));
        });
    return builder.build();
  }

  private SpawnLogContext createSpawnLogContext() {
    return createSpawnLogContext(ImmutableSortedMap.of());
  }

  private SpawnLogContext createSpawnLogContext(ImmutableMap<String, String> platformProperties) {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteDefaultExecProperties = platformProperties.entrySet().asList();

    return new SpawnLogContext(
        execRoot.asFragment(),
        outputStream,
        Options.getDefaults(ExecutionOptions.class),
        remoteOptions,
        DigestHashFunction.SHA256,
        SyscallCache.NO_CACHE);
  }

  private Digest getDigest(String content) {
    return Digest.newBuilder()
        .setHash(digestHashFunction.getHashFunction().hashString(content, UTF_8).toString())
        .setSizeBytes(Utf8.encodedLength(content))
        .setHashFunctionName(digestHashFunction.toString())
        .build();
  }

  private static void writeFile(Artifact artifact, String contents) throws IOException {
    writeFile(artifact.getPath(), contents);
  }

  private static void writeFile(Path path, String contents) throws IOException {
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(path, UTF_8, contents);
  }
}
