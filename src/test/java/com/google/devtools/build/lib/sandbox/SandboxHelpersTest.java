// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxContents;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link SandboxHelpers}. */
@RunWith(TestParameterInjector.class)
public class SandboxHelpersTest {

  private static class CustomInMemoryFileSystem extends InMemoryFileSystem {
    private boolean forbidRenameTo = false;

    CustomInMemoryFileSystem() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public void renameTo(PathFragment source, PathFragment target) throws IOException {
      if (forbidRenameTo) {
        throw new IOException("error injected by test");
      }
      super.renameTo(source, target);
    }

    void forbidRenameTo() {
      forbidRenameTo = true;
    }
  }

  private final TreeDeleter treeDeleter = new SynchronousTreeDeleter();

  private final CustomInMemoryFileSystem fs = new CustomInMemoryFileSystem();
  private final Scratch scratch = new Scratch(fs);
  private Path execRoot;
  private Path sandboxRoot;
  @Nullable private ExecutorService executorToCleanup;

  @Before
  public void setUp() throws IOException {
    execRoot = scratch.dir("/execroot");
    sandboxRoot = scratch.dir("/sandbox");
  }

  @After
  public void tearDown() throws InterruptedException {
    if (executorToCleanup == null) {
      return;
    }

    executorToCleanup.shutdown();
    executorToCleanup.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
  }

  @Test
  public void processInputFiles_materializesParamFile() throws Exception {
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("paramFile"),
            ImmutableList.of("-a", "-b"),
            ParameterFileType.UNQUOTED);

    SandboxInputs inputs = SandboxHelpers.processInputFiles(inputMap(paramFile), execRoot);

    assertThat(inputs.getFiles())
        .containsExactly(PathFragment.create("paramFile"), execRoot.getChild("paramFile"));
    assertThat(inputs.getSymlinks()).isEmpty();
    assertThat(FileSystemUtils.readLines(execRoot.getChild("paramFile"), UTF_8))
        .containsExactly("-a", "-b")
        .inOrder();
    assertThat(execRoot.getChild("paramFile").isExecutable()).isTrue();
  }

  @Test
  public void processInputFiles_materializesBinToolsFile() throws Exception {
    BinTools.PathActionInput tool =
        new BinTools.PathActionInput(
            scratch.file("tool", "#!/bin/bash", "echo hello"),
            PathFragment.create("_bin/say_hello"));

    SandboxInputs inputs = SandboxHelpers.processInputFiles(inputMap(tool), execRoot);

    assertThat(inputs.getFiles())
        .containsExactly(
            PathFragment.create("_bin/say_hello"), execRoot.getRelative("_bin/say_hello"));
    assertThat(inputs.getSymlinks()).isEmpty();
    assertThat(FileSystemUtils.readLines(execRoot.getRelative("_bin/say_hello"), UTF_8))
        .containsExactly("#!/bin/bash", "echo hello")
        .inOrder();
    assertThat(execRoot.getRelative("_bin/say_hello").isExecutable()).isTrue();
  }

  /**
   * Test simulating a scenario when 2 parallel writes of the same virtual input both complete write
   * of the temp file and then proceed with post-processing steps one-by-one.
   */
  @Test
  public void sandboxInputMaterializeVirtualInput_parallelWritesForSameInput_writesCorrectFile()
      throws Exception {
    VirtualActionInput input = ActionsTestUtil.createVirtualActionInput("file", "hello");
    executorToCleanup = Executors.newSingleThreadExecutor();
    CyclicBarrier bothWroteTempFile = new CyclicBarrier(2);
    Semaphore finishProcessingSemaphore = new Semaphore(1);
    FileSystem customFs =
        new InMemoryFileSystem(DigestHashFunction.SHA1) {
          @Override
          @SuppressWarnings("UnsynchronizedOverridesSynchronized") // .await() inside
          public void setExecutable(PathFragment path, boolean executable) throws IOException {
            try {
              bothWroteTempFile.await();
              finishProcessingSemaphore.acquire();
            } catch (BrokenBarrierException | InterruptedException e) {
              throw new IllegalArgumentException(e);
            }
            super.setExecutable(path, executable);
          }
        };
    Scratch customScratch = new Scratch(customFs);
    Path customExecRoot = customScratch.dir("/execroot");

    Future<?> future =
        executorToCleanup.submit(
            () -> {
              try {
                SandboxHelpers.processInputFiles(inputMap(input), customExecRoot);
                finishProcessingSemaphore.release();
              } catch (IOException | InterruptedException e) {
                throw new IllegalArgumentException(e);
              }
            });
    SandboxHelpers.processInputFiles(inputMap(input), customExecRoot);
    finishProcessingSemaphore.release();
    future.get();

    assertThat(customExecRoot.readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("file", Dirent.Type.FILE));
    Path outputFile = customExecRoot.getChild("file");
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("hello");
    assertThat(outputFile.isExecutable()).isTrue();
  }

  private static ImmutableMap<PathFragment, ActionInput> inputMap(ActionInput... inputs) {
    return Arrays.stream(inputs)
        .collect(toImmutableMap(ActionInput::getExecPath, Function.identity()));
  }

  @Test
  public void atomicallyWriteVirtualInput_writesParamFile() throws Exception {
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("paramFile"),
            ImmutableList.of("-a", "-b"),
            ParameterFileType.UNQUOTED);

    paramFile.atomicallyWriteRelativeTo(scratch.resolve("/outputs"));

    assertThat(scratch.resolve("/outputs").readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("paramFile", Dirent.Type.FILE));
    Path outputFile = scratch.resolve("/outputs/paramFile");
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("-a", "-b").inOrder();
    assertThat(outputFile.isExecutable()).isTrue();
  }

  @Test
  public void atomicallyWriteVirtualInput_writesBinToolsFile() throws Exception {
    BinTools.PathActionInput tool =
        new BinTools.PathActionInput(
            scratch.file("tool", "tool_code"), PathFragment.create("tools/tool"));

    tool.atomicallyWriteRelativeTo(scratch.resolve("/outputs"));

    assertThat(scratch.resolve("/outputs").readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("tools", Dirent.Type.DIRECTORY));
    Path outputFile = scratch.resolve("/outputs/tools/tool");
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("tool_code");
    assertThat(outputFile.isExecutable()).isTrue();
  }

  @Test
  public void cleanExisting_updatesDirs() throws IOException, InterruptedException {
    Path inputTxt = scratch.getFileSystem().getPath(PathFragment.create("/hello.txt"));
    Path rootDir = execRoot.getParentDirectory();
    PathFragment input1 = PathFragment.create("existing/directory/with/input1.txt");
    PathFragment input2 = PathFragment.create("partial/directory/input2.txt");
    PathFragment input3 = PathFragment.create("new/directory/input3.txt");
    SandboxInputs inputs =
        new SandboxInputs(
            ImmutableMap.of(input1, inputTxt, input2, inputTxt, input3, inputTxt),
            ImmutableMap.of(),
            ImmutableMap.of());
    Set<PathFragment> inputsToCreate = new LinkedHashSet<>();
    LinkedHashSet<PathFragment> dirsToCreate = new LinkedHashSet<>();
    SandboxHelpers.populateInputsAndDirsToCreate(
        ImmutableSet.of(),
        inputsToCreate,
        dirsToCreate,
        Iterables.concat(
            ImmutableSet.of(), inputs.getFiles().keySet(), inputs.getSymlinks().keySet()),
        SandboxOutputs.create(
            ImmutableSet.of(PathFragment.create("out/dir/output.txt")), ImmutableSet.of()));

    PathFragment inputDir1 = input1.getParentDirectory();
    PathFragment inputDir2 = input2.getParentDirectory();
    PathFragment inputDir3 = input3.getParentDirectory();
    PathFragment outputDir = PathFragment.create("out/dir");
    assertThat(dirsToCreate).containsExactly(inputDir1, inputDir2, inputDir3, outputDir);
    assertThat(inputsToCreate).containsExactly(input1, input2, input3);

    // inputdir1 exists fully
    execRoot.getRelative(inputDir1).createDirectoryAndParents();
    // inputdir2 exists partially, should be kept nonetheless.
    execRoot
        .getRelative(inputDir2)
        .getParentDirectory()
        .getRelative("doomedSubdir")
        .createDirectoryAndParents();
    // inputDir3 just doesn't exist
    // outputDir only exists partially
    execRoot.getRelative(outputDir).getParentDirectory().createDirectoryAndParents();
    execRoot.getRelative("justSomeDir/thatIsDoomed").createDirectoryAndParents();
    // `thiswillbeafile/output` simulates a directory that was in the stashed dir but whose same
    // path is used later for a regular file.
    scratch.dir("/execroot/thiswillbeafile/output");
    scratch.file("/execroot/thiswillbeafile/output/file1");
    dirsToCreate.add(PathFragment.create("thiswillbeafile"));
    PathFragment input4 = PathFragment.create("thiswillbeafile/output");
    SandboxInputs inputs2 =
        new SandboxInputs(
            ImmutableMap.of(input1, inputTxt, input2, inputTxt, input3, inputTxt, input4, inputTxt),
            ImmutableMap.of(),
            ImmutableMap.of());
    SandboxHelpers.cleanExisting(
        rootDir, inputs2, inputsToCreate, dirsToCreate, execRoot, treeDeleter);
    assertThat(dirsToCreate).containsExactly(inputDir2, inputDir3, outputDir);
    assertThat(execRoot.getRelative("existing/directory/with").exists()).isTrue();
    assertThat(execRoot.getRelative("partial").exists()).isTrue();
    assertThat(execRoot.getRelative("partial/doomedSubdir").exists()).isFalse();
    assertThat(execRoot.getRelative("partial/directory").exists()).isFalse();
    assertThat(execRoot.getRelative("justSomeDir/thatIsDoomed").exists()).isFalse();
    assertThat(execRoot.getRelative("out").exists()).isTrue();
    assertThat(execRoot.getRelative("out/dir").exists()).isFalse();
  }

  @Test
  public void createContentMap_withOnlyEmptyInput_tracksContainingDirectories() {
    PathFragment emptyInput = PathFragment.create("api/__init__.py");
    Map<PathFragment, Path> files = new HashMap<>();
    files.put(emptyInput, null);
    SandboxInputs inputs = new SandboxInputs(files, ImmutableMap.of(), ImmutableMap.of());

    SandboxContents sandboxContents =
        SandboxHelpers.createContentMap(execRoot, inputs, SandboxOutputs.getEmptyInstance());

    SandboxContents workDirContents = sandboxContents.dirMap().get(execRoot.getBaseName());
    assertThat(workDirContents).isNotNull();
    SandboxContents apiContents = workDirContents.dirMap().get("api");
    assertThat(apiContents).isNotNull();
    assertThat(apiContents.fileMap()).containsEntry("__init__.py", null);
  }

  @Test
  public void updateContentMap_preservesEmptyInputInUpdatedDirectory() throws Exception {
    ManualClock clock = new ManualClock();
    Scratch scratch = new Scratch(new InMemoryFileSystem(clock, DigestHashFunction.SHA256));
    Path workDir = scratch.dir("/execroot");
    Path emptyFile = scratch.file("/execroot/api/__init__.py");
    PathFragment emptyInput = PathFragment.create("api/__init__.py");
    Map<PathFragment, Path> files = new HashMap<>();
    files.put(emptyInput, null);
    SandboxInputs inputs = new SandboxInputs(files, ImmutableMap.of(), ImmutableMap.of());
    SandboxContents contents =
        SandboxHelpers.createContentMap(workDir, inputs, SandboxOutputs.getEmptyInstance());
    long timestamp = clock.currentTimeMillis();

    clock.advanceMillis(1);
    Path unexpectedFile = scratch.file("/execroot/api/unexpected", "unexpected");
    SandboxHelpers.updateContentMap(workDir.getParentDirectory(), timestamp, contents);

    assertThat(emptyFile.isFile()).isTrue();
    assertThat(unexpectedFile.exists()).isFalse();
    assertThat(contents.dirMap().get("execroot").dirMap().get("api").fileMap())
        .containsEntry("__init__.py", null);
  }

  @Test
  public void updateContentMap_preservesKnownSymlinksAndDeletesUnexpected() throws Exception {
    ManualClock clock = new ManualClock();
    Scratch scratch = new Scratch(new InMemoryFileSystem(clock, DigestHashFunction.SHA256));
    Path workDir = scratch.dir("/execroot");
    Path sourceFile = scratch.file("/source/input.txt", "content");
    PathFragment knownInput = PathFragment.create("pkg/input.txt");
    Path knownSymlink = workDir.getRelative(knownInput);
    knownSymlink.getParentDirectory().createDirectoryAndParents();
    knownSymlink.createSymbolicLink(sourceFile.asFragment());

    Map<PathFragment, Path> files = new HashMap<>();
    files.put(knownInput, sourceFile);
    SandboxInputs inputs = new SandboxInputs(files, ImmutableMap.of(), ImmutableMap.of());
    SandboxContents contents =
        SandboxHelpers.createContentMap(workDir, inputs, SandboxOutputs.getEmptyInstance());
    long timestamp = clock.currentTimeMillis();

    clock.advanceMillis(1);
    Path unexpectedSymlink = workDir.getRelative("pkg/unexpected");
    unexpectedSymlink.createSymbolicLink(sourceFile.asFragment());
    FileSystemUtils.appendIsoLatin1(sourceFile, "_modified");

    SandboxHelpers.updateContentMap(workDir.getParentDirectory(), timestamp, contents);

    assertThat(knownSymlink.isSymbolicLink()).isTrue();
    assertThat(unexpectedSymlink.exists()).isFalse();
    assertThat(contents.dirMap().get("execroot").dirMap().get("pkg").fileMap())
        .containsEntry("input.txt", sourceFile.asFragment());
    assertThat(contents.dirMap().get("execroot").dirMap().get("pkg").fileMap())
        .doesNotContainKey("unexpected");
  }

  @Test
  public void cleanExisting_withInMemoryContents_tracksEmptyInputs(
      @TestParameter boolean keepEmptyInput) throws Exception {
    PathFragment emptyInput = PathFragment.create("api/__init__.py");
    PathFragment sharedInput = PathFragment.create("api/shared.py");
    Path sharedSource = scratch.file("/inputs/shared.py", "shared");
    Path emptyFile = scratch.file("/execroot/api/__init__.py");
    execRoot.getRelative(sharedInput).createSymbolicLink(sharedSource.asFragment());

    Map<PathFragment, Path> previousFiles = new HashMap<>();
    previousFiles.put(emptyInput, null);
    previousFiles.put(sharedInput, sharedSource);
    SandboxInputs previousInputs =
        new SandboxInputs(previousFiles, ImmutableMap.of(), ImmutableMap.of());
    SandboxContents sandboxContents =
        SandboxHelpers.createContentMap(
            execRoot, previousInputs, SandboxOutputs.getEmptyInstance());

    Map<PathFragment, Path> currentFiles = new HashMap<>(previousFiles);
    if (!keepEmptyInput) {
      currentFiles.remove(emptyInput);
    }
    SandboxInputs currentInputs =
        new SandboxInputs(currentFiles, ImmutableMap.of(), ImmutableMap.of());
    Set<PathFragment> inputsToCreate = new LinkedHashSet<>();
    Set<PathFragment> dirsToCreate = new LinkedHashSet<>();
    SandboxHelpers.populateInputsAndDirsToCreate(
        ImmutableSet.of(),
        inputsToCreate,
        dirsToCreate,
        currentFiles.keySet(),
        SandboxOutputs.getEmptyInstance());

    SandboxHelpers.cleanExisting(
        execRoot.getParentDirectory(),
        currentInputs,
        inputsToCreate,
        dirsToCreate,
        execRoot,
        treeDeleter,
        sandboxContents);

    assertThat(emptyFile.exists()).isEqualTo(keepEmptyInput);
    assertThat(inputsToCreate).isEmpty();
  }

  @Test
  public void populateInputsAndDirsToCreate_createsMappedDirectories() {
    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "outputs");
    ActionInput outputFile = ActionsTestUtil.createArtifact(outputRoot, "bin/config/dir/file");
    ActionInput outputDir =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            outputRoot, "bin/config/other_dir/subdir");
    PathMapper pathMapper =
        execPath -> PathFragment.create(execPath.getPathString().replace("config/", ""));
    Spawn spawn =
        new SpawnBuilder().withOutputs(outputFile, outputDir).setPathMapper(pathMapper).build();
    LinkedHashSet<PathFragment> writableDirs = new LinkedHashSet<>();
    LinkedHashSet<PathFragment> inputsToCreate = new LinkedHashSet<>();
    LinkedHashSet<PathFragment> dirsToCreate = new LinkedHashSet<>();

    SandboxHelpers.populateInputsAndDirsToCreate(
        writableDirs,
        inputsToCreate,
        dirsToCreate,
        ImmutableList.of(),
        SandboxHelpers.getOutputs(spawn));

    assertThat(writableDirs).isEmpty();
    assertThat(inputsToCreate).isEmpty();
    assertThat(dirsToCreate)
        .containsExactly(
            PathFragment.create("outputs/bin/dir"),
            PathFragment.create("outputs/bin/other_dir/subdir"));
  }

  @Test
  public void moveOutputs_movesFile(@TestParameter boolean forceCopy) throws Exception {
    if (forceCopy) {
      fs.forbidRenameTo();
    }

    Path sandboxFile = sandboxRoot.getRelative("output");
    FileSystemUtils.writeContent(sandboxFile, UTF_8, "hello");

    Spawn spawn = new SpawnBuilder().withOutputs("output").build();
    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    Path realFile = execRoot.getRelative("output");
    assertThat(realFile.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(realFile, UTF_8)).isEqualTo("hello");
  }

  @Test
  public void moveOutputs_movesSymlink(@TestParameter boolean forceCopy) throws Exception {
    if (forceCopy) {
      fs.forbidRenameTo();
    }

    Path sandboxSymlink = sandboxRoot.getRelative("output");
    sandboxSymlink.createSymbolicLink(PathFragment.create("target"));

    Spawn spawn = new SpawnBuilder().withOutputs("output").build();
    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    Path realSymlink = execRoot.getRelative("output");
    assertThat(realSymlink.isSymbolicLink()).isTrue();
    assertThat(realSymlink.readSymbolicLink()).isEqualTo(PathFragment.create("target"));
  }

  @Test
  public void moveOutputs_movesDirectory(@TestParameter boolean forceCopy) throws Exception {
    if (forceCopy) {
      fs.forbidRenameTo();
    }

    Path sandboxDir = sandboxRoot.getRelative("output");
    sandboxDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(sandboxDir.getRelative("file"), UTF_8, "hello");
    sandboxDir.getRelative("symlink").createSymbolicLink(PathFragment.create("target"));
    sandboxDir.getRelative("subdir").createDirectoryAndParents();

    Spawn spawn = new SpawnBuilder().withOutputs("output").build();
    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    Path realDir = execRoot.getRelative("output");
    assertThat(realDir.isDirectory()).isTrue();
    assertThat(realDir.getRelative("file").isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(realDir.getRelative("file"), UTF_8)).isEqualTo("hello");
    assertThat(realDir.getRelative("symlink").isSymbolicLink()).isTrue();
    assertThat(realDir.getRelative("symlink").readSymbolicLink())
        .isEqualTo(PathFragment.create("target"));
    assertThat(realDir.getRelative("subdir").isDirectory()).isTrue();
  }

  @Test
  public void moveOutputs_ignoresMissing(@TestParameter boolean forceCopy) throws Exception {
    if (forceCopy) {
      fs.forbidRenameTo();
    }

    Spawn spawn = new SpawnBuilder().withOutputs("output").build();
    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    assertThat(execRoot.getRelative("output").exists()).isFalse();
  }

  @Test
  public void moveOutputs_fixesPermissionsOnFileWhenCopying() throws Exception {
    fs.forbidRenameTo();

    Path sandboxFile = sandboxRoot.getRelative("output");
    FileSystemUtils.writeContent(sandboxFile, UTF_8, "hello");
    sandboxFile.chmod(0);

    Spawn spawn = new SpawnBuilder().withOutputs("output").build();
    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    Path realFile = execRoot.getRelative("output");
    assertThat(realFile.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(realFile, UTF_8)).isEqualTo("hello");
  }

  @Test
  public void moveOutputs_fixesPermissionsOnDirectoryWhenCopying() throws Exception {
    fs.forbidRenameTo();

    Path sandboxDir = sandboxRoot.getRelative("output");
    sandboxDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(sandboxDir.getRelative("file"), UTF_8, "hello");
    sandboxDir.chmod(0);

    Spawn spawn = new SpawnBuilder().withOutputs("output").build();
    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    Path realDir = execRoot.getRelative("output");
    assertThat(realDir.isDirectory()).isTrue();
    assertThat(realDir.getRelative("file").isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(realDir.getRelative("file"), UTF_8)).isEqualTo("hello");
  }

  @Test
  public void moveOutputs_mappedPathMovedToUnmappedPath(@TestParameter boolean forceCopy)
      throws Exception {
    if (forceCopy) {
      fs.forbidRenameTo();
    }

    PathFragment unmappedOutputPath = PathFragment.create("bin/config/output");
    PathMapper pathMapper =
        execPath -> PathFragment.create(execPath.getPathString().replace("config/", ""));
    Spawn spawn =
        new SpawnBuilder()
            .withOutputs(unmappedOutputPath.getPathString())
            .setPathMapper(pathMapper)
            .build();
    PathFragment mappedOutputPath = PathFragment.create("bin/output");
    sandboxRoot.getRelative(mappedOutputPath).getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeLinesAs(
        sandboxRoot.getRelative(mappedOutputPath), UTF_8, "hello", "pathmapper");

    SandboxHelpers.moveOutputs(SandboxHelpers.getOutputs(spawn), sandboxRoot, execRoot);

    assertThat(
            FileSystemUtils.readLines(
                execRoot.getRelative(unmappedOutputPath.getPathString()), UTF_8))
        .containsExactly("hello", "pathmapper")
        .inOrder();
  }

  @Test
  public void asynchronousTreeDeleter_shutdown_waitsForPendingDeletions() throws Exception {
    CountDownLatch deletionStarted = new CountDownLatch(1);
    CountDownLatch allowDeletionToComplete = new CountDownLatch(1);

    FileSystem customFs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public boolean delete(PathFragment path) throws IOException {
            deletionStarted.countDown();
            try {
              allowDeletionToComplete.await();
            } catch (InterruptedException e) {
              throw new IOException(e);
            }
            return super.delete(path);
          }
        };

    Scratch customScratch = new Scratch(customFs);
    Path trashBase = customScratch.dir("/trash");
    Path dir = customScratch.dir("/dir");
    customScratch.file("/dir/file.txt");

    AsynchronousTreeDeleter deleter = new AsynchronousTreeDeleter(trashBase);
    deleter.deleteTree(dir);

    // Wait until background thread starts deleting
    deletionStarted.await();

    ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor();
    executorToCleanup = executor;
    var unused = executor.schedule(allowDeletionToComplete::countDown, 100, TimeUnit.MILLISECONDS);

    deleter.shutdown();

    assertThat(trashBase.getDirectoryEntries()).isEmpty();
  }
  @Test
  public void asynchronousTreeDeleter_differentFileSystem_deletesSynchronously() throws Exception {
    FileSystem fs1 = new InMemoryFileSystem(DigestHashFunction.SHA256);
    FileSystem fs2 = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path trashBase = fs1.getPath("/trash");
    Path dir = fs2.getPath("/dir");
    dir.createDirectoryAndParents();
    dir.getChild("file.txt").createDirectoryAndParents();

    AsynchronousTreeDeleter deleter = new AsynchronousTreeDeleter(trashBase);
    deleter.deleteTree(dir);
    deleter.shutdown();

    assertThat(dir.exists()).isFalse();
    assertThat(trashBase.exists()).isFalse();
  }

  @Test
  public void asynchronousTreeDeleter_setThreads_allowsQueuedTasksToDrainInParallel()
      throws Exception {
    CountDownLatch task1Started = new CountDownLatch(1);
    CountDownLatch task2Started = new CountDownLatch(1);
    CountDownLatch allowActiveTasksToComplete = new CountDownLatch(1);

    CountDownLatch task3Started = new CountDownLatch(1);
    CountDownLatch task4Started = new CountDownLatch(1);
    CountDownLatch allowQueuedTasksToComplete = new CountDownLatch(1);

    FileSystem customFs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public boolean delete(PathFragment path) throws IOException {
            String base = path.getBaseName();
            switch (base) {
              case "file1.txt" -> {
                task1Started.countDown();
                awaitLatch(allowActiveTasksToComplete);
              }
              case "file2.txt" -> {
                task2Started.countDown();
                awaitLatch(allowActiveTasksToComplete);
              }
              case "file3.txt" -> {
                task3Started.countDown();
                awaitLatch(allowQueuedTasksToComplete);
              }
              case "file4.txt" -> {
                task4Started.countDown();
                awaitLatch(allowQueuedTasksToComplete);
              }
              default -> {}
            }
            return super.delete(path);
          }

          private void awaitLatch(CountDownLatch latch) throws IOException {
            try {
              latch.await();
            } catch (InterruptedException e) {
              throw new IOException(e);
            }
          }
        };

    Scratch customScratch = new Scratch(customFs);
    Path trashBase = customScratch.dir("/trash");
    Path dir1 = customScratch.dir("/dir1");
    Path dir2 = customScratch.dir("/dir2");
    Path dir3 = customScratch.dir("/dir3");
    Path dir4 = customScratch.dir("/dir4");
    customScratch.file("/dir1/file1.txt");
    customScratch.file("/dir2/file2.txt");
    customScratch.file("/dir3/file3.txt");
    customScratch.file("/dir4/file4.txt");

    AsynchronousTreeDeleter deleter = new AsynchronousTreeDeleter(trashBase);
    // Expand pool to 2 threads
    deleter.setThreads(2);
    // Queue task 1 and task 2 (occupying both workers)
    deleter.deleteTree(dir1);
    deleter.deleteTree(dir2);

    // Wait until both workers are actively running task 1 and task 2
    assertThat(task1Started.await(5, TimeUnit.SECONDS)).isTrue();
    assertThat(task2Started.await(5, TimeUnit.SECONDS)).isTrue();

    // Now submit task 3 and task 4 into the backlog queue
    deleter.deleteTree(dir3);
    deleter.deleteTree(dir4);

    // Downsize corePoolSize to 1 thread immediately (simulating post-startup reset)
    deleter.setThreads(1);

    // Allow task 1 and task 2 to complete, releasing the workers to drain the queue
    allowActiveTasksToComplete.countDown();

    // Both queued tasks (task 3 and task 4) should still be executed in parallel by the 2 workers
    assertThat(task3Started.await(5, TimeUnit.SECONDS)).isTrue();
    assertThat(task4Started.await(5, TimeUnit.SECONDS)).isTrue();

    // Allow queued tasks to finish
    allowQueuedTasksToComplete.countDown();

    deleter.shutdown();

    assertThat(trashBase.getDirectoryEntries()).isEmpty();
  }

  @Test
  public void sandboxStash_threadPoolConfiguration() throws Exception {
    int expectedPoolSize = Math.min(2, Math.max(1, Runtime.getRuntime().availableProcessors() / 8));
    assertThat(SandboxStash.getPoolSizeForTesting()).isEqualTo(expectedPoolSize);

    SandboxOptions options =
        Options.parse(
                SandboxOptions.class,
                "--reuse_sandbox_directories",
                "--experimental_inmemory_sandbox_stashes")
            .getOptions();
    Path sandboxBase = scratch.dir("/sandbox_stash_test");

    SandboxStash.initialize("ws", sandboxBase, options, new SynchronousTreeDeleter());
    try {
      SandboxStash stash = SandboxStash.getInstanceForTesting();
      assertThat(stash).isNotNull();
      ExecutorService executor = stash.getStashFileListingPoolForTesting();
      assertThat(executor).isInstanceOf(ThreadPoolExecutor.class);
      ThreadPoolExecutor pool = (ThreadPoolExecutor) executor;
      assertThat(pool.getMaximumPoolSize()).isEqualTo(expectedPoolSize);

      AtomicReference<Thread> workerThread = new AtomicReference<>();
      CountDownLatch threadCaptured = new CountDownLatch(1);
      var _ =
          pool.submit(
              () -> {
                workerThread.set(Thread.currentThread());
                threadCaptured.countDown();
              });
      assertThat(threadCaptured.await(5, TimeUnit.SECONDS)).isTrue();
      Thread thread = workerThread.get();
      assertThat(thread.getName()).startsWith("stash-file-listing-thread-");
      assertThat(thread.isDaemon()).isTrue();
      assertThat(thread.getPriority()).isEqualTo(Thread.MIN_PRIORITY);
    } finally {
      SandboxStash.initialize(
          "ws",
          sandboxBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          null);
    }
  }

  @Test
  public void sandboxStash_reinitialization_shutsDownPriorPool() throws Exception {
    SandboxOptions options =
        Options.parse(
                SandboxOptions.class,
                "--reuse_sandbox_directories",
                "--experimental_inmemory_sandbox_stashes")
            .getOptions();
    Path sandboxBase1 = scratch.dir("/sandbox_stash_reinit_1");
    Path sandboxBase2 = scratch.dir("/sandbox_stash_reinit_2");

    SandboxStash.initialize("ws1", sandboxBase1, options, new SynchronousTreeDeleter());
    SandboxStash stash1 = SandboxStash.getInstanceForTesting();
    assertThat(stash1).isNotNull();
    ExecutorService pool1 = stash1.getStashFileListingPoolForTesting();
    assertThat(pool1.isShutdown()).isFalse();

    // Re-initialize with a different workspace name
    SandboxStash.initialize("ws2", sandboxBase2, options, new SynchronousTreeDeleter());
    SandboxStash stash2 = SandboxStash.getInstanceForTesting();
    assertThat(stash2).isNotNull();
    assertThat(stash2).isNotSameInstanceAs(stash1);
    assertThat(pool1.isShutdown()).isTrue();

    // Disable reuse
    SandboxOptions optionsDisabled =
        Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions();
    SandboxStash.initialize("ws2", sandboxBase2, optionsDisabled, null);
    assertThat(SandboxStash.getInstanceForTesting()).isNull();
    assertThat(stash2.getStashFileListingPoolForTesting().isShutdown()).isTrue();
  }

  @Test
  public void sandboxStash_inMemoryStash_directRenameAndReuse() throws Exception {
    SandboxOptions options =
        Options.parse(
                SandboxOptions.class,
                "--reuse_sandbox_directories",
                "--experimental_inmemory_sandbox_stashes")
            .getOptions();
    Path sandboxBase = scratch.dir("/sandbox_stash_direct_rename");
    Path sandbox1 = scratch.dir("/sandbox_stash_direct_rename/1");
    Path execroot1 = sandbox1.getChild("execroot");
    execroot1.createDirectoryAndParents();
    Path file1 = scratch.file("/sandbox_stash_direct_rename/1/execroot/file.txt", "hello");

    SandboxStash.initialize("ws", sandboxBase, options, new SynchronousTreeDeleter());
    try {
      SandboxStash stash = SandboxStash.getInstanceForTesting();
      assertThat(stash).isNotNull();

      SandboxContents contents = new SandboxContents();
      contents.fileMap().put("file.txt", null);
      SandboxStash.setPathContents(sandbox1, contents);
      SandboxStash.setLastModified(sandbox1, file1.stat().getLastChangeTime());

      SandboxStash.stashSandbox(
          sandbox1,
          "Mnemonic",
          ImmutableMap.of(),
          SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
          new SynchronousTreeDeleter(),
          null);

      // Verify that tmp_sandbox_stash was never created (1-step direct rename)
      Path tmpStashDir = sandboxBase.getChild(SandboxStash.TEMPORARY_SANDBOX_STASH_BASE);
      assertThat(tmpStashDir.exists()).isFalse();

      // Wait for the background worker to finish scanning
      while (!stash.getInProgressStashesForTesting().isEmpty()) {
        Thread.sleep(10);
      }

      // Now prepare a second sandbox and reuse the stashed one
      Path sandbox2 = scratch.dir("/sandbox_stash_direct_rename/2");
      Path execroot2 = sandbox2.getChild("execroot");
      execroot2.createDirectoryAndParents();

      Optional<SandboxContents> taken =
          SandboxStash.takeStashedSandbox(
              sandbox2,
              "Mnemonic",
              ImmutableMap.of(),
              SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
              null);

      assertThat(taken).isNotNull();
      assertThat(taken).isPresent();
      assertThat(taken.get().fileMap()).containsKey("file.txt");
      assertThat(execroot2.getChild("file.txt").exists()).isTrue();
    } finally {
      SandboxStash.initialize(
          "ws",
          sandboxBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          null);
    }
  }

  @Test
  public void sandboxStash_inMemoryStash_inProgressStashFilteredOutUntilComplete()
      throws Exception {
    SandboxOptions options =
        Options.parse(
                SandboxOptions.class,
                "--reuse_sandbox_directories",
                "--experimental_inmemory_sandbox_stashes")
            .getOptions();
    Path sandboxBase = scratch.dir("/sandbox_stash_in_progress");
    Path sandbox1 = scratch.dir("/sandbox_stash_in_progress/1");
    Path execroot1 = sandbox1.getChild("execroot");
    execroot1.createDirectoryAndParents();
    Path file1 = scratch.file("/sandbox_stash_in_progress/1/execroot/file.txt", "content");

    SandboxStash.initialize("ws", sandboxBase, options, new SynchronousTreeDeleter());
    try {
      SandboxStash stash = SandboxStash.getInstanceForTesting();
      assertThat(stash).isNotNull();
      ExecutorService pool = stash.getStashFileListingPoolForTesting();

      // Block all worker threads in the pool so the stash background scan cannot start
      int poolSize = SandboxStash.getPoolSizeForTesting();
      CountDownLatch workersBlocked = new CountDownLatch(poolSize);
      CountDownLatch unblockWorkers = new CountDownLatch(1);
      for (int i = 0; i < poolSize; i++) {
        var _ =
            pool.submit(
                () -> {
                  workersBlocked.countDown();
                  try {
                    unblockWorkers.await();
                  } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                  }
                });
      }
      assertThat(workersBlocked.await(5, TimeUnit.SECONDS)).isTrue();

      SandboxContents contents = new SandboxContents();
      contents.fileMap().put("file.txt", null);
      SandboxStash.setPathContents(sandbox1, contents);
      SandboxStash.setLastModified(sandbox1, file1.stat().getLastChangeTime());

      SandboxStash.stashSandbox(
          sandbox1,
          "Mnemonic",
          ImmutableMap.of(),
          SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
          new SynchronousTreeDeleter(),
          null);

      // The stash directory and execroot are directly created on disk in
      // sandboxes/Mnemonic/1/execroot
      Path stashBase = sandboxBase.getChild(SandboxStash.SANDBOX_STASH_BASE);
      Path mnemonicStashDir = stashBase.getChild("Mnemonic");
      assertThat(mnemonicStashDir.exists()).isTrue();
      assertThat(mnemonicStashDir.getDirectoryEntries()).isNotEmpty();

      // But inProgressStashes must contain the stash, preventing reuse
      Set<Path> inProgress = stash.getInProgressStashesForTesting();
      assertThat(inProgress).isNotEmpty();

      Path sandbox2 = scratch.dir("/sandbox_stash_in_progress/2");
      Path execroot2 = sandbox2.getChild("execroot");
      execroot2.createDirectoryAndParents();

      // Attempting to take the stashed sandbox while it is in progress returns null
      Optional<SandboxContents> notTaken =
          SandboxStash.takeStashedSandbox(
              sandbox2,
              "Mnemonic",
              ImmutableMap.of(),
              SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
              null);
      assertThat(notTaken).isNull();

      // Now unblock the workers and allow the background listing to complete
      unblockWorkers.countDown();
      while (!stash.getInProgressStashesForTesting().isEmpty()) {
        Thread.sleep(10);
      }

      // After background scanning completes, the stash can now be taken
      Optional<SandboxContents> taken =
          SandboxStash.takeStashedSandbox(
              sandbox2,
              "Mnemonic",
              ImmutableMap.of(),
              SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
              null);
      assertThat(taken).isNotNull();
      assertThat(taken).isPresent();
      assertThat(taken.get().fileMap()).containsKey("file.txt");
      assertThat(execroot2.getChild("file.txt").exists()).isTrue();
    } finally {
      SandboxStash.initialize(
          "ws",
          sandboxBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          null);
    }
  }

  @Test
  public void sandboxStash_inMemoryStash_workerExceptionDeletesStashAndRemovesFromInProgress()
      throws Exception {
    SandboxOptions options =
        Options.parse(
                SandboxOptions.class,
                "--reuse_sandbox_directories",
                "--experimental_inmemory_sandbox_stashes")
            .getOptions();
    Path sandboxBase = scratch.dir("/sandbox_stash_error");
    Path sandbox1 = scratch.dir("/sandbox_stash_error/1");
    Path execroot1 = sandbox1.getChild("execroot");
    execroot1.createDirectoryAndParents();

    SandboxStash.initialize("ws", sandboxBase, options, new SynchronousTreeDeleter());
    try {
      SandboxStash stash = SandboxStash.getInstanceForTesting();
      assertThat(stash).isNotNull();

      // Omit pathToContents / pathToLastModified so the worker encounters an error
      SandboxStash.stashSandbox(
          sandbox1,
          "Mnemonic",
          ImmutableMap.of(),
          SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
          new SynchronousTreeDeleter(),
          null);

      // Wait for the background worker to finish handling the failure
      while (!stash.getInProgressStashesForTesting().isEmpty()) {
        Thread.sleep(10);
      }

      // Stash directory should be deleted so no corrupt or unusable stash remains
      Path stashBase = sandboxBase.getChild(SandboxStash.SANDBOX_STASH_BASE);
      Path mnemonicStashDir = stashBase.getChild("Mnemonic");
      assertThat(mnemonicStashDir.getDirectoryEntries()).isEmpty();
    } finally {
      SandboxStash.initialize(
          "ws",
          sandboxBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          null);
    }
  }
}

