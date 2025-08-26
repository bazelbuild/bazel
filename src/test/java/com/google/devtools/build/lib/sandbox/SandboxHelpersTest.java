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
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
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
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
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
          protected void setExecutable(PathFragment path, boolean executable) throws IOException {
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
}
