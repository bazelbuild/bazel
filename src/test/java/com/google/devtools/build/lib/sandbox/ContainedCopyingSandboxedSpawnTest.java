// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.NativePosixFilesException;
import com.google.devtools.build.lib.unix.NativePosixFilesService.StatErrorHandling;
import com.google.devtools.build.lib.unix.NativePosixFilesServiceImpl;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.nio.file.Files;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ContainedCopyingSandboxedSpawnTest {
  private Path workspace;
  private Path sandbox;
  private Path sandboxExecRoot;

  @Before
  public void setUp() throws IOException {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path root = fs.getPath(TestUtils.tmpDir());
    workspace = root.getRelative("workspace");
    workspace.createDirectoryAndParents();
    sandbox = root.getRelative("sandbox");
    sandbox.createDirectory();
    sandboxExecRoot = sandbox.getRelative("execroot");
  }

  private ContainedCopyingSandboxedSpawn spawn(
      ImmutableMap<PathFragment, Path> files,
      ImmutableMap<PathFragment, PathFragment> symlinks) {
    return spawn(files, symlinks, ImmutableMap.of());
  }

  private ContainedCopyingSandboxedSpawn spawn(
      ImmutableMap<PathFragment, Path> files,
      ImmutableMap<PathFragment, PathFragment> symlinks,
      ImmutableMap<String, String> rawLinks) {
    return new ContainedCopyingSandboxedSpawn(
        sandbox,
        sandboxExecRoot,
        ImmutableList.of("/bin/true"),
        ImmutableMap.of(),
        new SandboxInputs(files, ImmutableMap.of(), symlinks),
        SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
        ImmutableSet.of(),
        new SynchronousTreeDeleter(),
        /* statisticsPath= */ null,
        "Test",
        /* targetLabel= */ null) {
      @Override
      protected void cloneFile(Path source, Path target) throws IOException {
        if (source.isDirectory()) {
          target.createDirectory();
          FileSystemUtils.copyTreesBelow(source, target);
        } else {
          FileSystemUtils.copyFile(source, target);
        }
      }

      @Override
      protected String readLink(Path link) throws IOException {
        return rawLinks.getOrDefault(link.getBaseName(), link.readSymbolicLink().getPathString());
      }
    };
  }

  @Test
  public void copiesFilesWithoutLinkingToHost() throws Exception {
    Path source = workspace.getRelative("input.txt");
    FileSystemUtils.writeContent(source, "original".getBytes(UTF_8));
    spawn(ImmutableMap.of(PathFragment.create("input.txt"), source), ImmutableMap.of())
        .createFileSystem();

    Path copy = sandboxExecRoot.getRelative("input.txt");
    assertThat(copy.isSymbolicLink()).isFalse();
    assertThat(new String(copy.getInputStream().readAllBytes(), UTF_8)).isEqualTo("original");
    FileSystemUtils.writeContent(copy, "changed".getBytes(UTF_8));
    assertThat(new String(source.getInputStream().readAllBytes(), UTF_8)).isEqualTo("original");
  }

  @Test
  public void copiesLargeInputSetInParallel() throws Exception {
    ImmutableMap.Builder<PathFragment, Path> files = ImmutableMap.builder();
    for (int i = 0; i < 320; i++) {
      String name = "file-" + i;
      Path source = workspace.getRelative(name);
      FileSystemUtils.writeContent(source, name.getBytes(UTF_8));
      files.put(PathFragment.create(name), source);
    }

    spawn(files.buildOrThrow(), ImmutableMap.of()).createFileSystem();

    for (int i = 0; i < 320; i++) {
      String name = "file-" + i;
      assertThat(new String(sandboxExecRoot.getRelative(name).getInputStream().readAllBytes(), UTF_8))
          .isEqualTo(name);
    }
  }

  @Test
  public void parallelCloneFailureFailsSandboxSetup() throws Exception {
    ImmutableMap.Builder<PathFragment, Path> files = ImmutableMap.builder();
    for (int i = 0; i < 320; i++) {
      String name = "file-" + i;
      Path source = workspace.getRelative(name);
      if (i != 160) {
        FileSystemUtils.writeContent(source, name.getBytes(UTF_8));
      }
      files.put(PathFragment.create(name), source);
    }

    assertThrows(
        IOException.class, () -> spawn(files.buildOrThrow(), ImmutableMap.of()).createFileSystem());
  }

  @Test
  public void parallelDirectoryCloneStillChecksSymlinkContainment() throws Exception {
    ImmutableMap.Builder<PathFragment, Path> files = ImmutableMap.builder();
    for (int i = 0; i < 320; i++) {
      String name = "file-" + i;
      Path source = workspace.getRelative(name);
      FileSystemUtils.writeContent(source, name.getBytes(UTF_8));
      files.put(PathFragment.create(name), source);
    }
    Path tree = workspace.getRelative("tree");
    tree.createDirectory();
    tree.getRelative("escape").createSymbolicLink(workspace.asFragment());
    files.put(PathFragment.create("tree"), tree);

    IOException e =
        assertThrows(
            IOException.class,
            () -> spawn(files.buildOrThrow(), ImmutableMap.of()).createFileSystem());
    assertThat(e).hasMessageThat().contains("escapes execroot");
  }

  @Test
  public void copiesFileThroughSourceSymlinkWithoutPreservingHostLink() throws Exception {
    Path source = workspace.getRelative("input.txt");
    FileSystemUtils.writeContent(source, "original".getBytes(UTF_8));
    Path alias = workspace.getRelative("alias.txt");
    alias.createSymbolicLink(PathFragment.create("input.txt"));
    spawn(ImmutableMap.of(PathFragment.create("alias.txt"), alias), ImmutableMap.of())
        .createFileSystem();

    Path copy = sandboxExecRoot.getRelative("alias.txt");
    assertThat(copy.isSymbolicLink()).isFalse();
    assertThat(new String(copy.getInputStream().readAllBytes(), UTF_8)).isEqualTo("original");
  }

  @Test
  public void copiesDirectoryWithContainedRelativeSymlink() throws Exception {
    Path tree = workspace.getRelative("tree");
    tree.createDirectory();
    FileSystemUtils.writeContent(tree.getRelative("file.txt"), "ok".getBytes(UTF_8));
    tree.getRelative("link.txt").createSymbolicLink(PathFragment.create("file.txt"));
    spawn(ImmutableMap.of(PathFragment.create("tree"), tree), ImmutableMap.of())
        .createFileSystem();

    Path copy = sandboxExecRoot.getRelative("tree");
    assertThat(copy.getRelative("link.txt").resolveSymbolicLinks())
        .isEqualTo(copy.getRelative("file.txt"));
    assertThat(copy.getRelative("file.txt").isSymbolicLink()).isFalse();
  }

  @Test
  public void rejectsDeclaredSymlinkEscapingExecroot() {
    var symlinks =
        ImmutableMap.of(PathFragment.create("link"), PathFragment.create("../../workspace"));
    IOException e =
        assertThrows(
            IOException.class,
            () -> spawn(ImmutableMap.of(), symlinks).createFileSystem());
    assertThat(e).hasMessageThat().contains("escapes execroot");
  }

  @Test
  public void rejectsDirectorySymlinkEscapingExecroot() throws Exception {
    Path tree = workspace.getRelative("tree");
    tree.createDirectory();
    tree.getRelative("link").createSymbolicLink(workspace.asFragment());
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                spawn(ImmutableMap.of(PathFragment.create("tree"), tree), ImmutableMap.of())
                    .createFileSystem());
    assertThat(e).hasMessageThat().contains("escapes execroot");
  }

  @Test
  public void validatesClonedDirectoryEvenIfSourceAppearsToBeAFile() throws Exception {
    Path source = workspace.getRelative("input");
    FileSystemUtils.writeContent(source, "original".getBytes(UTF_8));
    ContainedCopyingSandboxedSpawn changingSource =
        new ContainedCopyingSandboxedSpawn(
            sandbox,
            sandboxExecRoot,
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("input"), source),
                ImmutableMap.of(),
                ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableSet.of(),
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null,
            "Test",
            /* targetLabel= */ null) {
          @Override
          protected void cloneFile(Path ignoredSource, Path target) throws IOException {
            target.createDirectory();
            target.getRelative("escape").createSymbolicLink(workspace.asFragment());
          }

          @Override
          protected String readLink(Path link) throws IOException {
            return link.readSymbolicLink().getPathString();
          }
        };

    IOException e = assertThrows(IOException.class, changingSource::createFileSystem);
    assertThat(e).hasMessageThat().contains("escapes execroot");
  }

  @Test
  public void rejectsDirectorySymlinkEscapingThroughAnotherLink() throws Exception {
    Path tree = workspace.getRelative("tree");
    tree.createDirectory();
    tree.getRelative("alias").createSymbolicLink(PathFragment.create(".."));
    // PathFragment normalizes alias/..; a native readlink preserves the raw target instead.
    tree.getRelative("tricky").createSymbolicLink(PathFragment.create("alias/../outside"));

    IOException e =
        assertThrows(
            IOException.class,
            () ->
                spawn(
                        ImmutableMap.of(PathFragment.create("tree"), tree),
                        ImmutableMap.of(),
                        ImmutableMap.of("tricky", "alias/../outside"))
                    .createFileSystem());
    assertThat(e).hasMessageThat().contains("escapes execroot");
  }

  @Test
  public void rejectsRawClonedDirectoryLinkEscapingThroughAnotherLink() throws Exception {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    FileSystem fs =
        new UnixFileSystem(
            DigestHashFunction.SHA256,
            /* hashAttributeName= */ "",
            new NativePosixFilesServiceImpl());
    Path root = fs.getPath(TestUtils.tmpDir());
    Path tree = root.getRelative("nativeTree");
    tree.createDirectory();
    Files.createSymbolicLink(
        java.nio.file.Path.of(tree.getRelative("alias").getPathString()),
        java.nio.file.Path.of(".."));
    Files.createSymbolicLink(
        java.nio.file.Path.of(tree.getRelative("tricky").getPathString()),
        java.nio.file.Path.of("alias/../outside"));
    Path nativeSandbox = root.getRelative("nativeSandbox");
    nativeSandbox.createDirectory();
    Path nativeExecRoot = nativeSandbox.getRelative("execroot");
    ContainedCopyingSandboxedSpawn nativeSpawn =
        new ContainedCopyingSandboxedSpawn(
            nativeSandbox,
            nativeExecRoot,
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("tree"), tree),
                ImmutableMap.of(),
                ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableSet.of(),
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null,
            "Test",
            /* targetLabel= */ null);

    IOException e = assertThrows(IOException.class, nativeSpawn::createFileSystem);
    assertThat(e).hasMessageThat().contains("escapes execroot");
  }

  @Test
  public void reusesOnlyUnmodifiedClonesWithCurrentSources() throws Exception {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    NativePosixFilesServiceImpl nativeFiles = new NativePosixFilesServiceImpl();
    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, "", nativeFiles);
    Path root = fs.getPath(TestUtils.tmpDir()).getRelative("clone-stash-test");
    root.createDirectoryAndParents();
    Path source = root.getRelative("source.txt");
    FileSystemUtils.writeContent(source, "first".getBytes(UTF_8));
    Path stableSource = root.getRelative("stable.txt");
    FileSystemUtils.writeContent(stableSource, "stable".getBytes(UTF_8));
    Path stashBase = root.getRelative("stashes");
    stashBase.createDirectory();
    AsynchronousTreeDeleter deleter = new AsynchronousTreeDeleter(root.getRelative("trash"));
    SandboxStash.initialize(
        stashBase,
        Options.parse(SandboxOptions.class, "--reuse_sandbox_directories").getOptions(),
        new SynchronousTreeDeleter());
    try {
      ImmutableMap<PathFragment, Path> files =
          ImmutableMap.of(
              PathFragment.create("input.txt"), source,
              PathFragment.create("stable.txt"), stableSource);
      Path firstPath = root.getRelative("first");
      firstPath.createDirectory();
      ContainedCopyingSandboxedSpawn first =
          nativeSpawn(firstPath, files, ImmutableMap.of(), deleter);
      first.createFileSystem();
      long originalInode = inode(nativeFiles, firstPath.getRelative("execroot/input.txt"));
      long stableInode = inode(nativeFiles, firstPath.getRelative("execroot/stable.txt"));
      first.delete();

      Path secondPath = root.getRelative("second");
      secondPath.createDirectory();
      ContainedCopyingSandboxedSpawn second =
          nativeSpawn(secondPath, files, ImmutableMap.of(), deleter);
      second.createFileSystem();
      assertThat(inode(nativeFiles, secondPath.getRelative("execroot/input.txt")))
          .isEqualTo(originalInode);
      FileSystemUtils.writeContent(
          secondPath.getRelative("execroot/scratch.tmp"), "disposable".getBytes(UTF_8));
      secondPath.getRelative("execroot/scratch-link").createSymbolicLink(source.asFragment());
      secondPath
          .getRelative("execroot/scratch-link-dangling")
          .createSymbolicLink(PathFragment.create("missing-target"));
      second.delete();

      Path thirdPath = root.getRelative("third");
      thirdPath.createDirectory();
      ContainedCopyingSandboxedSpawn third =
          nativeSpawn(thirdPath, files, ImmutableMap.of(), deleter);
      third.createFileSystem();
      assertThat(thirdPath.getRelative("execroot/scratch.tmp").exists()).isFalse();
      assertThat(thirdPath.getRelative("execroot/scratch-link").exists()).isFalse();
      assertThat(thirdPath.getRelative("execroot/scratch-link-dangling").exists(Symlinks.NOFOLLOW))
          .isFalse();
      assertThat(new String(source.getInputStream().readAllBytes(), UTF_8)).isEqualTo("first");
      assertThat(inode(nativeFiles, thirdPath.getRelative("execroot/input.txt")))
          .isEqualTo(originalInode);
      third.delete();

      FileSystemUtils.writeContent(source, "updated".getBytes(UTF_8));
      Path fourthPath = root.getRelative("fourth");
      fourthPath.createDirectory();
      ContainedCopyingSandboxedSpawn fourth =
          nativeSpawn(fourthPath, files, ImmutableMap.of(), deleter);
      fourth.createFileSystem();
      Path fourthCopy = fourthPath.getRelative("execroot/input.txt");
      assertThat(new String(fourthCopy.getInputStream().readAllBytes(), UTF_8))
          .isEqualTo("updated");
      assertThat(inode(nativeFiles, fourthPath.getRelative("execroot/stable.txt")))
          .isEqualTo(stableInode);
      var originalMtime =
          Files.getLastModifiedTime(java.nio.file.Path.of(fourthCopy.getPathString()));
      FileSystemUtils.writeContent(fourthCopy, "tampered".getBytes(UTF_8));
      Files.setLastModifiedTime(
          java.nio.file.Path.of(fourthCopy.getPathString()), originalMtime);
      fourth.delete();

      Path fifthPath = root.getRelative("fifth");
      fifthPath.createDirectory();
      ContainedCopyingSandboxedSpawn fifth =
          nativeSpawn(fifthPath, files, ImmutableMap.of(), deleter);
      fifth.createFileSystem();
      assertThat(
              new String(
                  fifthPath.getRelative("execroot/input.txt").getInputStream().readAllBytes(),
                  UTF_8))
          .isEqualTo("updated");
      fifth.delete();
    } finally {
      deleter.shutdown();
      SandboxStash.initialize(
          stashBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          new SynchronousTreeDeleter());
    }
  }

  @Test
  public void directoryInputsAreNotStashed() throws Exception {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    FileSystem fs =
        new UnixFileSystem(DigestHashFunction.SHA256, "", new NativePosixFilesServiceImpl());
    Path root = fs.getPath(TestUtils.tmpDir()).getRelative("clone-directory-stash-test");
    root.createDirectoryAndParents();
    Path source = root.getRelative("source");
    source.createDirectory();
    Path child = source.getRelative("child.txt");
    FileSystemUtils.writeContent(child, "old".getBytes(UTF_8));
    Path stashBase = root.getRelative("stashes");
    stashBase.createDirectory();
    SandboxStash.initialize(
        stashBase,
        Options.parse(SandboxOptions.class, "--reuse_sandbox_directories").getOptions(),
        new SynchronousTreeDeleter());
    try {
      ImmutableMap<PathFragment, Path> files =
          ImmutableMap.of(PathFragment.create("input"), source);
      Path firstPath = root.getRelative("first");
      firstPath.createDirectory();
      ContainedCopyingSandboxedSpawn first = nativeSpawn(firstPath, files, ImmutableMap.of());
      first.createFileSystem();
      first.delete();
      assertThat(stashBase.getChild(SandboxStash.CLONE_SANDBOX_STASH_BASE).exists()).isFalse();

      FileSystemUtils.writeContent(child, "new".getBytes(UTF_8));
      Path secondPath = root.getRelative("second");
      secondPath.createDirectory();
      ContainedCopyingSandboxedSpawn second = nativeSpawn(secondPath, files, ImmutableMap.of());
      second.createFileSystem();
      assertThat(
              new String(
                  secondPath.getRelative("execroot/input/child.txt").getInputStream().readAllBytes(),
                  UTF_8))
          .isEqualTo("new");
      second.delete();
    } finally {
      SandboxStash.initialize(
          stashBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          new SynchronousTreeDeleter());
    }
  }

  @Test
  public void recreatesMovedOutputDirectoryWithoutRecloningInputs() throws Exception {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    NativePosixFilesServiceImpl nativeFiles = new NativePosixFilesServiceImpl();
    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, "", nativeFiles);
    Path root = fs.getPath(TestUtils.tmpDir()).getRelative("clone-output-stash-test");
    root.createDirectoryAndParents();
    Path source = root.getRelative("source.txt");
    FileSystemUtils.writeContent(source, "input".getBytes(UTF_8));
    Path stashBase = root.getRelative("stashes");
    stashBase.createDirectory();
    SandboxStash.initialize(
        stashBase,
        Options.parse(SandboxOptions.class, "--reuse_sandbox_directories").getOptions(),
        new SynchronousTreeDeleter());
    try {
      ImmutableMap<PathFragment, Path> files =
          ImmutableMap.of(PathFragment.create("input.txt"), source);
      SandboxOutputs outputs =
          SandboxOutputs.create(
              ImmutableSet.of(), ImmutableSet.of(PathFragment.create("test.outputs")));
      Path firstPath = root.getRelative("first");
      firstPath.createDirectory();
      ContainedCopyingSandboxedSpawn first =
          nativeSpawn(firstPath, files, ImmutableMap.of(), outputs);
      first.createFileSystem();
      long originalInode = inode(nativeFiles, firstPath.getRelative("execroot/input.txt"));
      firstPath.getRelative("execroot/test.outputs").renameTo(root.getRelative("moved-output"));
      first.delete();

      Path secondPath = root.getRelative("second");
      secondPath.createDirectory();
      ContainedCopyingSandboxedSpawn second =
          nativeSpawn(secondPath, files, ImmutableMap.of(), outputs);
      second.createFileSystem();
      assertThat(secondPath.getRelative("execroot/test.outputs").isDirectory()).isTrue();
      assertThat(inode(nativeFiles, secondPath.getRelative("execroot/input.txt")))
          .isEqualTo(originalInode);
      second.delete();
    } finally {
      SandboxStash.initialize(
          stashBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          new SynchronousTreeDeleter());
    }
  }

  @Test
  public void failedSetupDoesNotReturnClaimedCloneToStash() throws Exception {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    NativePosixFilesServiceImpl nativeFiles = new NativePosixFilesServiceImpl();
    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, "", nativeFiles);
    Path root = fs.getPath(TestUtils.tmpDir()).getRelative("failed-clone-stash-test");
    root.createDirectoryAndParents();
    Path source = root.getRelative("source.txt");
    FileSystemUtils.writeContent(source, "input".getBytes(UTF_8));
    Path stashBase = root.getRelative("stashes");
    stashBase.createDirectory();
    SandboxStash.initialize(
        stashBase,
        Options.parse(SandboxOptions.class, "--reuse_sandbox_directories").getOptions(),
        new SynchronousTreeDeleter());
    try {
      ImmutableMap<PathFragment, Path> files =
          ImmutableMap.of(PathFragment.create("input.txt"), source);
      Path firstPath = root.getRelative("first");
      firstPath.createDirectory();
      ContainedCopyingSandboxedSpawn first = nativeSpawn(firstPath, files, ImmutableMap.of());
      first.createFileSystem();
      long originalInode = inode(nativeFiles, firstPath.getRelative("execroot/input.txt"));
      first.delete();

      Path failedPath = root.getRelative("failed");
      failedPath.createDirectory();
      ContainedCopyingSandboxedSpawn failed =
          new ContainedCopyingSandboxedSpawn(
              failedPath,
              failedPath.getRelative("execroot"),
              ImmutableList.of("/bin/true"),
              ImmutableMap.of(),
              new SandboxInputs(files, ImmutableMap.of(), ImmutableMap.of()),
              SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
              ImmutableSet.of(),
              new SynchronousTreeDeleter(),
              /* statisticsPath= */ null,
              "Test",
              /* targetLabel= */ null) {
            @Override
            void createInputs(Iterable<PathFragment> inputsToCreate, SandboxInputs inputs)
                throws IOException, InterruptedException {
              super.createInputs(inputsToCreate, inputs);
              throw new IOException("simulated failure after claiming a clone stash");
            }
          };
      assertThrows(IOException.class, failed::createFileSystem);
      failed.delete();

      Path nextPath = root.getRelative("next");
      nextPath.createDirectory();
      ContainedCopyingSandboxedSpawn next = nativeSpawn(nextPath, files, ImmutableMap.of());
      next.createFileSystem();
      assertThat(inode(nativeFiles, nextPath.getRelative("execroot/input.txt")))
          .isNotEqualTo(originalInode);
      next.delete();
    } finally {
      SandboxStash.initialize(
          stashBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          new SynchronousTreeDeleter());
    }
  }

  @Test
  public void reusedCloneCreatesNewOutputParentsAfterRemovingScratch() throws Exception {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    NativePosixFilesServiceImpl nativeFiles = new NativePosixFilesServiceImpl();
    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, "", nativeFiles);
    Path root = fs.getPath(TestUtils.tmpDir()).getRelative("changed-clone-outputs-test");
    root.createDirectoryAndParents();
    Path source = root.getRelative("source.txt");
    FileSystemUtils.writeContent(source, "input".getBytes(UTF_8));
    Path stashBase = root.getRelative("stashes");
    stashBase.createDirectory();
    SandboxStash.initialize(
        stashBase,
        Options.parse(SandboxOptions.class, "--reuse_sandbox_directories").getOptions(),
        new SynchronousTreeDeleter());
    try {
      ImmutableMap<PathFragment, Path> files =
          ImmutableMap.of(PathFragment.create("input.txt"), source);
      Path firstPath = root.getRelative("first");
      firstPath.createDirectory();
      ContainedCopyingSandboxedSpawn first =
          nativeSpawn(
              firstPath,
              files,
              ImmutableMap.of(),
              SandboxOutputs.create(
                  ImmutableSet.of(PathFragment.create("old/out.txt")), ImmutableSet.of()));
      first.createFileSystem();
      long originalInode = inode(nativeFiles, firstPath.getRelative("execroot/input.txt"));
      firstPath.getRelative("execroot/new/child").createDirectoryAndParents();
      first.delete();

      Path secondPath = root.getRelative("second");
      secondPath.createDirectory();
      ContainedCopyingSandboxedSpawn second =
          nativeSpawn(
              secondPath,
              files,
              ImmutableMap.of(),
              SandboxOutputs.create(
                  ImmutableSet.of(PathFragment.create("new/child/out.txt")), ImmutableSet.of()));
      second.createFileSystem();
      assertThat(secondPath.getRelative("execroot/new/child").isDirectory()).isTrue();
      assertThat(inode(nativeFiles, secondPath.getRelative("execroot/input.txt")))
          .isEqualTo(originalInode);
      second.delete();
    } finally {
      SandboxStash.initialize(
          stashBase,
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          new SynchronousTreeDeleter());
    }
  }

  private static long inode(NativePosixFilesServiceImpl nativeFiles, Path path)
      throws NativePosixFilesException {
    return nativeFiles.lstat(path.getPathString(), StatErrorHandling.ALWAYS_THROW).ino();
  }

  private static ContainedCopyingSandboxedSpawn nativeSpawn(
      Path path,
      ImmutableMap<PathFragment, Path> files,
      ImmutableMap<PathFragment, PathFragment> symlinks) {
    return nativeSpawn(path, files, symlinks, new SynchronousTreeDeleter());
  }

  private static ContainedCopyingSandboxedSpawn nativeSpawn(
      Path path,
      ImmutableMap<PathFragment, Path> files,
      ImmutableMap<PathFragment, PathFragment> symlinks,
      TreeDeleter treeDeleter) {
    return nativeSpawn(
        path,
        files,
        symlinks,
        SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
        treeDeleter);
  }

  private static ContainedCopyingSandboxedSpawn nativeSpawn(
      Path path,
      ImmutableMap<PathFragment, Path> files,
      ImmutableMap<PathFragment, PathFragment> symlinks,
      SandboxOutputs outputs) {
    return nativeSpawn(path, files, symlinks, outputs, new SynchronousTreeDeleter());
  }

  private static ContainedCopyingSandboxedSpawn nativeSpawn(
      Path path,
      ImmutableMap<PathFragment, Path> files,
      ImmutableMap<PathFragment, PathFragment> symlinks,
      SandboxOutputs outputs,
      TreeDeleter treeDeleter) {
    return new ContainedCopyingSandboxedSpawn(
        path,
        path.getRelative("execroot"),
        ImmutableList.of("/bin/true"),
        ImmutableMap.of(),
        new SandboxInputs(files, ImmutableMap.of(), symlinks),
        outputs,
        ImmutableSet.of(),
        treeDeleter,
        /* statisticsPath= */ null,
        "Test",
        /* targetLabel= */ null);
  }
}
