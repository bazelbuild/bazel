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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.unix.NativePosixFilesException;
import com.google.devtools.build.lib.unix.NativePosixFilesService.Stat;
import com.google.devtools.build.lib.unix.NativePosixFilesService.StatErrorHandling;
import com.google.devtools.build.lib.unix.NativePosixFilesServiceImpl;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/** Copies Darwin sandbox inputs without creating links back to the host execroot. */
class ContainedCopyingSandboxedSpawn extends CopyingSandboxedSpawn {
  private static final NativePosixFilesServiceImpl NATIVE_FILES = new NativePosixFilesServiceImpl();
  private static final int CLONE_WORKERS = 8;
  private static final int CLONE_CHUNK_SIZE = 64;
  private static final ExecutorService CLONE_POOL =
      Executors.newFixedThreadPool(
          CLONE_WORKERS, Thread.ofVirtual().name("sandbox-clone-", 0).factory());
  private final List<Path> clonedDirectories = Collections.synchronizedList(new ArrayList<>());
  private final Map<PathFragment, SourceStamp> clonedSources =
      Collections.synchronizedMap(new HashMap<>());
  private final Map<PathFragment, SourceStamp> retainedSources = new HashMap<>();
  @Nullable private CloneSnapshot snapshot;
  private final boolean trackForReuse;
  private final String mnemonic;
  @Nullable private final Label targetLabel;

  ContainedCopyingSandboxedSpawn(
      Path sandboxPath,
      Path sandboxExecRoot,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      SandboxInputs inputs,
      SandboxOutputs outputs,
      Set<Path> writableDirs,
      TreeDeleter treeDeleter,
      @Nullable Path statisticsPath,
      String mnemonic,
      @Nullable Label targetLabel) {
    super(
        sandboxPath,
        sandboxExecRoot,
        arguments,
        environment,
        inputs,
        outputs,
        writableDirs,
        treeDeleter,
        /* sandboxDebugPath= */ null,
        statisticsPath,
        () -> {},
        mnemonic);
    this.mnemonic = mnemonic;
    this.targetLabel = targetLabel;
    this.trackForReuse = SandboxStash.gotInstance() && SandboxStash.useInMemoryStashes();
  }

  /** In-memory metadata for the inputs actually cloned into a stash. */
  static final class CloneSnapshot {
    private final Map<PathFragment, Path> files;
    private final Map<PathFragment, PathFragment> symlinks;
    private final Map<PathFragment, SourceStamp> sources;
    private final Map<PathFragment, NodeStamp> staged;

    CloneSnapshot(
        Map<PathFragment, Path> files,
        Map<PathFragment, PathFragment> symlinks,
        Map<PathFragment, SourceStamp> sources,
        Map<PathFragment, NodeStamp> staged) {
      this.files = new HashMap<>(files);
      this.symlinks = new HashMap<>(symlinks);
      this.sources = new HashMap<>(sources);
      this.staged = new HashMap<>(staged);
    }
  }

  private record FileStamp(int mode, long mtime, long ctime, long size, long inode) {
    static FileStamp of(Stat stat) {
      return new FileStamp(
          stat.mode(), stat.mtimeNanos(), stat.ctimeNanos(), stat.size(), stat.ino());
    }
  }

  private record SourceStamp(Path path, FileStamp stamp) {}

  private record NodeStamp(FileStamp stamp, @Nullable String link) {
    boolean matches(NodeStamp current, boolean directory) {
      // Output creation and removal changes ancestor directory times, but not their entries.
      return directory
          ? stamp.mode == current.stamp.mode && stamp.inode == current.stamp.inode
          : equals(current);
    }
  }

  @Override
  protected boolean recordSandboxStashTimestamp() {
    return false;
  }

  @Override
  public void filterInputsAndDirsToCreate(
      Set<PathFragment> inputsToCreate, Set<PathFragment> dirsToCreate)
      throws IOException, InterruptedException {
    if (!trackForReuse) {
      return;
    }
    CloneSnapshot candidate =
        SandboxStash.takeClonedSandbox(sandboxPath, mnemonic, outputs, targetLabel);
    if (candidate == null) {
      return;
    }
    Set<PathFragment> changedInputs = new HashSet<>();
    Set<PathFragment> dirsToRecreate = new HashSet<>();
    if (matches(candidate, dirsToCreate, changedInputs, dirsToRecreate)) {
      if (changedInputs.isEmpty()) {
        snapshot = candidate;
      } else {
        retainedSources.putAll(candidate.sources);
      }
      inputsToCreate.retainAll(changedInputs);
      dirsToCreate.clear();
      dirsToCreate.addAll(dirsToRecreate);
    } else {
      // Never use a partially validated tree as a base for another action.
      treeDeleter.deleteTree(sandboxExecRoot.getParentDirectory());
    }
  }

  private boolean matches(
      CloneSnapshot candidate,
      Set<PathFragment> requiredDirs,
      Set<PathFragment> changedInputs,
      Set<PathFragment> dirsToRecreate)
      throws InterruptedException {
    if (!candidate.files.equals(inputs.getFiles())
        || !candidate.symlinks.equals(inputs.getSymlinks())) {
      return false;
    }
    try {
      for (var entry : candidate.sources.entrySet()) {
        if (Thread.interrupted()) {
          throw new InterruptedException("Interrupted checking cloned sandbox sources");
        }
        SourceStamp source = entry.getValue();
        if (!source.path.equals(inputs.getFiles().get(entry.getKey()))
            || !source.stamp.equals(stamp(source.path, /* follow= */ true))) {
          changedInputs.add(entry.getKey());
        }
      }
      Map<PathFragment, NodeStamp> actual = new HashMap<>();
      scan(sandboxExecRoot, PathFragment.EMPTY_FRAGMENT, actual);
      for (var entry : candidate.staged.entrySet()) {
        PathFragment path = entry.getKey();
        NodeStamp current = actual.get(path);
        if (current == null) {
          if (outputs.dirs().containsValue(path) && isDirectory(entry.getValue().stamp.mode)) {
            dirsToRecreate.add(path);
          } else if (isInput(path)) {
            changedInputs.add(path);
          } else {
            return false;
          }
          continue;
        }
        boolean directory = isDirectory(entry.getValue().stamp.mode);
        if (!entry.getValue().matches(current, directory)) {
          if (isInput(path) && !directory) {
            changedInputs.add(path);
          } else {
            return false;
          }
        }
      }
      for (PathFragment path : requiredDirs) {
        NodeStamp current = actual.get(path);
        if (current == null || !candidate.staged.containsKey(path)) {
          dirsToRecreate.add(path);
        } else if (!isDirectory(current.stamp.mode)) {
          return false;
        }
      }
      Set<PathFragment> extras = new HashSet<>(actual.keySet());
      extras.removeAll(candidate.staged.keySet());
      for (PathFragment changed : changedInputs) {
        if (actual.containsKey(changed)) {
          deleteStagedPath(sandboxExecRoot.getRelative(changed));
        }
      }
      for (PathFragment extra : extras) {
        if (extra.isEmpty() || extras.contains(extra.getParentDirectory())) {
          continue;
        }
        deleteStagedPath(sandboxExecRoot.getRelative(extra));
      }
      return true;
    } catch (IOException e) {
      return false;
    }
  }

  private boolean isInput(PathFragment path) {
    return inputs.getFiles().containsKey(path) || inputs.getSymlinks().containsKey(path);
  }

  private void deleteStagedPath(Path path) throws IOException {
    // The async tree deleter's existence check follows symlinks, skipping dangling ones.
    if (path.isSymbolicLink()) {
      path.delete();
    } else {
      treeDeleter.deleteTree(path);
    }
  }

  @Override
  public void createFileSystem() throws IOException, InterruptedException {
    try {
      super.createFileSystem();
      if (trackForReuse && snapshot == null && clonedDirectories.isEmpty()) {
        Map<PathFragment, NodeStamp> staged = new HashMap<>();
        try {
          scan(sandboxExecRoot, PathFragment.EMPTY_FRAGMENT, staged);
          // An absolute link to the old sandbox would stop resolving after the stash is renamed.
          if (staged.values().stream()
              .noneMatch(node -> node.link != null && node.link.startsWith("/"))) {
            retainedSources.putAll(clonedSources);
            snapshot =
                new CloneSnapshot(inputs.getFiles(), inputs.getSymlinks(), retainedSources, staged);
          }
        } catch (IOException e) {
          // Stash eligibility must not make an otherwise valid action fail.
        }
      }
    } catch (IOException | InterruptedException | RuntimeException | Error e) {
      snapshot = null;
      throw e;
    }
  }

  @Override
  public void delete() {
    if (snapshot != null) {
      SandboxStash.stashClonedSandbox(
          sandboxPath,
          mnemonic,
          outputs,
          treeDeleter,
          targetLabel,
          snapshot);
    }
    super.delete();
  }

  private static boolean isDirectory(int mode) {
    return (mode & 0170000) == 0040000;
  }

  private FileStamp stamp(Path path, boolean follow) throws IOException {
    try {
      return FileStamp.of(
          follow
              ? NATIVE_FILES.stat(path.getPathString(), StatErrorHandling.ALWAYS_THROW)
              : NATIVE_FILES.lstat(path.getPathString(), StatErrorHandling.ALWAYS_THROW));
    } catch (NativePosixFilesException e) {
      throw new IOException(e.getMessage(), e);
    }
  }

  private void scan(Path directory, PathFragment prefix, Map<PathFragment, NodeStamp> nodes)
      throws IOException, InterruptedException {
    FileStamp directoryStamp = stamp(directory, /* follow= */ false);
    if (!isDirectory(directoryStamp.mode)) {
      throw new IOException("Sandbox directory changed type: " + directory);
    }
    if (prefix.isEmpty()) {
      nodes.put(prefix, new NodeStamp(directoryStamp, null));
    }
    for (Dirent entry : directory.readdir(Symlinks.NOFOLLOW)) {
      if (Thread.interrupted()) {
        throw new InterruptedException("Interrupted checking cloned sandbox");
      }
      Path child = directory.getChild(entry.getName());
      PathFragment relative = prefix.getRelative(entry.getName());
      FileStamp stat = stamp(child, /* follow= */ false);
      if (entry.getType() != Dirent.Type.FILE
          && entry.getType() != Dirent.Type.DIRECTORY
          && entry.getType() != Dirent.Type.SYMLINK) {
        throw new IOException("Unsupported cloned sandbox entry: " + child);
      }
      String link = entry.getType() == Dirent.Type.SYMLINK ? readLink(child) : null;
      nodes.put(relative, new NodeStamp(stat, link));
      if (entry.getType() == Dirent.Type.DIRECTORY && isDirectory(stat.mode)) {
        scan(child, relative, nodes);
      }
    }
  }

  @Override
  void createInputs(Iterable<PathFragment> inputsToCreate, SandboxInputs inputs)
      throws IOException, InterruptedException {
    List<PathFragment> paths = ImmutableList.copyOf(inputsToCreate);
    if (paths.size() < 256) {
      super.createInputs(paths, inputs);
    } else {
      cloneInputsInParallel(paths, inputs);
    }
    for (var entry : inputs.getSymlinks().entrySet()) {
      if (entry.getValue() != null) {
        checkContained(sandboxExecRoot.getRelative(entry.getKey()));
      }
    }
    for (Path directory : clonedDirectories) {
      validateDirectorySymlinks(directory);
    }
  }

  private void cloneInputsInParallel(List<PathFragment> paths, SandboxInputs inputs)
      throws IOException, InterruptedException {
    AtomicInteger next = new AtomicInteger();
    AtomicBoolean stop = new AtomicBoolean();
    int workerCount = Math.min(CLONE_WORKERS, (paths.size() - 1) / CLONE_CHUNK_SIZE + 1);
    List<Future<?>> workers = new ArrayList<>(workerCount);
    for (int i = 0; i < workerCount; i++) {
      workers.add(
          CLONE_POOL.submit(
              () -> {
                int start;
                while (!stop.get() && (start = next.getAndAdd(CLONE_CHUNK_SIZE)) < paths.size()) {
                  try {
                    super.createInputs(
                        paths.subList(start, Math.min(start + CLONE_CHUNK_SIZE, paths.size())),
                        inputs);
                  } catch (IOException | InterruptedException e) {
                    stop.set(true);
                    throw new RuntimeException(e);
                  }
                }
              }));
    }
    Throwable failure = null;
    boolean interrupted = false;
    for (Future<?> worker : workers) {
      while (true) {
        try {
          worker.get();
          break;
        } catch (InterruptedException e) {
          stop.set(true);
          interrupted = true;
          Thread.interrupted();
        } catch (ExecutionException e) {
          stop.set(true);
          if (failure == null) {
            Throwable cause = e.getCause();
            failure = cause.getCause() != null ? cause.getCause() : cause;
          }
          break;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException("Interrupted cloning sandbox inputs");
    }
    if (failure instanceof IOException e) {
      throw e;
    }
    if (failure instanceof InterruptedException e) {
      throw e;
    }
    if (failure instanceof Error e) {
      throw e;
    }
    if (failure instanceof RuntimeException e) {
      throw e;
    }
    if (failure != null) {
      throw new IOException("Failed to clone sandbox inputs", failure);
    }
  }

  @Override
  protected void copyFile(Path source, Path target) throws IOException {
    FileStamp before = trackForReuse ? stamp(source, /* follow= */ true) : null;
    cloneFile(source, target);
    if (trackForReuse) {
      FileStamp after = stamp(source, /* follow= */ true);
      if (!Objects.equals(before, after)) {
        throw new IOException("Sandbox input changed while cloning: " + source);
      }
      clonedSources.put(target.relativeTo(sandboxExecRoot), new SourceStamp(source, after));
    }
    // Inspect the clone, not the source: a source symlink can change while clonefile runs.
    if (target.isDirectory(Symlinks.NOFOLLOW)) {
      clonedDirectories.add(target);
    }
  }

  private void validateDirectorySymlinks(Path directory) throws IOException {
    for (Dirent entry : directory.readdir(Symlinks.NOFOLLOW)) {
      Path child = directory.getChild(entry.getName());
      switch (entry.getType()) {
        case FILE -> {}
        case DIRECTORY -> validateDirectorySymlinks(child);
        case SYMLINK -> checkContained(child);
        case UNKNOWN -> throw new IOException("Don't know how to copy " + child);
      }
    }
  }

  protected void cloneFile(Path source, Path target) throws IOException {
    try {
      NATIVE_FILES.clonefile(source.getPathString(), target.getPathString());
    } catch (NativePosixFilesException e) {
      throw new IOException(e.getMessage(), e);
    }
  }

  protected String readLink(Path link) throws IOException {
    try {
      // PathFragment.readSymbolicLink() normalizes a/.., which changes symlink semantics.
      return NATIVE_FILES.readlink(link.getPathString());
    } catch (NativePosixFilesException e) {
      throw new IOException(e.getMessage(), e);
    }
  }

  private void checkContained(Path link) throws IOException {
    String destination = readLink(link);
    ArrayDeque<String> remaining = new ArrayDeque<>();
    Path current = addTarget(remaining, destination, link.getParentDirectory(), link);
    int links = 0;
    while (!remaining.isEmpty()) {
      String segment = remaining.removeFirst();
      if (segment.isEmpty() || segment.equals(".")) {
        continue;
      }
      current = segment.equals("..") ? current.getParentDirectory() : current.getChild(segment);
      if (current == null || !current.startsWith(sandboxExecRoot)) {
        throw new IOException("Sandbox input symlink escapes execroot: " + link + " -> " + destination);
      }
      if (current.isSymbolicLink()) {
        if (++links > 40) {
          throw new IOException("Too many sandbox input symlinks resolving " + link);
        }
        String next = readLink(current);
        ArrayDeque<String> prefix = new ArrayDeque<>();
        Path nextParent = addTarget(prefix, next, current.getParentDirectory(), link);
        prefix.addAll(remaining);
        remaining = prefix;
        current = nextParent;
      }
    }
  }

  private Path addTarget(ArrayDeque<String> segments, String target, Path parent, Path link)
      throws IOException {
    if (target.startsWith("/")) {
      String root = sandboxExecRoot.getPathString();
      if (!target.equals(root) && !target.startsWith(root + "/")) {
        throw new IOException("Sandbox input symlink escapes execroot: " + link + " -> " + target);
      }
      addSegments(segments, target.substring(root.length()));
      return sandboxExecRoot;
    }
    addSegments(segments, target);
    return parent;
  }

  private static void addSegments(ArrayDeque<String> segments, String path) {
    for (String segment : path.split("/")) {
      segments.addLast(segment);
    }
  }
}
