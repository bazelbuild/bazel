package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SymlinkedSandboxedSpawn;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

final class WorkerExecRoot extends SymlinkedSandboxedSpawn {

  private final Path workDir;
  private final Set<PathFragment> workerFiles;

  public WorkerExecRoot(Path workDir, Map<PathFragment, Path> inputs, Collection<PathFragment> outputs, Set<PathFragment> workerFiles) {
    super(workDir, workDir, ImmutableList.of(), ImmutableMap.of(), inputs, outputs, ImmutableSet.of());
    this.workDir = workDir;
    this.workerFiles = workerFiles;
  }

  @Override
  public void createFileSystem() throws IOException {
    workDir.createDirectoryAndParents();
    deleteExceptAllowedFiles(workDir, workerFiles);
    super.createFileSystem();
  }

  private void deleteExceptAllowedFiles(Path root, Set<PathFragment> allowedFiles)
      throws IOException {
    for (Path p : root.getDirectoryEntries()) {
      FileStatus stat = p.stat(Symlinks.NOFOLLOW);
      if (!stat.isDirectory()) {
        if (!allowedFiles.contains(p.relativeTo(workDir))) {
          p.delete();
        }
      } else {
        deleteExceptAllowedFiles(p, allowedFiles);
        if (p.readdir(Symlinks.NOFOLLOW).isEmpty()) {
          p.delete();
        }
      }
    }
  }

  @Override
  protected void createInputs(Map<PathFragment, Path> inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path key = workDir.getRelative(entry.getKey());
      FileStatus keyStat = key.statNullable(Symlinks.NOFOLLOW);
      if (keyStat != null) {
        if (keyStat.isSymbolicLink()
            && entry.getValue() != null
            && key.readSymbolicLink().equals(entry.getValue().asFragment())) {
          continue;
        }
        key.delete();
      }
      // A null value means that we're supposed to create an empty file as the input.
      if (entry.getValue() != null) {
        key.createSymbolicLink(entry.getValue());
      } else {
        FileSystemUtils.createEmptyFile(key);
      }
    }
  }
}
