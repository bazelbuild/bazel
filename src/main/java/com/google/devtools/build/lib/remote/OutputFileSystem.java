// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static java.util.Collections.binarySearch;

import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkNode;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.ReadonlyFileSystem;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

class OutputFileSystem extends ReadonlyFileSystem {
  private final Set<String> outputFiles;
  private final Map<String, Directory> outputDirectories;
  private final Map<Digest, Directory> children;

  OutputFileSystem(
      DigestHashFunction digestFunction,
      Iterable<String> outputFiles,
      Map<String, Directory> outputDirectories,
      Map<Digest, Directory> children) {
    super(digestFunction);
    this.outputFiles = Sets.newHashSet(outputFiles);
    this.outputDirectories = outputDirectories;
    this.children = children;
  }

  @Override
  protected boolean isExecutable(Path path) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected boolean isWritable(Path path) {
    return false;
  }

  @Override
  protected boolean isReadable(Path path) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected Collection<String> getDirectoryEntries(Path path) {
    throw new UnsupportedOperationException();
  }

  private Directory findDirectory(String name, List<DirectoryNode> directories) {
    DirectoryNode key = DirectoryNode.newBuilder().setName(name).build();
    int index = binarySearch(directories, key, new Comparator<DirectoryNode>() {
      @Override
      public int compare(DirectoryNode o1, DirectoryNode o2) {
        return o1.getName().compareTo(o2.getName());
      }

      @Override
      public boolean equals(Object o) {
        return this == o;
      }
    });
    if (index >= 0) {
      return children.get(directories.get(index).getDigest());
    }
    return null;
  }

  private boolean containsFile(String name, List<FileNode> files) {
    FileNode key = FileNode.newBuilder().setName(name).build();
    return binarySearch(files, key, new Comparator<FileNode>() {
      @Override
      public int compare(FileNode o1, FileNode o2) {
        return o1.getName().compareTo(o2.getName());
      }

      @Override
      public boolean equals(Object o) {
        return this == o;
      }
    }) >= 0;
  }

  private boolean containsSymlink(String name, List<SymlinkNode> symlinks) {
    SymlinkNode key = SymlinkNode.newBuilder().setName(name).build();
    return binarySearch(symlinks, key, new Comparator<SymlinkNode>() {
      @Override
      public int compare(SymlinkNode o1, SymlinkNode o2) {
        return o1.getName().compareTo(o2.getName());
      }

      @Override
      public boolean equals(Object o) {
        return this == o;
      }
    }) >= 0;
  }

  private boolean directoryContains(Directory directory, boolean matchesDirectory, Iterable<String> path) {
    for (String name : path) {
      if (directory == null) {
        return false;
      }
      Directory child = findDirectory(name, directory.getDirectoriesList());
      if (child == null) {
        return containsFile(name, directory.getFilesList())
            || containsSymlink(name, directory.getSymlinksList());
      }
      directory = child;
    }
    return matchesDirectory;
  }

  private boolean outputDirectoriesContains(PathFragment fragment, boolean matchesDirectory, boolean followSymlinks) {
    // symlinks?
    Stack<String> relative = new Stack<>();
    while (!fragment.toString().isEmpty()) {
      relative.push(fragment.getBaseName());
      fragment = fragment.getParentDirectory();
      if (fragment == null) {
        return false;
      }
      Directory directory = outputDirectories.get(fragment.toString());
      if (directory != null) {
        return directoryContains(directory, matchesDirectory, relative);
      }
    }
    return false;
  }

  private boolean contains(Path path, boolean matchesDirectory, boolean followSymlinks) {
    PathFragment fragment = path.relativeTo(getPath("/"));
    return outputFiles.contains(fragment.toString())
        || outputDirectoriesContains(fragment, matchesDirectory, followSymlinks);
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    return contains(path, /* matchesDirectory=*/ true, followSymlinks);
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    return false;
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    return contains(path, /* matchesDirectory=*/ false, followSymlinks);
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) {
    throw new UnsupportedOperationException();
  }
}
