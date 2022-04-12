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
package com.google.devtools.build.android;

import static java.nio.file.attribute.PosixFilePermission.OWNER_EXECUTE;
import static java.nio.file.attribute.PosixFilePermission.OWNER_READ;
import static java.nio.file.attribute.PosixFilePermission.OWNER_WRITE;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.FileStore;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.DosFileAttributeView;
import java.nio.file.attribute.PosixFileAttributeView;
import java.util.EnumSet;

/**
 * Creates a temporary directory that will be deleted once a scope closes. NOTE: errors during
 * deletion are ignored, which can lead to inclomplete clean up of the temporary files. However, as
 * they are created in the temp location, the system should eventually clean them up.
 */
final class ScopedTemporaryDirectory extends SimpleFileVisitor<Path> implements Closeable {

  private static final boolean IS_WINDOWS = System.getProperty("os.name").startsWith("Windows");

  private final Path path;

  public ScopedTemporaryDirectory(String prefix) throws IOException {
    this.path = Files.createTempDirectory(prefix);
  }

  public Path getPath() {
    return this.path;
  }

  public Path subDirectoryOf(String... directories) throws IOException {
    Path sub = this.path;
    for (String directory : directories) {
      sub = sub.resolve(directory);
    }
    return Files.createDirectories(sub);
  }

  private void makeWritable(Path file) throws IOException {
    FileStore fileStore = Files.getFileStore(file);
    if (IS_WINDOWS && fileStore.supportsFileAttributeView(DosFileAttributeView.class)) {
      DosFileAttributeView dosAttribs =
          Files.getFileAttributeView(file, DosFileAttributeView.class);
      if (dosAttribs != null) {
        dosAttribs.setReadOnly(false);
      }
    } else if (fileStore.supportsFileAttributeView(PosixFileAttributeView.class)) {
      PosixFileAttributeView posixAttribs =
          Files.getFileAttributeView(file, PosixFileAttributeView.class);
      if (posixAttribs != null) {
        posixAttribs.setPermissions(EnumSet.of(OWNER_READ, OWNER_WRITE, OWNER_EXECUTE));
      }
    }
  }

  @Override
  public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
    if (IS_WINDOWS) {
      makeWritable(dir);
    }
    return FileVisitResult.CONTINUE;
  }

  @Override
  public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
    if (IS_WINDOWS) {
      makeWritable(file);
    }
    Files.delete(file);
    return FileVisitResult.CONTINUE;
  }

  @Override
  public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
    try {
      Files.delete(dir);
    } catch (DirectoryNotEmptyException e) {
      // Ignore.
    }
    return FileVisitResult.CONTINUE;
  }

  @Override
  public void close() throws IOException {
    Files.walkFileTree(path, this);
  }
}
