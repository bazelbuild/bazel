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

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.DosFileAttributeView;

/**
 * Creates a temporary directory that will be deleted once a scope closes. NOTE: If an error occurs
 * during deletion, it will just stop rather than try an continue.
 */
final class ScopedTemporaryDirectory extends SimpleFileVisitor<Path> implements Closeable {

  private final Path path;

  public ScopedTemporaryDirectory(String prefix) throws IOException {
    this.path = Files.createTempDirectory(prefix);
  }

  public Path getPath() {
    return this.path;
  }

  @Override
  public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
    // Make the file deletable on Windows.
    // Setting this attribute on other platforms than Windows has no effect.
    DosFileAttributeView dosAttribs = Files.getFileAttributeView(path, DosFileAttributeView.class);
    if (dosAttribs != null) {
      dosAttribs.setReadOnly(false);
    }
    Files.delete(file);
    return FileVisitResult.CONTINUE;
  }

  @Override
  public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
    Files.delete(dir);
    return FileVisitResult.CONTINUE;
  }

  @Override
  public void close() throws IOException {
    Files.walkFileTree(path, this);
  }
}
