// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.zip;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.xcode.util.Value;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

/**
 * Describes an entry in a zip file when the zip file is being created.
 */
public class ZipInputEntry extends Value<ZipInputEntry> {
  private final Path source;
  private final String zipPath;

  public ZipInputEntry(Path source, String zipPath) {
    super(source, zipPath);
    this.source = source;
    this.zipPath = zipPath;
  }

  /**
   * The location of the source file to place in the zip.
   */
  public Path getSource() {
    return source;
  }

  /**
   * The full path of the item in the zip.
   */
  public String getZipPath() {
    return zipPath;
  }

  /**
   * Adds this entry to a zip using the given {@code ZipCombiner}.
   */
  public void add(ZipCombiner combiner) throws IOException {
    try (InputStream inputStream = Files.newInputStream(source)) {
      combiner.addFile(zipPath, ZipCombiner.DOS_EPOCH, inputStream);
    }
  }

  public static void addAll(ZipCombiner combiner, Iterable<ZipInputEntry> inputs)
      throws IOException {
    for (ZipInputEntry input : inputs) {
      input.add(combiner);
    }
  }

  /**
   * Returns the entries to place in a zip file as if the zip file mirrors some directory structure.
   * For instance, if {@code rootDirectory} is /tmp/foo, and the following files exist:
   * <ul>
   *   <li>/tmp/foo/a
   *   <li>/tmp/foo/bar/c
   *   <li>/tmp/foo/baz/d
   * </ul>
   * This function will return entries which point to these files and have in-zip paths of:
   * <ul>
   *   <li>a
   *   <li>bar/c
   *   <li>baz/d
   * </ul>
   */
  public static Iterable<ZipInputEntry> fromDirectory(final Path rootDirectory) throws IOException {
    final ImmutableList.Builder<ZipInputEntry> zipInputs = new ImmutableList.Builder<>();
    Files.walkFileTree(rootDirectory, new SimpleFileVisitor<Path>() {
      @Override
      public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
        zipInputs.add(new ZipInputEntry(file, rootDirectory.relativize(file).toString()));
        return FileVisitResult.CONTINUE;
      }
    });
    return zipInputs.build();
  }
}
