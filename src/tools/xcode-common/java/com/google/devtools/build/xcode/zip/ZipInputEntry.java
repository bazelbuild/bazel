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

import static com.google.devtools.build.singlejar.ZipCombiner.DOS_EPOCH;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.xcode.util.Value;
import com.google.devtools.build.zip.ZipFileEntry;

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
  /**
   * The external file attribute used for files by default. This indicates a non-executable regular
   * file that is readable by group and world.
   */
  public static final int DEFAULT_EXTERNAL_FILE_ATTRIBUTE = (0100644 << 16);

  /**
   * An external file attribute that indicates an executable regular file that is readable and
   * executable by group and world.
   */
  public static final int EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE = (0100755 << 16);

  /**
   * Made by version of .ipa files built by Xcode. Upper byte indicates Unix host. Lower byte
   * indicates version of encoding software (note that 0x1e = 30 = (3.0 * 10), so 0x1e translates
   * to 3.0). The Unix host value in the upper byte is what causes the external file attribute to
   * be interpreted as POSIX permission and file type bits.
   */
  public static final short MADE_BY_VERSION = (short) 0x031e;

  private final Path source;
  private final String zipPath;
  private final int externalFileAttribute;

  public ZipInputEntry(Path source, String zipPath) {
    this(source, zipPath, DEFAULT_EXTERNAL_FILE_ATTRIBUTE);
  }

  public ZipInputEntry(Path source, String zipPath, int externalFileAttribute) {
    super(source, zipPath, externalFileAttribute);
    this.source = source;
    this.zipPath = zipPath;
    this.externalFileAttribute = externalFileAttribute;
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
   * The external file attribute field of the zip entry in the central directory record. On
   * Unix-originated .zips, this corresponds to the permission bits (e.g. 0755 for an excutable
   * file).
   */
  public int getExternalFileAttribute() {
    return externalFileAttribute;
  }

  /**
   * Adds this entry to a zip using the given {@code ZipCombiner}.
   */
  public void add(ZipCombiner combiner) throws IOException {
    try (InputStream inputStream = Files.newInputStream(source)) {
      ZipFileEntry entry = new ZipFileEntry(zipPath);
      entry.setTime(DOS_EPOCH.getTime());
      entry.setVersion(MADE_BY_VERSION);
      entry.setExternalAttributes(externalFileAttribute);
      combiner.addFile(entry, inputStream);
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
        // TODO(bazel-team): Set the external file attribute based on the attribute of the
        // permissions of the file on-disk.
        zipInputs.add(new ZipInputEntry(file, rootDirectory.relativize(file).toString()));
        return FileVisitResult.CONTINUE;
      }
    });
    return zipInputs.build();
  }
}
