// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.integration.blackbox.framework;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.List;

/** Helper class for work with java.nio.file.Path. */
public class PathUtils {
  /** Recursively delete a directory. Does not follow the symbolic links. */
  public static void deleteTree(final Path directory) throws IOException {
    if (Files.exists(directory)) {
      Files.walkFileTree(
          directory,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
                throws IOException {
              Files.delete(file);
              return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc)
                throws IOException {
              Files.delete(dir);
              return FileVisitResult.CONTINUE;
            }
          });
    }
  }

  /**
   * Recursively copy the contents of source into the target; target does not have to exist, it can
   * exist. Does not follow the symbolic links.
   */
  public static void copyTree(final Path source, final Path target) throws IOException {
    if (!Files.exists(source)) {
      throw new IOException(
          String.format(
              "Can not copy: source directory %s does not exist",
              source.toAbsolutePath().toString()));
    }
    Files.createDirectories(target);
    Files.walkFileTree(
        source,
        new SimpleFileVisitor<Path>() {
          Path currentTarget = target;

          @Override
          public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
              throws IOException {
            if (!source.equals(dir)) {
              currentTarget = currentTarget.resolve(dir.getFileName().toString());
              Files.createDirectories(currentTarget);
            }
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            Files.copy(file, currentTarget.resolve(file.getFileName().toString()));
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path dir, IOException exc) {
            currentTarget = currentTarget.getParent();
            return FileVisitResult.CONTINUE;
          }
        });
  }

  /**
   * Creates the file under the <code>directory/parts[0]/parts[1]/.../parts[n]</code>. Will also
   * create all subdirectories, if they do not exist.
   *
   * @param directory directory under which to create the subdirectories tree with a file
   * @param parts parts of the path relative to directory, must be not empty, last element denotes
   *     the file name
   * @return Path to created file
   * @throws IOException in case file or subdirectory can not be created
   */
  public static Path createFile(Path directory, String... parts) throws IOException {
    Preconditions.checkArgument(parts.length > 0);
    Path path = resolve(directory, parts);
    return createFile(path);
  }

  /**
   * Creates the file in <code>path</code> location. Will also create all subdirectories, if they do
   * not exist.
   *
   * @param path location where to create the file
   * @return Path to created file
   * @throws IOException in case file or subdirectory can not be created
   */
  public static Path createFile(Path path) throws IOException {
    Files.createDirectories(path.getParent());
    if (Files.exists(path)) {
      return path;
    }
    Files.createFile(path);
    return path;
  }

  /**
   * Resolves the Path to the file or directory under the <code>
   * directory/parts[0]/parts[1]/.../parts[n]</code>.
   *
   * @param directory root directory for resolve
   * @param parts parts of the path relative to directory
   * @return resolved Path
   */
  public static Path resolve(Path directory, String... parts) {
    Path current = directory;
    for (String part : parts) {
      current = current.resolve(part);
    }
    return current;
  }

  /**
   * Reads the file under the <code>directory/parts[0]/parts[1]/.../parts[n]</code>
   * using ISO_8859_1.
   *
   * @param directory root directory for resolve
   * @param parts parts of the path relative to directory
   * @return the List<String> of lines of the file
   * @throws IOException in case file can not be read
   */
  public static List<String> readFile(Path directory, String... parts) throws IOException {
    return readFile(resolve(directory, parts));
  }

  /**
   * Reads the <code>file</code> using ISO_8859_1.
   *
   * @param file file to read
   * @return the List<String></String> of lines of the file
   * @throws IOException in case file can not be read
   */
  public static List<String> readFile(Path file) throws IOException {
    return Files.readAllLines(file, StandardCharsets.ISO_8859_1);
  }

  /**
   * Writes the file in the <code>path</code> location using ISO_8859_1. Overrides the file if it
   * exists, creates the file if it does not exist.
   *
   * @param path location where to write the file
   * @param lines lines to be written
   * @throws IOException in case file can not be written
   */
  public static void writeFile(Path path, String... lines) throws IOException {
    Files.write(path, Lists.newArrayList(lines), StandardCharsets.ISO_8859_1);
  }

  /**
   * Writes the BUILD file under <code>directory</code> using ISO_8859_1. Overrides the file if it
   * exists, creates the file if it does not exist.
   *
   * @param directory directory to write BUILD file under
   * @param lines lines to be written
   * @throws IOException in case file can not be written
   */
  public static void writeBuild(Path directory, String... lines) throws IOException {
    Path buildFile = createFile(directory, "BUILD");
    writeFile(buildFile, lines);
  }

  /**
   * Replaces the symlink file with the contents of the file it refers to.
   *
   * @param path Path to file that will be replaced. Must be a symlink.
   * @throws IOException if files can not be read or written
   */
  public static void replaceWithSymlinkContents(Path path) throws IOException {
    assertThat(Files.isSymbolicLink(path)).isTrue();
    Path target = Files.readSymbolicLink(path);
    Files.delete(path);
    Files.write(path, Files.readAllBytes(target));
  }

  /**
   * Appends <code>lines</code> of text to the <code>path</code>.
   *
   * @param path path of the file to be appended to
   * @param lines lines of the text
   * @throws IOException if file can not be appended to
   */
  public static void append(Path path, String... lines) throws IOException {
    Files.write(path, Arrays.asList(lines), StandardOpenOption.APPEND);
  }
}
