// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.resourcejar;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Constructs a jar file of Java resources. */
public class ResourceJarBuilder implements Closeable {

  public static void main(String[] args) throws Exception {
    build(ResourceJarOptionsParser.parse(Arrays.asList(args)));
  }

  public static void build(ResourceJarOptions options) throws Exception {
    try (ResourceJarBuilder builder = new ResourceJarBuilder(options)) {
      builder.build();
    }
  }

  /** Cache of opened zip filesystems. */
  private final Map<Path, FileSystem> filesystems = new HashMap<>();

  private final ResourceJarOptions options;

  private ResourceJarBuilder(ResourceJarOptions options) {
    this.options = options;
  }

  public void build() throws IOException {
    final JarCreator jar = new JarCreator(options.output());
    jar.setNormalize(true);
    jar.setCompression(true);

    addResourceJars(jar, options.resourceJars());
    jar.addRootEntries(options.classpathResources());
    addResourceEntries(jar, options.resources());
    addMessageEntries(jar, options.messages());

    jar.execute();
  }

  private void addResourceJars(final JarCreator jar, ImmutableList<String> resourceJars)
      throws IOException {
    for (String resourceJar : resourceJars) {
      for (final Path root : getJarFileSystem(Paths.get(resourceJar)).getRootDirectories()) {
        Files.walkFileTree(
            root,
            new SimpleFileVisitor<Path>() {
              @Override
              public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
                  throws IOException {
                // TODO(b/28452451): omit directories entries from jar files
                if (dir.getNameCount() > 0) {
                  jar.addEntry(root.relativize(dir).toString(), dir);
                }
                return FileVisitResult.CONTINUE;
              }

              @Override
              public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                  throws IOException {
                jar.addEntry(root.relativize(path).toString(), path);
                return FileVisitResult.CONTINUE;
              }
            });
      }
    }
  }

  /**
   * Adds a collection of resource entries. Each entry is a string composed of a pair of parts
   * separated by a colon ':'. The name of the resource comes from the second part, and the path to
   * the resource comes from the whole string with the colon replaced by a slash '/'.
   *
   * <pre>
   * prefix:name => (name, prefix/name)
   * </pre>
   */
  private static void addResourceEntries(JarCreator jar, Collection<String> resources)
      throws IOException {
    for (String resource : resources) {
      int colon = resource.indexOf(':');
      if (colon < 0) {
        throw new IOException("" + resource + ": Illegal resource entry.");
      }
      String prefix = resource.substring(0, colon);
      String name = resource.substring(colon + 1);
      String path = colon > 0 ? prefix + "/" + name : name;
      addEntryWithParents(jar, name, path);
    }
  }

  private static void addMessageEntries(JarCreator jar, List<String> messages) throws IOException {
    for (String message : messages) {
      int colon = message.indexOf(':');
      if (colon < 0) {
        throw new IOException("" + message + ": Illegal message entry.");
      }
      String prefix = message.substring(0, colon);
      String name = message.substring(colon + 1);
      String path = colon > 0 ? prefix + "/" + name : name;
      File messageFile = new File(path);
      // Ignore empty messages. They get written by the translation importer
      // when there is no translation for a particular language.
      if (messageFile.length() != 0L) {
        addEntryWithParents(jar, name, path);
      }
    }
  }

  /**
   * Adds an entry to the jar, making sure that all the parent dirs up to the base of {@code entry}
   * are also added.
   *
   * @param entry the PathFragment of the entry going into the Jar file
   * @param file the PathFragment of the input file for the entry
   */
  @VisibleForTesting
  static void addEntryWithParents(JarCreator creator, String entry, String file) {
    while ((entry != null) && creator.addEntry(entry, file)) {
      entry = new File(entry).getParent();
      file = new File(file).getParent();
    }
  }

  private FileSystem getJarFileSystem(Path sourceJar) throws IOException {
    FileSystem fs = filesystems.get(sourceJar);
    if (fs == null) {
      filesystems.put(sourceJar, fs = FileSystems.newFileSystem(sourceJar, null));
    }
    return fs;
  }

  @Override
  public void close() throws IOException {
    for (FileSystem fs : filesystems.values()) {
      fs.close();
    }
  }
}
