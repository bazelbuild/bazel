// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.buildjar.instrumentation.JacocoInstrumentationProcessor;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.devtools.build.buildjar.javac.BlazeJavacArguments;
import com.google.devtools.build.buildjar.javac.BlazeJavacMain;
import com.google.devtools.build.buildjar.javac.BlazeJavacResult;
import com.google.devtools.build.buildjar.javac.JavacRunner;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** An implementation of the JavaBuilder that uses in-process javac to compile java files. */
public class SimpleJavaLibraryBuilder implements Closeable {

  /** The name of the protobuf meta file. */
  private static final String PROTOBUF_META_NAME = "protobuf.meta";

  /** Cache of opened zip filesystems for srcjars. */
  private final Map<Path, FileSystem> filesystems = new HashMap<>();

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

  BlazeJavacResult compileSources(JavaLibraryBuildRequest build, JavacRunner javacRunner)
      throws IOException {
    return javacRunner.invokeJavac(build.toBlazeJavacArguments(build.getClassPath()));
  }

  protected void prepareSourceCompilation(JavaLibraryBuildRequest build) throws IOException {
    Path classDirectory = Paths.get(build.getClassDir());
    if (Files.exists(classDirectory)) {
      try {
        // Necessary for local builds in order to discard previous outputs
        cleanupDirectory(classDirectory);
      } catch (IOException e) {
        throw new IOException("Cannot clean output directory '" + classDirectory + "'", e);
      }
    }
    Files.createDirectories(classDirectory);

    setUpSourceJars(build);

    // Create sourceGenDir if necessary.
    if (build.getSourceGenDir() != null) {
      Path sourceGenDir = Paths.get(build.getSourceGenDir());
      if (Files.exists(sourceGenDir)) {
        try {
          cleanupDirectory(sourceGenDir);
        } catch (IOException e) {
          throw new IOException("Cannot clean output directory '" + sourceGenDir + "'", e);
        }
      }
      Files.createDirectories(sourceGenDir);
    }
  }

  public void buildGensrcJar(JavaLibraryBuildRequest build) throws IOException {
    JarCreator jar = new JarCreator(build.getGeneratedSourcesOutputJar());
    try {
      jar.setNormalize(true);
      jar.setCompression(build.compressJar());
      jar.addDirectory(build.getSourceGenDir());
    } finally {
      jar.execute();
    }
  }

  /**
   * Prepares a compilation run and sets everything up so that the source files in the build request
   * can be compiled. Invokes compileSources to do the actual compilation.
   *
   * @param build A JavaLibraryBuildRequest request object describing what to compile
   */
  public BlazeJavacResult compileJavaLibrary(final JavaLibraryBuildRequest build) throws Exception {
    prepareSourceCompilation(build);
    if (build.getSourceFiles().isEmpty()) {
      return BlazeJavacResult.ok();
    }
    JavacRunner javacRunner =
        new JavacRunner() {
          @Override
          public BlazeJavacResult invokeJavac(BlazeJavacArguments arguments) {
            return BlazeJavacMain.compile(arguments);
          }
        };
    BlazeJavacResult result = compileSources(build, javacRunner);
    JacocoInstrumentationProcessor processor = build.getJacocoInstrumentationProcessor();
    if (processor != null) {
      processor.processRequest(build);
    }
    return result;
  }

  /** Perform the build. */
  public BlazeJavacResult run(JavaLibraryBuildRequest build) throws Exception {
    BlazeJavacResult result = BlazeJavacResult.error("");
    try {
      result = compileJavaLibrary(build);
      if (result.isOk()) {
        buildJar(build);
      }
      if (!build.getProcessors().isEmpty()) {
        if (build.getGeneratedSourcesOutputJar() != null) {
          buildGensrcJar(build);
        }
      }
    } finally {
      build.getDependencyModule().emitDependencyInformation(build.getClassPath(), result.isOk());
      build.getProcessingModule().emitManifestProto();
    }
    return result;
  }

  public void buildJar(JavaLibraryBuildRequest build) throws IOException {
    JarCreator jar = new JarCreator(build.getOutputJar());
    try {
      jar.setNormalize(true);
      jar.setCompression(build.compressJar());

      for (String resourceJar : build.getResourceJars()) {
        for (Path root : getJarFileSystem(Paths.get(resourceJar)).getRootDirectories()) {
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

      jar.addDirectory(build.getClassDir());

      jar.addRootEntries(build.getRootResourceFiles());
      addResourceEntries(jar, build.getResourceFiles());
      addMessageEntries(jar, build.getMessageFiles());
    } finally {
      jar.execute();
    }
  }

  /**
   * Extracts the all source jars from the build request into the temporary directory specified in
   * the build request. Empties the temporary directory, if it exists.
   */
  private void setUpSourceJars(JavaLibraryBuildRequest build) throws IOException {
    String sourcesDir = build.getTempDir();

    Path sourceDirFile = Paths.get(sourcesDir);
    if (Files.exists(sourceDirFile)) {
      cleanupDirectory(sourceDirFile);
    }

    if (build.getSourceJars().isEmpty()) {
      return;
    }

    final ByteArrayOutputStream protobufMetadataBuffer = new ByteArrayOutputStream();
    for (String sourceJar : build.getSourceJars()) {
      for (Path root : getJarFileSystem(Paths.get(sourceJar)).getRootDirectories()) {
        Files.walkFileTree(
            root,
            new SimpleFileVisitor<Path>() {
              @Override
              public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                  throws IOException {
                String fileName = path.getFileName().toString();
                if (fileName.endsWith(".java")) {
                  build.getSourceFiles().add(path);
                } else if (fileName.equals(PROTOBUF_META_NAME)) {
                  Files.copy(path, protobufMetadataBuffer);
                }
                return FileVisitResult.CONTINUE;
              }
            });
      }
    }
    Path output = Paths.get(build.getClassDir(), PROTOBUF_META_NAME);
    if (protobufMetadataBuffer.size() > 0) {
      try (OutputStream outputStream = Files.newOutputStream(output)) {
        protobufMetadataBuffer.writeTo(outputStream);
      }
    } else if (Files.exists(output)) {
      // Delete stalled meta file.
      Files.delete(output);
    }
  }

  private FileSystem getJarFileSystem(Path sourceJar) throws IOException {
    FileSystem fs = filesystems.get(sourceJar);
    if (fs == null) {
      filesystems.put(sourceJar, fs = FileSystems.newFileSystem(sourceJar, null));
    }
    return fs;
  }

  // TODO(b/27069912): handle symlinks
  private static void cleanupDirectory(Path dir) throws IOException {
    Files.walkFileTree(
        dir,
        new SimpleFileVisitor<Path>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            Files.delete(file);
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
            Files.delete(dir);
            return FileVisitResult.CONTINUE;
          }
        });
  }

  @Override
  public void close() throws IOException {
    for (FileSystem fs : filesystems.values()) {
      fs.close();
    }
  }
}
