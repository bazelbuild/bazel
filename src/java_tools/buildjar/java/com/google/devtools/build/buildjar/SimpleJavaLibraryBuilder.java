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
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.buildjar.instrumentation.JacocoInstrumentationProcessor;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.devtools.build.buildjar.javac.BlazeJavacArguments;
import com.google.devtools.build.buildjar.javac.BlazeJavacMain;
import com.google.devtools.build.buildjar.javac.JavacRunner;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.sun.tools.javac.main.Main.Result;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** An implementation of the JavaBuilder that uses in-process javac to compile java files. */
public class SimpleJavaLibraryBuilder {

  /** The name of the protobuf meta file. */
  private static final String PROTOBUF_META_NAME = "protobuf.meta";
  /** Enables more verbose output from the compiler. */
  protected boolean debug = false;

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

  private static List<SourceJarEntryListener> getSourceJarEntryListeners(
      JavaLibraryBuildRequest build) {
    return ImmutableList.of(
        new SourceJavaFileCollector(build),
        new ProtoMetaFileCollector(build.getTempDir(), build.getClassDir()));
  }

  Result compileSources(JavaLibraryBuildRequest build, JavacRunner javacRunner, PrintWriter err)
      throws IOException {
    return javacRunner.invokeJavac(
        build.getPlugins(), build.toBlazeJavacArguments(build.getClassPath()), err);
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
    jar.setNormalize(true);
    jar.setCompression(build.compressJar());
    jar.addDirectory(build.getSourceGenDir());
    jar.execute();
  }

  /**
   * Prepares a compilation run and sets everything up so that the source files in the build request
   * can be compiled. Invokes compileSources to do the actual compilation.
   *
   * @param build A JavaLibraryBuildRequest request object describing what to compile
   * @param err PrintWriter for logging any diagnostic output
   */
  public Result compileJavaLibrary(final JavaLibraryBuildRequest build, final PrintWriter err)
      throws Exception {
    prepareSourceCompilation(build);
    if (build.getSourceFiles().isEmpty()) {
      return Result.OK;
    }
    JavacRunner javacRunner =
        new JavacRunner() {
          @Override
          public Result invokeJavac(
              ImmutableList<BlazeJavaCompilerPlugin> plugins,
              BlazeJavacArguments arguments,
              PrintWriter output) {
            return new BlazeJavacMain(output, plugins).compile(arguments);
          }
        };
    Result result = compileSources(build, javacRunner, err);
    JacocoInstrumentationProcessor processor = build.getJacocoInstrumentationProcessor();
    if (processor != null) {
      processor.processRequest(build);
    }
    return result;
  }

  /** Perform the build. */
  public Result run(JavaLibraryBuildRequest build, PrintWriter err) throws Exception {
    Result result = Result.ERROR;
    try {
      result = compileJavaLibrary(build, err);
      if (result.isOK()) {
        buildJar(build);
      }
      if (!build.getProcessors().isEmpty()) {
        if (build.getGeneratedSourcesOutputJar() != null) {
          buildGensrcJar(build);
        }
      }
    } finally {
      build.getDependencyModule().emitDependencyInformation(build.getClassPath(), result.isOK());
      build.getProcessingModule().emitManifestProto();
    }
    return result;
  }

  public void buildJar(JavaLibraryBuildRequest build) throws IOException {
    JarCreator jar = new JarCreator(build.getOutputJar());
    jar.setNormalize(true);
    jar.setCompression(build.compressJar());

    // The easiest way to handle resource jars is to unpack them into the class directory, just
    // before we start zipping it up.
    for (String resourceJar : build.getResourceJars()) {
      setUpSourceJar(
          new File(resourceJar), build.getClassDir(), new ArrayList<SourceJarEntryListener>());
    }

    jar.addDirectory(build.getClassDir());

    jar.addRootEntries(build.getRootResourceFiles());
    SimpleJavaLibraryBuilder.addResourceEntries(jar, build.getResourceFiles());
    SimpleJavaLibraryBuilder.addMessageEntries(jar, build.getMessageFiles());

    jar.execute();
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

    List<SourceJarEntryListener> listeners = getSourceJarEntryListeners(build);
    for (String sourceJar : build.getSourceJars()) {
      setUpSourceJar(new File(sourceJar), sourcesDir, listeners);
    }
    for (SourceJarEntryListener listener : listeners) {
      listener.finish();
    }
  }

  /**
   * Extracts the source jar into the directory sourceDir. Calls each of the SourceJarEntryListeners
   * for each non-directory entry to do additional work.
   */
  private void setUpSourceJar(
      File sourceJar, String sourceDir, List<SourceJarEntryListener> listeners) throws IOException {
    try (ZipFile zipFile = new ZipFile(sourceJar)) {
      Enumeration<? extends ZipEntry> zipEntries = zipFile.entries();
      while (zipEntries.hasMoreElements()) {
        ZipEntry currentEntry = zipEntries.nextElement();
        String entryName = currentEntry.getName();
        File outputFile = new File(sourceDir, entryName);

        outputFile.getParentFile().mkdirs();

        if (currentEntry.isDirectory()) {
          outputFile.mkdir();
        } else {
          // Copy the data from the zip file to the output file.
          try (InputStream in = zipFile.getInputStream(currentEntry);
              OutputStream out = new FileOutputStream(outputFile)) {
            ByteStreams.copy(in, out);
          }

          for (SourceJarEntryListener listener : listeners) {
            listener.onEntry(currentEntry);
          }
        }
      }
    }
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

  /**
   * Internal interface which will listen on each entry of the source jar files during the source
   * jar setup process.
   */
  protected interface SourceJarEntryListener {
    void onEntry(ZipEntry entry) throws IOException;

    void finish() throws IOException;
  }

  /** A SourceJarEntryListener that collects protobuf meta data files from the source jar files. */
  private static class ProtoMetaFileCollector implements SourceJarEntryListener {

    private final String sourceDir;
    private final String outputDir;
    private final ByteArrayOutputStream buffer;

    public ProtoMetaFileCollector(String sourceDir, String outputDir) {
      this.sourceDir = sourceDir;
      this.outputDir = outputDir;
      this.buffer = new ByteArrayOutputStream();
    }

    @Override
    public void onEntry(ZipEntry entry) throws IOException {
      String entryName = entry.getName();
      if (!entryName.equals(PROTOBUF_META_NAME)) {
        return;
      }
      Files.copy(Paths.get(sourceDir, PROTOBUF_META_NAME), buffer);
    }

    /**
     * Writes the combined the meta files into the output directory. Delete the stalling meta file
     * if no meta file is collected.
     */
    @Override
    public void finish() throws IOException {
      File outputFile = new File(outputDir, PROTOBUF_META_NAME);
      if (buffer.size() > 0) {
        try (OutputStream outputStream = new FileOutputStream(outputFile)) {
          buffer.writeTo(outputStream);
        }
      } else if (outputFile.exists()) {
        // Delete stalled meta file.
        outputFile.delete();
      }
    }
  }

  /**
   * A SourceJarEntryListener that collects a lists of source Java files from the source jar files.
   */
  private static class SourceJavaFileCollector implements SourceJarEntryListener {
    private final List<String> sources;
    private final JavaLibraryBuildRequest build;

    public SourceJavaFileCollector(JavaLibraryBuildRequest build) {
      this.sources = new ArrayList<>();
      this.build = build;
    }

    @Override
    public void onEntry(ZipEntry entry) {
      String entryName = entry.getName();
      if (entryName.endsWith(".java")) {
        sources.add(build.getTempDir() + File.separator + entryName);
      }
    }

    @Override
    public void finish() {
      build.getSourceFiles().addAll(sources);
    }
  }
}
