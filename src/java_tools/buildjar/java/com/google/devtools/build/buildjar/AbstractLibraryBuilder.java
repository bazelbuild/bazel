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
import com.google.common.io.ByteStreams;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Base class for java_library builders.
 *
 * <p>Implements common functionality like source files preparation and output jar creation.
 */
public abstract class AbstractLibraryBuilder extends CommonJavaLibraryProcessor {

  /**
   * Prepares a compilation run. This involves cleaning up temporary directories and writing the
   * classpath files.
   */
  protected void prepareSourceCompilation(JavaLibraryBuildRequest build) throws IOException {
    File classDirectory = new File(build.getClassDir());
    if (classDirectory.exists()) {
      try {
        // Necessary for local builds in order to discard previous outputs
        cleanupOutputDirectory(classDirectory);
      } catch (IOException e) {
        throw new IOException("Cannot clean output directory '" + classDirectory + "'", e);
      }
    }
    classDirectory.mkdirs();

    setUpSourceJars(build);
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
    addResourceEntries(jar, build.getResourceFiles());
    addMessageEntries(jar, build.getMessageFiles());

    jar.execute();
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

  /**
   * Internal interface which will listen on each entry of the source jar files during the source
   * jar setup process.
   */
  protected interface SourceJarEntryListener {
    void onEntry(ZipEntry entry) throws IOException;

    void finish() throws IOException;
  }

  protected List<SourceJarEntryListener> getSourceJarEntryListeners(JavaLibraryBuildRequest build) {
    List<SourceJarEntryListener> result = new ArrayList<>();
    result.add(new SourceJavaFileCollector(build));
    return result;
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

  /**
   * Extracts the all source jars from the build request into the temporary directory specified in
   * the build request. Empties the temporary directory, if it exists.
   */
  private void setUpSourceJars(JavaLibraryBuildRequest build) throws IOException {
    String sourcesDir = build.getTempDir();

    File sourceDirFile = new File(sourcesDir);
    if (sourceDirFile.exists()) {
      cleanupDirectory(sourceDirFile, true);
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

  /**
   * Recursively cleans up the files beneath the specified output directory. Does not follow
   * symbolic links. Throws IOException if any deletion fails.
   *
   * <p>Will delete all empty directories.
   *
   * @param dir the directory to clean up.
   * @return true if the directory itself was removed as well.
   */
  boolean cleanupOutputDirectory(File dir) throws IOException {
    return cleanupDirectory(dir, false);
  }

  /**
   * Recursively cleans up the files beneath the specified output directory. Does not follow
   * symbolic links. Throws IOException if any deletion fails. If removeEverything is false, keeps
   * .class files if keepClassFilesDuringCleanup() returns true. If removeEverything is true,
   * removes everything. Will delete all empty directories.
   *
   * @param dir the directory to clean up.
   * @param removeEverything whether to remove all files, or keep flags.xml/.class files.
   * @return true if the directory itself was removed as well.
   */
  private boolean cleanupDirectory(File dir, boolean removeEverything) throws IOException {
    boolean isEmpty = true;
    File[] files = dir.listFiles();
    if (files == null) {
      return false;
    } // avoid race condition
    for (File file : files) {
      if (file.isDirectory()) {
        isEmpty &= cleanupDirectory(file, removeEverything);
      } else if (!removeEverything
          && keepClassFilesDuringCleanup()
          && file.getName().endsWith(".class")) {
        isEmpty = false;
      } else {
        file.delete();
      }
    }
    if (isEmpty) {
      dir.delete();
    }
    return isEmpty;
  }

  /**
   * Returns true if cleaning the output directory should remove all .class files in the output
   * directory.
   */
  protected boolean keepClassFilesDuringCleanup() {
    return false;
  }
}
