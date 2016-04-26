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

package com.google.devtools.build.buildjar.jarhelper;

import com.google.common.hash.Hashing;
import com.google.common.io.Files;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarOutputStream;

/**
 * A simple helper class for creating Jar files. All Jar entries are sorted alphabetically. Allows
 * normalization of Jar entries by setting the timestamp of non-.class files to the DOS epoch.
 * Timestamps of .class files are set to the DOS epoch + 2 seconds (The zip timestamp granularity)
 * Adjusting the timestamp for .class files is neccessary since otherwise javac will recompile java
 * files if both the java file and its .class file are present.
 */
public class JarHelper {

  public static final String MANIFEST_DIR = "META-INF/";
  public static final String MANIFEST_NAME = JarFile.MANIFEST_NAME;
  public static final String SERVICES_DIR = "META-INF/services/";

  public static final long DOS_EPOCH_IN_JAVA_TIME = 315561600000L;

  // ZIP timestamps have a resolution of 2 seconds.
  // see http://www.info-zip.org/FAQ.html#limits
  public static final long MINIMUM_TIMESTAMP_INCREMENT = 2000L;

  // The name of the Jar file we want to create
  protected final String jarFile;

  // The properties to describe how to create the Jar
  protected boolean normalize;
  protected int storageMethod = JarEntry.DEFLATED;
  protected boolean verbose = false;

  // The state needed to create the Jar
  protected final Set<String> names = new HashSet<>();
  protected JarOutputStream out;

  public JarHelper(String filename) {
    jarFile = filename;
  }

  /**
   * Enables or disables the Jar entry normalization.
   *
   * @param normalize If true the timestamps of Jar entries will be set to the
   *        DOS epoch.
   */
  public void setNormalize(boolean normalize) {
    this.normalize = normalize;
  }

  /**
   * Enables or disables compression for the Jar file entries.
   *
   * @param compression if true enables compressions for the Jar file entries.
   */
  public void setCompression(boolean compression) {
    storageMethod = compression ? JarEntry.DEFLATED : JarEntry.STORED;
  }

  /**
   * Enables or disables verbose messages.
   *
   * @param verbose if true enables verbose messages.
   */
  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  /**
   * Returns the normalized timestamp for a jar entry based on its name.
   * This is necessary since javac will, when loading a class X, prefer a
   * source file to a class file, if both files have the same timestamp.
   * Therefore, we need to adjust the timestamp for class files to slightly
   * after the normalized time.
   * @param name The name of the file for which we should return the
   *     normalized timestamp.
   * @return the time for a new Jar file entry in milliseconds since the epoch.
   */
  private long normalizedTimestamp(String name) {
    if (name.endsWith(".class")) {
      return DOS_EPOCH_IN_JAVA_TIME + MINIMUM_TIMESTAMP_INCREMENT;
    } else {
      return DOS_EPOCH_IN_JAVA_TIME;
    }
  }

  /**
   * Returns the time for a new Jar file entry in milliseconds since the epoch.
   * Uses {@link JarCreator#DOS_EPOCH_IN_JAVA_TIME} for normalized entries,
   * {@link System#currentTimeMillis()} otherwise.
   *
   * @param filename The name of the file for which we are entering the time
   * @return the time for a new Jar file entry in milliseconds since the epoch.
   */
  protected long newEntryTimeMillis(String filename) {
    return normalize ? normalizedTimestamp(filename) : System.currentTimeMillis();
  }

  /**
   * Writes an entry with specific contents to the jar. Directory entries must
   * include the trailing '/'.
   */
  protected void writeEntry(JarOutputStream out, String name, byte[] content) throws IOException {
    if (names.add(name)) {
      // Create a new entry
      JarEntry entry = new JarEntry(name);
      entry.setTime(newEntryTimeMillis(name));
      int size = content.length;
      entry.setSize(size);
      if (size == 0) {
        entry.setMethod(JarEntry.STORED);
        entry.setCrc(0);
        out.putNextEntry(entry);
      } else {
        entry.setMethod(storageMethod);
        if (storageMethod == JarEntry.STORED) {
          entry.setCrc(Hashing.crc32().hashBytes(content).padToLong());
        }
        out.putNextEntry(entry);
        out.write(content);
      }
      out.closeEntry();
    }
  }

  /**
   * Writes a standard Java manifest entry into the JarOutputStream. This
   * includes the directory entry for the "META-INF" directory
   *
   * @param content the Manifest content to write to the manifest entry.
   * @throws IOException
   */
  protected void writeManifestEntry(byte[] content) throws IOException {
    int oldStorageMethod = storageMethod;
    // Do not compress small manifest files, the compressed one is frequently
    // larger than the original. The threshold of 256 bytes is somewhat arbitrary.
    if (content.length < 256) {
      storageMethod = JarEntry.STORED;
    }
    try {
      writeEntry(out, MANIFEST_DIR, new byte[]{});
      writeEntry(out, MANIFEST_NAME, content);
    } finally {
      storageMethod = oldStorageMethod;
    }
  }

  /**
   * Copies file or directory entries from the file system into the jar.
   * Directory entries will be detected and their names automatically '/'
   * suffixed.
   */
  protected void copyEntry(String name, File file) throws IOException {
    if (!names.contains(name)) {
      if (!file.exists()) {
        throw new FileNotFoundException(file.getAbsolutePath() + " (No such file or directory)");
      }
      boolean isDirectory = file.isDirectory();
      if (isDirectory && !name.endsWith("/")) {
        name = name + '/';  // always normalize directory names before checking set
      }
      if (names.add(name)) {
        if (verbose) {
          System.err.println("adding " + file);
        }
        // Create a new entry
        long size = isDirectory ? 0 : file.length();
        JarEntry outEntry = new JarEntry(name);
        long newtime = normalize ? normalizedTimestamp(name) : file.lastModified();
        outEntry.setTime(newtime);
        outEntry.setSize(size);
        if (size == 0L) {
          outEntry.setMethod(JarEntry.STORED);
          outEntry.setCrc(0);
          out.putNextEntry(outEntry);
        } else {
          outEntry.setMethod(storageMethod);
          if (storageMethod == JarEntry.STORED) {
            outEntry.setCrc(Files.hash(file, Hashing.crc32()).padToLong());
          }
          out.putNextEntry(outEntry);
          Files.copy(file, out);
        }
        out.closeEntry();
      }
    }
  }
}
