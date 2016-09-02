// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.ideinfo;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.JavaSourcePackage;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.PackageManifest;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * Filters a jar, keeping only the classes that are contained
 * in the supplied package manifest.
 */
public final class JarFilter {

  /** The options for a {@JarFilter} action. */
  public static final class JarFilterOptions extends OptionsBase {
    @Option(name = "jars",
        defaultValue = "null",
        converter = PathListConverter.class,
        category = "input",
        help = "A list of the paths to jars to filter for generated sources.")
    public List<Path> jars;

    @Option(name = "manifest",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "input",
        help = "The path to a package manifest generated only from generated sources.")
    public Path manifest;

    @Option(name = "output",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "The path to the jar to output.")
    public Path output;
  }

  private static final Logger logger = Logger.getLogger(JarFilter.class.getName());

  public static void main(String[] args) throws Exception {
    JarFilterOptions options = parseArgs(args);
    Preconditions.checkNotNull(options.jars);
    Preconditions.checkNotNull(options.manifest);
    Preconditions.checkNotNull(options.output);

    try {
      List<String> archiveFileNamePrefixes = parsePackageManifest(options.manifest);
      filterJars(options.jars, options.output, archiveFileNamePrefixes);
    } catch (Throwable e) {
      logger.log(Level.SEVERE, "Error parsing package strings", e);
      System.exit(1);
    }
    System.exit(0);
  }

  @VisibleForTesting
  static JarFilterOptions parseArgs(String[] args) {
    args = parseParamFileIfUsed(args);
    OptionsParser optionsParser = OptionsParser.newOptionsParser(JarFilterOptions.class);
    optionsParser.parseAndExitUponError(args);
    return optionsParser.getOptions(JarFilterOptions.class);
  }

  private static String[] parseParamFileIfUsed(@Nonnull String[] args) {
    if (args.length != 1 || !args[0].startsWith("@")) {
      return args;
    }
    File paramFile = new File(args[0].substring(1));
    try {
      return Files.readLines(paramFile, StandardCharsets.UTF_8).toArray(new String[0]);
    } catch (IOException e) {
      throw new RuntimeException("Error parsing param file: " + args[0], e);
    }
  }

  private static void filterJars(List<Path> jars, Path output,
      List<String> archiveFileNamePrefixes) throws IOException {
    final int bufferSize = 8 * 1024;
    byte[] buffer = new byte[bufferSize];

    try (ZipOutputStream outputStream = new ZipOutputStream(
        new FileOutputStream(output.toFile()))) {
      for (Path jar : jars) {
        try (ZipFile sourceZipFile = new ZipFile(jar.toFile())) {
          Enumeration<? extends ZipEntry> entries = sourceZipFile.entries();
          while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            if (!shouldKeep(archiveFileNamePrefixes, entry.getName())) {
              continue;
            }

            ZipEntry newEntry = new ZipEntry(entry.getName());
            outputStream.putNextEntry(newEntry);
            try (InputStream inputStream = sourceZipFile.getInputStream(entry)) {
              int len;
              while ((len = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, len);
              }
            }
          }
        }
      }
    }
  }

  @VisibleForTesting
  static boolean shouldKeep(List<String> archiveFileNamePrefixes, String name) {
    for (String archiveFileNamePrefix : archiveFileNamePrefixes) {
      if (name.startsWith(archiveFileNamePrefix)
          && name.length() > archiveFileNamePrefix.length()) {
        char c = name.charAt(archiveFileNamePrefix.length());
        if (c == '.' || c == '$') {
          return true;
        }
      }
    }
    return false;
  }

  @Nullable
  private static List<String> parsePackageManifest(Path manifest) throws IOException {
    try (InputStream inputStream = java.nio.file.Files.newInputStream(manifest)) {
      PackageManifest packageManifest = PackageManifest.parseFrom(inputStream);
      return parsePackageManifest(packageManifest);
    }
  }

  /**
   * Reads the package manifest and computes a list of the expected jar archive
   * file names.
   *
   * Eg.:
   * file java/com/google/foo/Foo.java, package com.google.foo ->
   * com/google/foo/Foo
   */
  @VisibleForTesting
  static List<String> parsePackageManifest(PackageManifest packageManifest) {
    List<String> result = Lists.newArrayList();
    for (JavaSourcePackage javaSourcePackage : packageManifest.getSourcesList()) {
      ArtifactLocation artifactLocation = javaSourcePackage.getArtifactLocation();
      String packageString = javaSourcePackage.getPackageString();
      String archiveFileNamePrefix = getArchiveFileNamePrefix(artifactLocation, packageString);
      result.add(archiveFileNamePrefix);
    }
    return result;
  }

  @Nullable
  private static String getArchiveFileNamePrefix(ArtifactLocation artifactLocation,
      String packageString) {
    String relativePath = artifactLocation.getRelativePath();
    int lastSlashIndex = relativePath.lastIndexOf('/');
    String fileName = lastSlashIndex != -1
        ? relativePath.substring(lastSlashIndex + 1) : relativePath;
    String className = fileName.substring(0, fileName.length() - ".java".length());
    return packageString.replace('.', '/') + '/' + className;
  }
}
