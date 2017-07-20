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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.JavaSourcePackage;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.PackageManifest;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/** Filters a jar, keeping only the classes that are indicated. */
public final class JarFilter {

  /** The options for a {@JarFilter} action. */
  public static final class JarFilterOptions extends OptionsBase {
    @Option(
      name = "filter_jar",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      help =
          "Paths to target output jars to filter for generated sources. You may use this flag "
              + "multiple times, specify each path with a separate instance of the flag."
    )
    public List<Path> filterJars;

    // TODO(laszlocsomor): remove this flag after 2018-01-31 (about 6 months from now). Everyone
    // should have updated to newer Bazel versions by then.
    @Deprecated
    @Option(
      name = "filter_jars",
      deprecationWarning = "Deprecated in favour of \"--filter_jar\"",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathListConverter.class,
      category = "input",
      help = "A list of the paths to target output jars to filter for generated sources.",
      metadataTags = {OptionMetadataTag.DEPRECATED}
    )
    public List<Path> deprecatedFilterJars;

    @Option(
      name = "filter_source_jar",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      help =
          "Paths to target output source jars to filter for generated sources. You may use this "
              + "flag multiple times, specify each path with a separate instance of the flag."
    )
    public List<Path> filterSourceJars;

    // TODO(laszlocsomor): remove this flag after 2018-01-31 (about 6 months from now). Everyone
    // should have updated to newer Bazel versions by then.
    @Deprecated
    @Option(
      name = "filter_source_jars",
      deprecationWarning = "Deprecated in favour of \"--filter_source_jar\"",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathListConverter.class,
      category = "input",
      help = "A list of the paths to target output source jars to filter for generated sources.",
      metadataTags = {OptionMetadataTag.DEPRECATED}
    )
    public List<Path> deprecatedFilterSourceJars;

    @Option(
      name = "keep_java_file",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      help =
          "Path of target input java files to keep. You may use this flag multiple times, "
              + "specify each path with a separate instance of the flag."
    )
    public List<Path> keepJavaFiles;

    // TODO(laszlocsomor): remove this flag after 2018-01-31 (about 6 months from now). Everyone
    // should have updated to newer Bazel versions by then.
    @Deprecated
    @Option(
      name = "keep_java_files",
      deprecationWarning = "Deprecated in favour of \"--keep_java_file\"",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathListConverter.class,
      category = "input",
      help = "A list of target input java files to keep.",
      metadataTags = {OptionMetadataTag.DEPRECATED}
    )
    public List<Path> deprecatedKeepJavaFiles;

    @Option(
      name = "keep_source_jar",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      help =
          "Path of target input .srcjar files to keep. You may use this flag multiple times, "
              + "specify each path with a separate instance of the flag."
    )
    public List<Path> keepSourceJars;

    // TODO(laszlocsomor): remove this flag after 2018-01-31 (about 6 months from now). Everyone
    // should have updated to newer Bazel versions by then.
    @Deprecated
    @Option(
      name = "keep_source_jars",
      deprecationWarning = "Deprecated in favour of \"--keep_source_jar\"",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathListConverter.class,
      category = "input",
      help = "A list of target input .srcjar files to keep.",
      metadataTags = {OptionMetadataTag.DEPRECATED}
    )
    public List<Path> deprecatedKeepSourceJars;

    @Option(
      name = "filtered_jar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      help = "The path to the jar to output."
    )
    public Path filteredJar;

    @Option(
      name = "filtered_source_jar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      help = "The path to the source jar to output."
    )
    public Path filteredSourceJar;

    // Deprecated options -- only here to maintain command line backwards compatibility
    // with the current blaze native IDE aspect

    @Deprecated
    @Option(
      name = "jar",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      help = "A list of the paths to jars to filter for generated sources."
    )
    public List<Path> jars;

    // TODO(laszlocsomor): remove this flag after 2018-01-31 (about 6 months from now). Everyone
    // should have updated to newer Bazel versions by then.
    @Deprecated
    @Option(
      name = "jars",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathListConverter.class,
      category = "input",
      help = "A list of the paths to jars to filter for generated sources."
    )
    public List<Path> deprecatedJars;

    @Deprecated
    @Option(
      name = "manifest",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      help = "The path to a package manifest generated only from generated sources."
    )
    public Path manifest;

    @Deprecated
    @Option(
      name = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      help = "The path to the jar to output."
    )
    public Path output;
  }

  private static final Logger logger = Logger.getLogger(JarFilter.class.getName());

  private static final Pattern JAVA_PACKAGE_PATTERN =
      Pattern.compile("^\\s*package\\s+([\\w\\.]+);");

  public static void main(String[] args) throws Exception {
    JarFilterOptions options = parseArgs(args);
    try {
      main(options);
    } catch (Throwable e) {
      logger.log(Level.SEVERE, "Error parsing package strings", e);
      System.exit(1);
    }
    System.exit(0);
  }

  @VisibleForTesting
  static void main(JarFilterOptions options) throws Exception {
    Preconditions.checkNotNull(options.filteredJar);

    if (options.filterJars == null) {
      options.filterJars = ImmutableList.of();
    }
    if (options.filterSourceJars == null) {
      options.filterSourceJars = ImmutableList.of();
    }

    final List<String> archiveFileNamePrefixes = Lists.newArrayList();
    if (options.manifest != null) {
      archiveFileNamePrefixes.addAll(parsePackageManifest(options.manifest));
    }
    if (options.keepJavaFiles != null) {
      archiveFileNamePrefixes.addAll(parseJavaFiles(options.keepJavaFiles));
    }
    if (options.keepSourceJars != null) {
      archiveFileNamePrefixes.addAll(parseSrcJars(options.keepSourceJars));
    }

    filterJars(
        options.filterJars,
        options.filteredJar,
        new Predicate<String>() {
          @Override
          public boolean apply(@Nullable String s) {
            return shouldKeepClass(archiveFileNamePrefixes, s);
          }
        });
    if (options.filteredSourceJar != null) {
      filterJars(
          options.filterSourceJars,
          options.filteredSourceJar,
          new Predicate<String>() {
            @Override
            public boolean apply(@Nullable String s) {
              return shouldKeepJavaFile(archiveFileNamePrefixes, s);
            }
          });
    }
  }

  @VisibleForTesting
  static JarFilterOptions parseArgs(String[] args) {
    args = parseParamFileIfUsed(args);
    OptionsParser optionsParser = OptionsParser.newOptionsParser(JarFilterOptions.class);
    optionsParser.parseAndExitUponError(args);
    JarFilterOptions options = optionsParser.getOptions(JarFilterOptions.class);

    options.filterJars = PathListConverter.concatLists(
        options.filterJars, options.deprecatedFilterJars);
    options.filterSourceJars = PathListConverter.concatLists(
        options.filterSourceJars, options.deprecatedFilterSourceJars);
    options.keepJavaFiles = PathListConverter.concatLists(
        options.keepJavaFiles, options.deprecatedKeepJavaFiles);
    options.keepSourceJars = PathListConverter.concatLists(
        options.keepSourceJars, options.deprecatedKeepSourceJars);
    options.jars = PathListConverter.concatLists(
        options.jars, options.deprecatedJars);
    // Migrate options from v1 jar filter
    if (options.filterJars.isEmpty() && !options.jars.isEmpty()) {
      options.filterJars = options.jars;
    }
    if (options.filteredJar == null && options.output != null) {
      options.filteredJar = options.output;
    }
    return options;
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

  /** Finds the expected jar archive file name prefixes for the java files. */
  static List<String> parseJavaFiles(List<Path> javaFiles) throws IOException {
    ListeningExecutorService executorService =
        MoreExecutors.listeningDecorator(
            Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()));

    List<ListenableFuture<String>> futures = Lists.newArrayList();
    for (final Path javaFile : javaFiles) {
      futures.add(
          executorService.submit(
              new Callable<String>() {
                @Override
                public String call() throws Exception {
                  String packageString = getDeclaredPackageOfJavaFile(javaFile);
                  return packageString != null
                      ? getArchiveFileNamePrefix(javaFile.toString(), packageString)
                      : null;
                }
              }));
    }
    try {
      List<String> archiveFileNamePrefixes = Futures.allAsList(futures).get();
      List<String> result = Lists.newArrayList();
      for (String archiveFileNamePrefix : archiveFileNamePrefixes) {
        if (archiveFileNamePrefix != null) {
          result.add(archiveFileNamePrefix);
        }
      }
      return result;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException(e);
    } catch (ExecutionException e) {
      throw new IOException(e);
    }
  }

  static List<String> parseSrcJars(List<Path> srcJars) throws IOException {
    List<String> result = Lists.newArrayList();
    for (Path srcJar : srcJars) {
      try (ZipFile sourceZipFile = new ZipFile(srcJar.toFile())) {
        Enumeration<? extends ZipEntry> entries = sourceZipFile.entries();
        while (entries.hasMoreElements()) {
          ZipEntry entry = entries.nextElement();
          if (!entry.getName().endsWith(".java")) {
            continue;
          }
          try (BufferedReader reader =
              new BufferedReader(
                  new InputStreamReader(sourceZipFile.getInputStream(entry), UTF_8))) {
            String packageString = parseDeclaredPackage(reader);
            if (packageString != null) {
              String archiveFileNamePrefix =
                  getArchiveFileNamePrefix(entry.getName(), packageString);
              result.add(archiveFileNamePrefix);
            }
          }
        }
      }
    }
    return result;
  }

  @Nullable
  private static String getDeclaredPackageOfJavaFile(Path javaFile) {
    try (BufferedReader reader =
        java.nio.file.Files.newBufferedReader(javaFile, StandardCharsets.UTF_8)) {
      return parseDeclaredPackage(reader);

    } catch (IOException e) {
      logger.log(Level.WARNING, "Error parsing package string from java source: " + javaFile, e);
      return null;
    }
  }

  @Nullable
  private static String parseDeclaredPackage(BufferedReader reader) throws IOException {
    String line;
    while ((line = reader.readLine()) != null) {
      Matcher packageMatch = JAVA_PACKAGE_PATTERN.matcher(line);
      if (packageMatch.find()) {
        return packageMatch.group(1);
      }
    }
    return null;
  }

  /**
   * Computes the expected archive file name prefix of a java class.
   *
   * <p>Eg.: file java/com/google/foo/Foo.java, package com.google.foo -> com/google/foo/Foo
   */
  private static String getArchiveFileNamePrefix(String javaFile, String packageString) {
    int lastSlashIndex = javaFile.lastIndexOf('/');
    // On Windows, the separator could be '\\'
    if (lastSlashIndex == -1) {
      lastSlashIndex = javaFile.lastIndexOf('\\');
    }
    String fileName = lastSlashIndex != -1 ? javaFile.substring(lastSlashIndex + 1) : javaFile;
    String className = fileName.substring(0, fileName.length() - ".java".length());
    return packageString.replace('.', '/') + '/' + className;
  }

  /** Reads the package manifest and computes a list of the expected jar archive file names. */
  private static List<String> parsePackageManifest(Path manifest) throws IOException {
    try (InputStream inputStream = java.nio.file.Files.newInputStream(manifest)) {
      PackageManifest packageManifest = PackageManifest.parseFrom(inputStream);
      return parsePackageManifest(packageManifest);
    }
  }

  @VisibleForTesting
  static List<String> parsePackageManifest(PackageManifest packageManifest) {
    List<String> result = Lists.newArrayList();
    for (JavaSourcePackage javaSourcePackage : packageManifest.getSourcesList()) {
      ArtifactLocation artifactLocation = javaSourcePackage.getArtifactLocation();
      String packageString = javaSourcePackage.getPackageString();
      String archiveFileNamePrefix =
          getArchiveFileNamePrefix(artifactLocation.getRelativePath(), packageString);
      result.add(archiveFileNamePrefix);
    }
    return result;
  }

  /** Filters a list of jars, keeping anything matching the passed predicate. */
  private static void filterJars(List<Path> jars, Path output, Predicate<String> shouldKeep)
      throws IOException {
    final int bufferSize = 8 * 1024;
    byte[] buffer = new byte[bufferSize];

    try (ZipOutputStream outputStream =
        new ZipOutputStream(new FileOutputStream(output.toFile()))) {
      for (Path jar : jars) {
        try (ZipFile sourceZipFile = new ZipFile(jar.toFile())) {
          Enumeration<? extends ZipEntry> entries = sourceZipFile.entries();
          while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            if (!shouldKeep.apply(entry.getName())) {
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
  static boolean shouldKeepClass(List<String> archiveFileNamePrefixes, String name) {
    if (!name.endsWith(".class")) {
      return false;
    }
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

  private static boolean shouldKeepJavaFile(List<String> archiveFileNamePrefixes, String name) {
    if (!name.endsWith(".java")) {
      return false;
    }
    String nameWithoutJava = name.substring(0, name.length() - ".java".length());
    return archiveFileNamePrefixes.contains(nameWithoutJava);
  }
}
