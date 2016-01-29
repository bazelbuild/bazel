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
package com.google.devtools.build.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hashing;
import com.google.devtools.build.android.Converters.DependencyAndroidDataListConverter;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDataConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;

import com.android.ide.common.res2.MergingException;
import com.android.utils.StdLogger;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Action to generate an AAR archive for an Android library.
 *
 * <p><pre>
 * Example Usage:
 *   java/com/google/build/android/AarGeneratorAction\
 *      --primaryData path/to/resources:path/to/assets:path/to/manifest\
 *      --data p/t/res1:p/t/assets1:p/t/1/AndroidManifest.xml:p/t/1/R.txt,\
 *             p/t/res2:p/t/assets2:p/t/2/AndroidManifest.xml:p/t/2/R.txt\
 *      --manifest path/to/manifest\
 *      --rtxt path/to/rtxt\
 *      --classes path/to/classes.jar\
 *      --strictMerge\
 *      --aarOutput path/to/write/archive.aar
 * </pre>
 */
public class AarGeneratorAction {
  private static final Long EPOCH = 0L;

  private static final Logger logger = Logger.getLogger(AarGeneratorAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {
    @Option(name = "mainData",
        defaultValue = "null",
        converter = UnvalidatedAndroidDataConverter.class,
        category = "input",
        help = "The directory containing the primary resource directory."
            + "The contents will override the contents of any other resource directories during "
            + "merging. The expected format is resources[#resources]:assets[#assets]:manifest")
    public UnvalidatedAndroidData mainData;

    @Option(name = "dependencyData",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        help = "Additional Data dependencies. These values will be used if not defined in "
            + "the primary resources. The expected format is "
            + "resources[#resources]:assets[#assets]:manifest:r.txt"
            + "[,resources[#resources]:assets[#assets]:manifest:r.txt]")
    public List<DependencyAndroidData> dependencyData;

    @Option(name = "manifest",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "input",
        help = "Path to AndroidManifest.xml.")
    public Path manifest;

    @Option(name = "rtxt",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "input",
        help = "Path to R.txt.")
    public Path rtxt;

    @Option(name = "classes",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "input",
        help = "Path to classes.jar.")
    public Path classes;

    @Option(name = "aarOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to write the archive.")
    public Path aarOutput;

    @Option(name = "strictMerge",
        defaultValue = "true",
        category = "option",
        help = "Merge strategy for resources.")
    public boolean strictMerge;
  }

  public static void main(String[] args) {
    Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);

    checkFlags(options);

    FileSystem fileSystem = FileSystems.getDefault();
    Path working = fileSystem.getPath("").toAbsolutePath();

    AndroidResourceProcessor resourceProcessor = new AndroidResourceProcessor(
        new StdLogger(com.android.utils.StdLogger.Level.VERBOSE));

    try {
      Path resourcesOut = Files.createTempDirectory("tmp-resources");
      resourcesOut.toFile().deleteOnExit();
      Path assetsOut = Files.createTempDirectory("tmp-assets");
      assetsOut.toFile().deleteOnExit();
      Path expandedOut = Files.createTempDirectory("tmp-expanded");
      expandedOut.toFile().deleteOnExit();
      Path deduplicatedOut = Files.createTempDirectory("tmp-deduplicated");
      deduplicatedOut.toFile().deleteOnExit();

      logger.fine(String.format("Setup finished at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));

      ImmutableList<DirectoryModifier> modifiers = ImmutableList.of(
          new PackedResourceTarExpander(expandedOut, working),
          new FileDeDuplicator(Hashing.murmur3_128(), deduplicatedOut, working));
      MergedAndroidData mergedData = resourceProcessor.mergeData(options.mainData,
          options.dependencyData,
          resourcesOut,
          assetsOut,
          modifiers,
          null,
          options.strictMerge);
      logger.fine(String.format("Merging finished at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));

      writeAar(options.aarOutput, mergedData, options.manifest, options.rtxt, options.classes);
      logger.fine(
          String.format("Packaging finished at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));

    } catch (IOException | MergingException e) {
      logger.log(Level.SEVERE, "Error during merging resources", e);
      System.exit(1);
    }
    System.exit(0);
  }

  @VisibleForTesting
  static void checkFlags(Options options) throws IllegalArgumentException {
    List<String> nullFlags = new LinkedList<>();
    if (options.manifest == null) {
      nullFlags.add("manifest");
    }
    if (options.rtxt == null) {
      nullFlags.add("rtxt");
    }
    if (options.classes == null) {
      nullFlags.add("classes");
    }
    if (!nullFlags.isEmpty()) {
      throw new IllegalArgumentException(String.format("%s must be specified. Building an .aar "
          + "without %s is unsupported.",
          Joiner.on(", ").join(nullFlags), Joiner.on(", ").join(nullFlags)));
    }
  }

  @VisibleForTesting
  static void writeAar(Path aar, final MergedAndroidData data, Path manifest, Path rtxt,
      Path classes) throws IOException {
    try (final ZipOutputStream zipOut = new ZipOutputStream(new FileOutputStream(aar.toFile()))) {
      ZipEntry manifestEntry = new ZipEntry("AndroidManifest.xml");
      zipOut.putNextEntry(manifestEntry);
      zipOut.write(Files.readAllBytes(manifest));
      zipOut.closeEntry();

      ZipEntry classJar = new ZipEntry("classes.jar");
      zipOut.putNextEntry(classJar);
      zipOut.write(Files.readAllBytes(classes));
      zipOut.closeEntry();

      Files.walkFileTree(data.getResourceDirFile().toPath(),
          new ZipDirectoryWriter(zipOut, data.getResourceDirFile().toPath(), "res"));

      ZipEntry r = new ZipEntry("R.txt");
      zipOut.putNextEntry(r);
      zipOut.write(Files.readAllBytes(rtxt));
      zipOut.closeEntry();

      if (data.getAssetDirFile().exists() && data.getAssetDirFile().list().length > 0) {
        Files.walkFileTree(data.getAssetDirFile().toPath(),
            new ZipDirectoryWriter(zipOut, data.getAssetDirFile().toPath(), "assets"));
      }
    }
    aar.toFile().setLastModified(EPOCH);
  }
  
  private static class ZipDirectoryWriter extends SimpleFileVisitor<Path> {
    private final ZipOutputStream zipOut;
    private final Path root;
    private final String dirName;
    
    public ZipDirectoryWriter(ZipOutputStream zipOut, Path root, String dirName) {
      this.zipOut = zipOut;
      this.root = root;
      this.dirName = dirName;
    }
    
    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      ZipEntry entry = new ZipEntry(new File(dirName, root.relativize(file).toString()).toString());
      zipOut.putNextEntry(entry);
      zipOut.write(Files.readAllBytes(file));
      zipOut.closeEntry();
      return FileVisitResult.CONTINUE;
    }
    
    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
        throws IOException {
      ZipEntry entry = new ZipEntry(new File(dirName, root.relativize(dir).toString())
          .toString() + "/");
      zipOut.putNextEntry(entry);
      zipOut.closeEntry();
      return FileVisitResult.CONTINUE;
    }
  }
}
