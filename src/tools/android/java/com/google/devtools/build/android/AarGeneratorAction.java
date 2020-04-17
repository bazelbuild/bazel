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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.android.builder.core.VariantType;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.android.AndroidDataMerger.MergeConflictException;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDataConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collection;
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
 * <p>
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/AarGeneratorAction\
 *      --mainData path/to/resources:path/to/assets:path/to/manifest\
 *      --manifest path/to/manifest\
 *      --rtxt path/to/rtxt\
 *      --classes path/to/classes.jar\
 *      --aarOutput path/to/write/archive.aar
 * </pre>
 */
public class AarGeneratorAction {
  public static final long DEFAULT_TIMESTAMP =
      LocalDateTime.of(2010, 1, 1, 0, 0, 0)
          .atZone(ZoneId.systemDefault())
          .toInstant()
          .toEpochMilli();

  private static final Logger logger = Logger.getLogger(AarGeneratorAction.class.getName());

  /** Flag specifications for this action. */
  public static final class AarGeneratorOptions extends OptionsBase {
    @Option(
      name = "mainData",
      defaultValue = "null",
      converter = UnvalidatedAndroidDataConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "The directory containing the primary resource directory."
              + "The contents will override the contents of any other resource directories during "
              + "merging. The expected format is resources[#resources]:assets[#assets]:manifest"
    )
    public UnvalidatedAndroidData mainData;

    @Option(
      name = "manifest",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to AndroidManifest.xml."
    )
    public Path manifest;

    @Option(
      name = "rtxt",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to R.txt."
    )
    public Path rtxt;

    @Option(
      name = "classes",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to classes.jar."
    )
    public Path classes;

    @Option(
        name = "proguardSpec",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        allowMultiple = true,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to proguard spec file.")
    public List<Path> proguardSpecs;

    @Option(
      name = "aarOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to write the archive."
    )
    public Path aarOutput;

    @Option(
      name = "throwOnResourceConflict",
      defaultValue = "false",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "If passed, resource merge conflicts will be treated as errors instead of warnings"
    )
    public boolean throwOnResourceConflict;
  }

  public static void main(String[] args) {
    Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(AarGeneratorOptions.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    AarGeneratorOptions options = optionsParser.getOptions(AarGeneratorOptions.class);

    checkFlags(options);

    try (ScopedTemporaryDirectory scopedTmp = new ScopedTemporaryDirectory("aar_gen_tmp")) {
      Path tmp = scopedTmp.getPath();
      Path resourcesOut = tmp.resolve("merged_resources");
      Files.createDirectories(resourcesOut);
      Path assetsOut = tmp.resolve("merged_assets");
      Files.createDirectories(assetsOut);
      logger.fine(String.format("Setup finished at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));
      // There aren't any dependencies, but we merge to combine primary resources from different
      // res/assets directories into a single res and single assets directory.
      MergedAndroidData mergedData =
          AndroidResourceMerger.mergeDataAndWrite(
              options.mainData,
              ImmutableList.<DependencyAndroidData>of(),
              ImmutableList.<DependencyAndroidData>of(),
              resourcesOut,
              assetsOut,
              null,
              VariantType.LIBRARY,
              null,
              /* filteredResources= */ ImmutableList.<String>of(),
              options.throwOnResourceConflict);
      logger.fine(String.format("Merging finished at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));

      writeAar(
          options.aarOutput,
          mergedData,
          options.manifest,
          options.rtxt,
          options.classes,
          options.proguardSpecs);
      logger.fine(
          String.format("Packaging finished at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (MergeConflictException e) {
      logger.log(Level.SEVERE, e.getMessage());
      System.exit(1);
    } catch (IOException | MergingException e) {
      logger.log(Level.SEVERE, "Error during merging resources", e);
      System.exit(1);
    }
    System.exit(0);
  }

  @VisibleForTesting
  static void checkFlags(AarGeneratorOptions options) {
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
      throw new IllegalArgumentException(
          String.format(
              "%s must be specified. Building an .aar without %s is unsupported.",
              Joiner.on(", ").join(nullFlags), Joiner.on(", ").join(nullFlags)));
    }
  }

  @VisibleForTesting
  static void writeAar(
      Path aar,
      final MergedAndroidData data,
      Path manifest,
      Path rtxt,
      Path classes,
      List<Path> proguardSpecs)
      throws IOException {
    try (final ZipOutputStream zipOut =
        new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(aar)))) {
      ZipEntry manifestEntry = new ZipEntry("AndroidManifest.xml");
      manifestEntry.setTime(DEFAULT_TIMESTAMP);
      zipOut.putNextEntry(manifestEntry);
      zipOut.write(Files.readAllBytes(manifest));
      zipOut.closeEntry();

      ZipEntry classJar = new ZipEntry("classes.jar");
      classJar.setTime(DEFAULT_TIMESTAMP);
      zipOut.putNextEntry(classJar);
      zipOut.write(Files.readAllBytes(classes));
      zipOut.closeEntry();

      ZipDirectoryWriter resWriter = new ZipDirectoryWriter(zipOut, data.getResourceDir(), "res");
      Files.walkFileTree(data.getResourceDir(), resWriter);
      resWriter.writeEntries();

      ZipEntry r = new ZipEntry("R.txt");
      r.setTime(DEFAULT_TIMESTAMP);
      zipOut.putNextEntry(r);
      zipOut.write(Files.readAllBytes(rtxt));
      zipOut.closeEntry();

      if (!proguardSpecs.isEmpty()) {
        ZipEntry proguardTxt = new ZipEntry("proguard.txt");
        proguardTxt.setTime(DEFAULT_TIMESTAMP);
        zipOut.putNextEntry(proguardTxt);
        for (Path proguardSpec : proguardSpecs) {
          zipOut.write(Files.readAllBytes(proguardSpec));
          zipOut.write("\r\n".getBytes(UTF_8));
        }
        zipOut.closeEntry();
      }

      if (Files.exists(data.getAssetDir()) && data.getAssetDir().toFile().list().length > 0) {
        ZipDirectoryWriter assetWriter =
            new ZipDirectoryWriter(zipOut, data.getAssetDir(), "assets");
        Files.walkFileTree(data.getAssetDir(), assetWriter);
        assetWriter.writeEntries();
      }
    }
    aar.toFile().setLastModified(DEFAULT_TIMESTAMP);
  }

  private static class ZipDirectoryWriter extends SimpleFileVisitor<Path> {
    private final ZipOutputStream zipOut;
    private final Path root;
    private final String dirName;
    private final Collection<Path> directories = new ArrayList<>();
    private final Collection<Path> files = new ArrayList<>();

    public ZipDirectoryWriter(ZipOutputStream zipOut, Path root, String dirName) {
      this.zipOut = zipOut;
      this.root = root;
      this.dirName = dirName;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      files.add(file);
      return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
        throws IOException {
      directories.add(dir);
      return FileVisitResult.CONTINUE;
    }

    void writeEntries() throws IOException {
      for (Path dir : Ordering.natural().immutableSortedCopy(directories)) {
        writeDirectoryEntry(dir);
      }
      for (Path file : Ordering.natural().immutableSortedCopy(files)) {
        writeFileEntry(file);
      }
    }

    private void writeFileEntry(Path file) throws IOException {
      ZipEntry entry = new ZipEntry(new File(dirName, root.relativize(file).toString()).toString());
      entry.setTime(DEFAULT_TIMESTAMP);
      zipOut.putNextEntry(entry);
      zipOut.write(Files.readAllBytes(file));
      zipOut.closeEntry();
    }

    private void writeDirectoryEntry(Path dir) throws IOException {
      ZipEntry entry =
          new ZipEntry(new File(dirName, root.relativize(dir).toString()).toString() + "/");
      entry.setTime(DEFAULT_TIMESTAMP);
      zipOut.putNextEntry(entry);
      zipOut.closeEntry();
    }
  }
}
