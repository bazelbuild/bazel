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

import com.android.builder.core.VariantType;
import com.android.utils.StdLogger;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.AndroidResourceProcessor.FlagAaptOptions;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Validates merged resources for an android_library via AAPT. Takes as input, the merged resources
 * from {@link AndroidResourceMergingAction}.
 */
public class AndroidResourceValidatorAction {

  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);

  private static final Logger logger =
      Logger.getLogger(AndroidResourceValidatorAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      name = "mergedResources",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the read merged resources archive."
    )
    public Path mergedResources;

    @Option(
      name = "manifest",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path for the AndroidManifest.xml."
    )
    public Path manifest;

    @Option(
      name = "packageForR",
      defaultValue = "null",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Custom java package to generate the R symbols files."
    )
    public String packageForR;

    @Option(
      name = "srcJarOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path for the generated java source jar."
    )
    public Path srcJarOutput;

    @Option(
      name = "rOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to where the R.txt should be written."
    )
    public Path rOutput;
  }

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, AaptConfigOptions.class);
    optionsParser.enableParamsFileSupport(FileSystems.getDefault());
    optionsParser.parseAndExitUponError(args);
    AaptConfigOptions aaptConfigOptions = optionsParser.getOptions(AaptConfigOptions.class);
    Options options = optionsParser.getOptions(Options.class);

    final AndroidResourceProcessor resourceProcessor = new AndroidResourceProcessor(stdLogger);
    VariantType packageType = VariantType.LIBRARY;

    Preconditions.checkNotNull(options.rOutput);
    Preconditions.checkNotNull(options.srcJarOutput);
    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("resource_validator_tmp")) {
      Path tmp = scopedTmp.getPath();
      Path expandedOut = tmp.resolve("tmp-expanded");
      Path resources = expandedOut.resolve("res");
      Path assets = expandedOut.resolve("assets");
      Path generatedSources = tmp.resolve("generated_resources");
      Path dummyManifest = tmp.resolve("manifest-aapt-dummy/AndroidManifest.xml");

      unpackZip(options.mergedResources, expandedOut);
      logger.fine(String.format("unpacked zip at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      // We need to make the manifest aapt safe (w.r.t., placeholders). For now, just stub it out.
      resourceProcessor.writeDummyManifestForAapt(dummyManifest, options.packageForR);

      resourceProcessor.runAapt(
          tmp,
          aaptConfigOptions.aapt,
          aaptConfigOptions.androidJar,
          aaptConfigOptions.buildToolsVersion,
          packageType,
          aaptConfigOptions.debug,
          options.packageForR,
          new FlagAaptOptions(aaptConfigOptions),
          aaptConfigOptions.resourceConfigs,
          ImmutableList.<String>of(),
          dummyManifest,
          resources,
          assets,
          generatedSources,
          null, /* packageOut */
          null, /* proguardOut */
          null, /* mainDexProguardOut */
          null /* publicResourcesOut */);
      logger.fine(String.format("aapt finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      AndroidResourceOutputs.copyRToOutput(
          generatedSources, options.rOutput, VariantType.LIBRARY == packageType);

      AndroidResourceOutputs.createSrcJar(
          generatedSources, options.srcJarOutput, VariantType.LIBRARY == packageType);
    } catch (Exception e) {
      logger.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      throw e;
    } finally {
      resourceProcessor.shutdown();
    }
    logger.fine(String.format("Resources merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }

  private static void unpackZip(Path mergedResources, Path expandedOut) throws IOException {
    byte[] buffer = new byte[4096];
    try (ZipInputStream zis =
        new ZipInputStream(new BufferedInputStream(Files.newInputStream(mergedResources)))) {
      ZipEntry z = zis.getNextEntry();
      while (z != null) {
        String entryName = z.getName();
        Path outputPath = expandedOut.resolve(entryName);
        Files.createDirectories(outputPath.getParent());
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(outputPath))) {
          int count = zis.read(buffer);
          while (count != -1) {
            out.write(buffer, 0, count);
            count = zis.read(buffer);
          }
        }
        z = zis.getNextEntry();
      }
    }
  }
}
