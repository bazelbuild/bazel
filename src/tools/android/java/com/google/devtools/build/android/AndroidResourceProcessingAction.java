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


import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.Hashing;
import com.google.devtools.build.android.Converters.DependencyAndroidDataListConverter;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.FullRevisionConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDataConverter;
import com.google.devtools.build.android.Converters.VariantConfigurationTypeConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.TriState;

import com.android.builder.core.VariantConfiguration;
import com.android.builder.model.AaptOptions;
import com.android.ide.common.internal.AaptCruncher;
import com.android.ide.common.internal.CommandLineRunner;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.ide.common.res2.MergingException;
import com.android.sdklib.repository.FullRevision;
import com.android.utils.StdLogger;

import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;


/**
 * Provides an entry point for the resource processing using the AOSP build tools.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/AndroidResourceProcessingAction\
 *      --sdkRoot path/to/sdk\
 *      --aapt path/to/sdk/aapt\
 *      --annotationJar path/to/sdk/annotationJar\
 *      --adb path/to/sdk/adb\
 *      --zipAlign path/to/sdk/zipAlign\
 *      --androidJar path/to/sdk/androidJar\
 *      --manifest path/to/manifest\
 *      --primaryData path/to/resources:path/to/assets:path/to/manifest:path/to/R.txt
 *      --data p/t/res1:p/t/assets1:p/t/1/AndroidManifest.xml:p/t/1/R.txt,\
 *             p/t/res2:p/t/assets2:p/t/2/AndroidManifest.xml:p/t/2/R.txt
 *      --packagePath path/to/write/archive.ap_
 *      --srcJarOutput path/to/write/archive.srcjar
 * </pre>
 */
public class AndroidResourceProcessingAction {

  private static final StdLogger STD_LOGGER =
      new StdLogger(com.android.utils.StdLogger.Level.WARNING);

  private static final Logger LOGGER =
      Logger.getLogger(AndroidResourceProcessingAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {
    @Option(name = "apiVersion",
        defaultValue = "21.0.0",
        converter = FullRevisionConverter.class,
        category = "config",
        help = "ApiVersion indicates the version passed to the AndroidBuilder. ApiVersion must be"
            + " > 19.10 when defined.")
    // TODO(bazel-team): Determine what the API version changes in AndroidBuilder.
    public FullRevision apiVersion;

    @Option(name = "aapt",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Aapt tool location for resource packaging.")
    public Path aapt;

    @Option(name = "annotationJar",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Annotation Jar for builder invocations.")
    public Path annotationJar;

    @Option(name = "adb",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Path to adb for builder functions.")
    //TODO(bazel-team): Determine if this is completely necessary for running AndroidBuilder.
    public Path adb;

    @Option(name = "zipAlign",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Path to zipAlign for building apks.")
    public Path zipAlign;

    @Option(name = "androidJar",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Path to the android jar for resource packaging and building apks.")
    public Path androidJar;

    @Option(name = "primaryData",
        defaultValue = "null",
        converter = UnvalidatedAndroidDataConverter.class,
        category = "input",
        help = "The directory containing the primary resource directory. The contents will override"
            + " the contents of any other resource directories during merging. The expected format"
            + " is resources[|resources]:assets[|assets]:manifest")
    public UnvalidatedAndroidData primaryData;

    @Option(name = "data",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        help = "Transitive Data dependencies. These values will be used if not defined in the "
            + "primary resources. The expected format is "
            + "resources[#resources]:assets[#assets]:manifest:r.txt:symbols.bin"
            + "[,resources[#resources]:assets[#assets]:manifest:r.txt:symbols.bin]")
    public List<DependencyAndroidData> transitiveData;

    @Option(name = "directData",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        help = "Direct Data dependencies. These values will be used if not defined in the "
            + "primary resources. The expected format is "
            + "resources[#resources]:assets[#assets]:manifest:r.txt:symbols.bin"
            + "[,resources[#resources]:assets[#assets]:manifest:r.txt:symbols.bin]")
    public List<DependencyAndroidData> directData;

    @Option(name = "rOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to where the R.txt should be written.")
    public Path rOutput;

    @Option(name = "symbolsTxtOut",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to where the symbolsTxt should be written.")
    public Path symbolsTxtOut;

    @Option(name = "packagePath",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to the write the archive.")
    public Path packagePath;

    @Option(name = "proguardOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path for the proguard file.")
    public Path proguardOutput;

    @Option(name = "manifestOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path for the modified manifest.")
    public Path manifestOutput;

    @Option(name = "srcJarOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path for the generated java source jar.")
    public Path srcJarOutput;

    @Option(name = "packageType",
        defaultValue = "DEFAULT",
        converter = VariantConfigurationTypeConverter.class,
        category = "config",
        help = "Variant configuration type for packaging the resources."
            + " Acceptible values DEFAULT, LIBRARY, TEST")
    public VariantConfiguration.Type packageType;

    @Option(name = "densities",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list densities to filter the resource drawables by.")
    public List<String> densities;

    @Option(name = "debug",
        defaultValue = "false",
        category = "config",
        help = "Indicates if it is a debug build.")
    public boolean debug;

    @Option(name = "resourceConfigs",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of resource config filters to pass to aapt.")
    public List<String> resourceConfigs;

    @Option(name = "useAaptCruncher",
        defaultValue = "auto",
        category = "config",
        help = "Use the legacy aapt cruncher, defaults to true for non-LIBRARY packageTypes. "
            + " LIBRARY packages do not benefit from the additional processing as the resources"
            + " will need to be reprocessed during the generation of the final apk. See"
            + " https://code.google.com/p/android/issues/detail?id=67525 for a discussion of the"
            + " different png crunching methods.")
    public TriState useAaptCruncher;

    @Option(name = "uncompressedExtensions",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of file extensions not to compress.")
    public List<String> uncompressedExtensions;

    @Option(name = "packageForR",
        defaultValue = "null",
        category = "config",
        help = "Custom java package to generate the R symbols files.")
    public String packageForR;

    @Option(name = "applicationId",
        defaultValue = "null",
        category = "config",
        help = "Custom application id (package manifest) for the packaged manifest.")
    public String applicationId;

    @Option(name = "versionName",
        defaultValue = "null",
        category = "config",
        help = "Version name to stamp into the packaged manifest.")
    public String versionName;

    @Option(name = "versionCode",
        defaultValue = "-1",
        category = "config",
        help = "Version code to stamp into the packaged manifest.")
    public int versionCode;

    @Option(name = "assetsToIgnore",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of assets extensions to ignore.")
    public List<String> assetsToIgnore;
  }

  private static Options options;

  public static void main(String[] args) {
    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.parseAndExitUponError(args);
    options = optionsParser.getOptions(Options.class);
    FileSystem fileSystem = FileSystems.getDefault();
    Path working = fileSystem.getPath("").toAbsolutePath();
    Path mergedAssets = working.resolve("merged_assets");
    Path mergedResources = working.resolve("merged_resources");

    final AndroidResourceProcessor resourceProcessor =
        new AndroidResourceProcessor(STD_LOGGER);

    try {

      Path expandedOut = Files.createTempDirectory("tmp-expanded");
      expandedOut.toFile().deleteOnExit();
      Path deduplicatedOut = Files.createTempDirectory("tmp-deduplicated");
      deduplicatedOut.toFile().deleteOnExit();

      Path generatedSources = null;
      if (options.srcJarOutput != null || options.rOutput != null
          || options.symbolsTxtOut != null) {
        generatedSources = Files.createTempDirectory("generated_resources");
        generatedSources.toFile().deleteOnExit();
      }

      LOGGER.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      final ImmutableList<DirectoryModifier> modifiers = ImmutableList.of(
          new PackedResourceTarExpander(expandedOut, working),
          new FileDeDuplicator(Hashing.murmur3_128(), deduplicatedOut, working));

      // Resources can appear in both the direct dependencies and transitive -- use a set to
      // ensure depeduplication.
      List<DependencyAndroidData> data =
          ImmutableSet.<DependencyAndroidData>builder()
              .addAll(options.directData)
              .addAll(options.transitiveData)
              .build()
              .asList();

      final MergedAndroidData mergedData = resourceProcessor.mergeData(
          options.primaryData,
          data,
          mergedResources,
          mergedAssets,
          modifiers,
          useAaptCruncher() ?  new AaptCruncher(options.aapt.toString(),
              new CommandLineRunner(STD_LOGGER)) : null,
          true);

      LOGGER.fine(String.format("Merging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      final Path filteredResources = fileSystem.getPath("resources-filtered");
      final Path densityManifest = fileSystem.getPath("manifest-filtered/AndroidManifest.xml");
      final DensityFilteredAndroidData filteredData = mergedData.filter(
          new DensitySpecificResourceFilter(options.densities, filteredResources, working),
          new DensitySpecificManifestProcessor(options.densities, densityManifest));
      LOGGER.fine(
          String.format("Density filtering finished at %sms",
              timer.elapsed(TimeUnit.MILLISECONDS)));
      resourceProcessor.processResources(
          options.aapt,
          options.androidJar,
          options.packageType,
          options.debug,
          options.packageForR,
          new FlagAaptOptions(),
          options.resourceConfigs,
          options.applicationId,
          options.versionCode,
          options.versionName,
          filteredData,
          data,
          working.resolve("manifest"),
          generatedSources,
          options.packagePath,
          options.proguardOutput,
          options.manifestOutput);
      LOGGER.fine(String.format("appt finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      if (options.srcJarOutput != null) {
        resourceProcessor.createSrcJar(generatedSources, options.srcJarOutput,
            VariantConfiguration.Type.LIBRARY == options.packageType);
      }
      if (options.rOutput != null) {
        resourceProcessor.copyRToOutput(generatedSources, options.rOutput,
            VariantConfiguration.Type.LIBRARY == options.packageType);
      }
      if (options.symbolsTxtOut != null) {
        resourceProcessor.copyRToOutput(generatedSources, options.symbolsTxtOut,
            VariantConfiguration.Type.LIBRARY == options.packageType);
      }
      LOGGER.fine(String.format("Packaging finished at %sms",
          timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (MergingException e) {
      LOGGER.log(java.util.logging.Level.SEVERE, "Error during merging resources", e);
      System.exit(1);
    } catch (IOException | InterruptedException | LoggedErrorException e) {
      LOGGER.log(java.util.logging.Level.SEVERE, "Error during processing resources", e);
      System.exit(2);
    } catch (Exception e) {
      LOGGER.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      System.exit(3);
    }
    LOGGER.fine(String.format("Resources processed in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    // AOSP code can leave dangling threads.
    System.exit(0);
  }

  private static boolean useAaptCruncher() {
    // If the value was set, use that.
    if (options.useAaptCruncher != TriState.AUTO) {
      return options.useAaptCruncher == TriState.YES;
    }
    // By default png cruncher shouldn't be invoked on a library -- the work is just thrown away.
    return options.packageType != VariantConfiguration.Type.LIBRARY;
  }

  private static final class FlagAaptOptions implements AaptOptions {
    @Override
    public boolean getUseAaptPngCruncher() {
      return options.useAaptCruncher != TriState.NO;
    }

    @Override
    public Collection<String> getNoCompress() {
      if (!options.uncompressedExtensions.isEmpty()) {
        return options.uncompressedExtensions;
      }
      return ImmutableList.of();
    }

    @Override
    public String getIgnoreAssets() {
      if (!options.assetsToIgnore.isEmpty()) {
        return Joiner.on(":").join(options.assetsToIgnore);
      }
      return null;
    }

    @Override
    public boolean getFailOnMissingConfigEntry() {
      return false;
    }
  }
}

