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
import com.android.ide.common.internal.AaptCruncher;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.process.DefaultProcessExecutor;
import com.android.ide.common.process.LoggedProcessOutputHandler;
import com.android.ide.common.res2.MergingException;
import com.android.utils.StdLogger;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.AndroidResourceProcessor.FlagAaptOptions;
import com.google.devtools.build.android.Converters.DependencyAndroidDataListConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDataConverter;
import com.google.devtools.build.android.Converters.VariantTypeConverter;
import com.google.devtools.build.android.SplitConfigurationFilter.UnrecognizedSplitsException;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.TriState;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Collections;
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
 *      --manifestOutput path/to/manifest\
 *      --primaryData path/to/resources:path/to/assets:path/to/manifest\
 *      --data p/t/res1:p/t/assets1:p/t/1/AndroidManifest.xml:p/t/1/R.txt:symbols,\
 *             p/t/res2:p/t/assets2:p/t/2/AndroidManifest.xml:p/t/2/R.txt:symbols\
 *      --packagePath path/to/write/archive.ap_\
 *      --srcJarOutput path/to/write/archive.srcjar
 * </pre>
 */
public class AndroidResourceProcessingAction {

  private static final StdLogger STD_LOGGER =
      new StdLogger(com.android.utils.StdLogger.Level.WARNING);

  private static final Logger logger =
      Logger.getLogger(AndroidResourceProcessingAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {
    @Option(name = "primaryData",
        defaultValue = "null",
        converter = UnvalidatedAndroidDataConverter.class,
        category = "input",
        help = "The directory containing the primary resource directory. The contents will override"
            + " the contents of any other resource directories during merging. The expected format"
            + " is " + UnvalidatedAndroidData.EXPECTED_FORMAT)
    public UnvalidatedAndroidData primaryData;

    @Option(name = "data",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        help = "Transitive Data dependencies. These values will be used if not defined in the "
            + "primary resources. The expected format is "
            + DependencyAndroidData.EXPECTED_FORMAT
            + "[,...]")
    public List<DependencyAndroidData> transitiveData;

    @Option(name = "directData",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        help = "Direct Data dependencies. These values will be used if not defined in the "
            + "primary resources. The expected format is "
            + DependencyAndroidData.EXPECTED_FORMAT
            + "[,...]")
    public List<DependencyAndroidData> directData;

    @Option(name = "rOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to where the R.txt should be written.")
    public Path rOutput;

    @Option(name = "symbolsOut",
        oldName = "symbolsTxtOut",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to where the symbols should be written.")
    public Path symbolsOut;

    @Option(name = "dataBindingInfoOut",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to where data binding's layout info output should be written.")
    public Path dataBindingInfoOut;

    @Option(name = "packagePath",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to the write the archive.")
    public Path packagePath;

    @Option(name = "resourcesOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path to the write merged resources archive.")
    public Path resourcesOutput;

    @Option(name = "proguardOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path for the proguard file.")
    public Path proguardOutput;

    @Option(name = "mainDexProguardOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "Path for the main dex proguard file.")
    public Path mainDexProguardOutput;

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
        converter = VariantTypeConverter.class,
        category = "config",
        help = "Variant configuration type for packaging the resources."
            + " Acceptible values DEFAULT, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantType packageType;

    @Option(name = "densities",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of densities to filter the resource drawables by.")
    public List<String> densities;

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

    @Option(name = "prefilteredResources",
        defaultValue = "",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of resources that were filtered out in analysis.")
    public List<String> prefilteredResources;
  }

  private static AaptConfigOptions aaptConfigOptions;
  private static Options options;

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser = OptionsParser.newOptionsParser(
        Options.class, AaptConfigOptions.class);
    optionsParser.enableParamsFileSupport(FileSystems.getDefault());
    optionsParser.parseAndExitUponError(args);
    aaptConfigOptions = optionsParser.getOptions(AaptConfigOptions.class);
    options = optionsParser.getOptions(Options.class);

    final AndroidResourceProcessor resourceProcessor = new AndroidResourceProcessor(STD_LOGGER);
    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("android_resources_tmp")) {
      final Path tmp = scopedTmp.getPath();
      final Path mergedAssets = tmp.resolve("merged_assets");
      final Path mergedResources = tmp.resolve("merged_resources");
      final Path filteredResources = tmp.resolve("resources-filtered");
      final Path densityManifest = tmp.resolve("manifest-filtered/AndroidManifest.xml");
      final Path processedManifest = tmp.resolve("manifest-processed/AndroidManifest.xml");
      final Path dummyManifest = tmp.resolve("manifest-aapt-dummy/AndroidManifest.xml");

      Path generatedSources = null;
      if (options.srcJarOutput != null
          || options.rOutput != null
          || options.symbolsOut != null) {
        generatedSources = tmp.resolve("generated_resources");
      }

      logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      List<DependencyAndroidData> data =
          ImmutableSet.<DependencyAndroidData>builder()
              .addAll(options.directData)
              .addAll(options.transitiveData)
              .build()
              .asList();

      final MergedAndroidData mergedData =
          AndroidResourceMerger.mergeData(
              options.primaryData,
              options.directData,
              options.transitiveData,
              mergedResources,
              mergedAssets,
              selectPngCruncher(),
              options.packageType,
              options.symbolsOut,
              options.prefilteredResources);

      logger.fine(String.format("Merging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      final List<String> densitiesToFilter =
          options.prefilteredResources.isEmpty()
              ? options.densities
              : Collections.<String>emptyList();

      final DensityFilteredAndroidData filteredData =
          mergedData.filter(
              new DensitySpecificResourceFilter(
                  densitiesToFilter, filteredResources, mergedResources),
              new DensitySpecificManifestProcessor(options.densities, densityManifest));

      logger.fine(
          String.format(
              "Density filtering finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      MergedAndroidData processedData =
          AndroidManifestProcessor.with(STD_LOGGER)
              .processManifest(
                  options.packageType,
                  options.packageForR,
                  options.applicationId,
                  options.versionCode,
                  options.versionName,
                  filteredData,
                  processedManifest);

      // Write manifestOutput now before the dummy manifest is created.
      if (options.manifestOutput != null) {
        AndroidResourceOutputs.copyManifestToOutput(processedData, options.manifestOutput);
      }

      if (options.packageType == VariantType.LIBRARY) {
        resourceProcessor.writeDummyManifestForAapt(dummyManifest, options.packageForR);
        processedData = new MergedAndroidData(
            processedData.getResourceDir(),
            processedData.getAssetDir(),
            dummyManifest);
      }

      resourceProcessor.processResources(
          aaptConfigOptions.aapt,
          aaptConfigOptions.androidJar,
          aaptConfigOptions.buildToolsVersion,
          options.packageType,
          aaptConfigOptions.debug,
          options.packageForR,
          new FlagAaptOptions(aaptConfigOptions),
          aaptConfigOptions.resourceConfigs,
          aaptConfigOptions.splits,
          processedData,
          data,
          generatedSources,
          options.packagePath,
          options.proguardOutput,
          options.mainDexProguardOutput,
          options.resourcesOutput != null
              ? processedData.getResourceDir().resolve("values").resolve("public.xml")
              : null,
          options.dataBindingInfoOut);
      logger.fine(String.format("aapt finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      if (options.srcJarOutput != null) {
        AndroidResourceOutputs.createSrcJar(
            generatedSources, options.srcJarOutput, VariantType.LIBRARY == options.packageType);
      }
      if (options.rOutput != null) {
        AndroidResourceOutputs.copyRToOutput(
            generatedSources, options.rOutput, VariantType.LIBRARY == options.packageType);
      }
      if (options.resourcesOutput != null) {
        AndroidResourceOutputs.createResourcesZip(
            processedData.getResourceDir(),
            processedData.getAssetDir(),
            options.resourcesOutput,
            false /* compress */);
      }
      logger.fine(
          String.format("Packaging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (MergingException e) {
      logger.log(java.util.logging.Level.SEVERE, "Error during merging resources", e);
      throw e;
    } catch (IOException
        | InterruptedException
        | LoggedErrorException
        | UnrecognizedSplitsException e) {
      logger.log(java.util.logging.Level.SEVERE, "Error during processing resources", e);
      throw e;
    } catch (Exception e) {
      logger.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      throw e;
    } finally {
      resourceProcessor.shutdown();
    }
    logger.fine(String.format("Resources processed in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }

  private static boolean usePngCruncher() {
    // If the value was set, use that.
    if (aaptConfigOptions.useAaptCruncher != TriState.AUTO) {
      return aaptConfigOptions.useAaptCruncher == TriState.YES;
    }
    // By default png cruncher shouldn't be invoked on a library -- the work is just thrown away.
    return options.packageType != VariantType.LIBRARY;
  }

  private static PngCruncher selectPngCruncher() {
    // Use the full cruncher if asked to do so.
    if (usePngCruncher()) {
      return new AaptCruncher(
          aaptConfigOptions.aapt.toString(),
          new DefaultProcessExecutor(STD_LOGGER),
          new LoggedProcessOutputHandler(STD_LOGGER));
    }
    // Otherwise, if this is a binary, we need to at least process nine-patch PNGs.
    if (options.packageType != VariantType.LIBRARY) {
      return new NinePatchOnlyCruncher(
          aaptConfigOptions.aapt.toString(),
          new DefaultProcessExecutor(STD_LOGGER),
          new LoggedProcessOutputHandler(STD_LOGGER));
    }
    return null;
  }
}
