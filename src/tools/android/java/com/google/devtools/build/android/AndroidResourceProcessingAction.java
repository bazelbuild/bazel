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

import static java.util.stream.Collectors.toList;

import com.android.builder.core.VariantType;
import com.android.ide.common.internal.AaptCruncher;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.process.DefaultProcessExecutor;
import com.android.ide.common.process.LoggedProcessOutputHandler;
import com.android.ide.common.xml.AndroidManifestParser;
import com.android.ide.common.xml.ManifestData.Instrumentation;
import com.android.io.StreamException;
import com.android.utils.StdLogger;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.AndroidDataMerger.MergeConflictException;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.AndroidResourceProcessor.FlagAaptOptions;
import com.google.devtools.build.android.Converters.DependencyAndroidDataListConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.SerializedAndroidDataListConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDataConverter;
import com.google.devtools.build.android.Converters.VariantTypeConverter;
import com.google.devtools.build.android.SplitConfigurationFilter.UnrecognizedSplitsException;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import com.google.devtools.common.options.TriState;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

/**
 * Provides an entry point for the resource processing using the AOSP build tools.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/AndroidResourceProcessingAction\
 *      --sdkRoot path/to/sdk\
 *      --aapt path/to/sdk/aapt\
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
    @Option(
        name = "primaryData",
        defaultValue = "null",
        converter = UnvalidatedAndroidDataConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "The directory containing the primary resource directory. The contents will override "
                + "the contents of any other resource directories during merging. The expected "
                + "format is "
                + UnvalidatedAndroidData.EXPECTED_FORMAT)
    public UnvalidatedAndroidData primaryData;

    @Option(
        name = "data",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Transitive Data dependencies. These values will be used if not defined in the "
                + "primary resources. The expected format is "
                + DependencyAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<DependencyAndroidData> transitiveData;

    @Option(
        name = "directData",
        defaultValue = "",
        converter = DependencyAndroidDataListConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Direct Data dependencies. These values will be used if not defined in the "
                + "primary resources. The expected format is "
                + DependencyAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<DependencyAndroidData> directData;

    @Option(
        name = "assets",
        defaultValue = "",
        converter = SerializedAndroidDataListConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Transitive asset dependencies. These can also be specified together with resources"
                + " using --data. Expected format: "
                + SerializedAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<SerializedAndroidData> transitiveAssets;

    @Option(
        name = "directAssets",
        defaultValue = "",
        converter = SerializedAndroidDataListConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Direct asset dependencies. These can also be specified together with resources using "
                + "--directData. Expected format: "
                + SerializedAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<SerializedAndroidData> directAssets;

    @Option(
        name = "rOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to where the R.txt should be written.")
    public Path rOutput;

    @Option(
        name = "symbolsOut",
        oldName = "symbolsTxtOut",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to where the symbols should be written.")
    public Path symbolsOut;

    @Option(
        name = "dataBindingInfoOut",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to where data binding's layout info output should be written.")
    public Path dataBindingInfoOut;

    @Option(
        name = "packagePath",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to the write the archive.")
    public Path packagePath;

    @Option(
        name = "resourcesOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to the write merged resources archive.")
    public Path resourcesOutput;

    @Option(
        name = "proguardOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path for the proguard file.")
    public Path proguardOutput;

    @Option(
        name = "mainDexProguardOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path for the main dex proguard file.")
    public Path mainDexProguardOutput;

    @Option(
        name = "manifestOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path for the modified manifest.")
    public Path manifestOutput;

    @Option(
        name = "srcJarOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path for the generated java source jar.")
    public Path srcJarOutput;

    @Option(
        name = "packageType",
        defaultValue = "DEFAULT",
        converter = VariantTypeConverter.class,
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Variant configuration type for packaging the resources."
                + " Acceptable values DEFAULT, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantType packageType;

    @Option(
        name = "densities",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "A list of densities to filter the resource drawables by.")
    public List<String> densities;

    @Option(
        name = "packageForR",
        defaultValue = "null",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Custom java package to generate the R symbols files.")
    public String packageForR;

    @Option(
        name = "applicationId",
        defaultValue = "null",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Custom application id (package manifest) for the packaged manifest.")
    public String applicationId;

    @Option(
        name = "versionName",
        defaultValue = "null",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Version name to stamp into the packaged manifest.")
    public String versionName;

    @Option(
        name = "versionCode",
        defaultValue = "-1",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Version code to stamp into the packaged manifest.")
    public int versionCode;

    @Option(
        name = "prefilteredResources",
        defaultValue = "",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "A list of resources that were filtered out in analysis.")
    public List<String> prefilteredResources;

    @Option(
        name = "throwOnResourceConflict",
        defaultValue = "false",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "If passed, resource merge conflicts will be treated as errors instead of warnings")
    public boolean throwOnResourceConflict;

    @Option(
        name = "packageUnderTest",
        defaultValue = "null",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "When building a test APK, the package of the binary being tested. Android resources"
                + " can only be provided if there is no package under test or if the test"
                + " instrumentation is in a different package.")
    public String packageUnderTest;

    @Option(
        name = "isTestWithResources",
        defaultValue = "false",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Indicates that these resources are being processed for a test APK. Tests can only have"
                + "resources if they are not instrumented or they instrument only themselves.")
    public boolean isTestWithResources;
  }

  private static AaptConfigOptions aaptConfigOptions;
  private static Options options;

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, AaptConfigOptions.class);
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
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
      final Path dummyManifestDirectory = tmp.resolve("manifest-aapt-dummy");
      final Path publicXmlOut = tmp.resolve("public-resources/public.xml");

      Path generatedSources = null;
      if (options.srcJarOutput != null || options.rOutput != null || options.symbolsOut != null) {
        generatedSources = tmp.resolve("generated_resources");
      }

      logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      List<DependencyAndroidData> resourceData =
          ImmutableSet.<DependencyAndroidData>builder()
              .addAll(options.directData)
              .addAll(options.transitiveData)
              .build()
              .asList();

      final MergedAndroidData mergedData =
          AndroidResourceMerger.mergeDataAndWrite(
              options.primaryData,
              ImmutableList.<SerializedAndroidData>builder()
                  .addAll(options.directData)
                  .addAll(options.directAssets)
                  .build(),
              ImmutableList.<SerializedAndroidData>builder()
                  .addAll(options.transitiveData)
                  .addAll(options.transitiveAssets)
                  .build(),
              mergedResources,
              mergedAssets,
              selectPngCruncher(),
              options.packageType,
              options.symbolsOut,
              options.prefilteredResources,
              options.throwOnResourceConflict);

      logger.fine(String.format("Merging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      final DensityFilteredAndroidData filteredData =
          mergedData.filter(
              // Even if filtering was done in analysis, we still need to filter by density again
              // in execution since Fileset contents are not available in analysis.
              new DensitySpecificResourceFilter(
                  options.densities, filteredResources, mergedResources),
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
        processedData =
            new MergedAndroidData(
                processedData.getResourceDir(),
                processedData.getAssetDir(),
                AndroidManifest.parseFrom(processedData.getManifest())
                    .writeDummyManifestForAapt(dummyManifestDirectory, options.packageForR));
      }

      if (hasConflictWithPackageUnderTest(
          options.packageUnderTest, processedData.getManifest(), timer)) {
        logger.log(
            Level.SEVERE,
            "Android resources cannot be provided if the instrumentation package is the same as "
                + "the package under test, but the instrumentation package (in the manifest) and "
                + "the package under test both had the same package: "
                + options.packageUnderTest);
        System.exit(1);
      }

      MergedAndroidData processedAndroidData =
          resourceProcessor.processResources(
              tmp,
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
              resourceData,
              generatedSources,
              options.packagePath,
              options.proguardOutput,
              options.mainDexProguardOutput,
              options.resourcesOutput != null ? publicXmlOut : null,
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
        if (Files.exists(publicXmlOut)) {
          try (BufferedReader reader =
              Files.newBufferedReader(publicXmlOut, StandardCharsets.UTF_8)) {
            Path publicXml =
                processedAndroidData.getResourceDir().resolve("values").resolve("public.xml");
            Files.createDirectories(publicXml.getParent());

            Pattern xmlComment = Pattern.compile("<!--.*-->");
            Files.write(
                publicXml,
                // Remove aapt debugging comment lines to fix hermaticity with generated files.
                reader.lines().filter(l -> !xmlComment.matcher(l).find()).collect(toList()),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
          }
        }

        ResourcesZip.from(processedAndroidData.getResourceDir(), processedAndroidData.getAssetDir())
            .writeTo(options.resourcesOutput, /* compress= */ false);
      }
      logger.fine(
          String.format("Packaging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (MergeConflictException e) {
      logger.severe(e.getMessage());
      System.exit(1);
    } catch (MergingException e) {
      logger.log(java.util.logging.Level.SEVERE, "Error during merging resources", e);
      throw e;
    } catch (IOException
        | InterruptedException
        | LoggedErrorException
        | UnrecognizedSplitsException e) {
      logger.log(java.util.logging.Level.SEVERE, "Error during processing resources", e);
      throw e;
    } catch (AndroidManifestProcessor.ManifestProcessingException e) {
      System.exit(1);
    } catch (Exception e) {
      logger.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      throw e;
    } finally {
      resourceProcessor.shutdown();
    }
    logger.fine(String.format("Resources processed in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }

  /**
   * Checks if there is a conflict between the package under test and the package being built.
   *
   * <p>When testing Android code, the test can be run in the same or a different process as the
   * code being tested. If it's in the same process, we do not allow Android resources to be used by
   * the test, as they could overwrite the resources used by the code being tested. If this APK
   * won't be testing another APK, the test and code under test are in different processes, or no
   * resources are being used, this isn't a concern.
   *
   * <p>To determine whether the test and code under test are run in the same process, we check the
   * package of the code under test, passed into this function, against the target packages of any
   * <code>instrumentation</code> tags in this APK's manifest.
   *
   * @param packageUnderTest the package of the code under test, or null if no code is under test
   * @param processedManifest the processed manifest for this APK
   * @return true if there is a conflict, false otherwise
   */
  @VisibleForTesting
  static boolean hasConflictWithPackageUnderTest(
      @Nullable String packageUnderTest, Path processedManifest, Stopwatch timer)
      throws SAXException, StreamException, ParserConfigurationException, IOException {
    if (packageUnderTest == null) {
      return false;
    }

    // We are building a test APK with resources. Validate instrumentation package is different
    // from the package under test. If it isn't, fail to prevent the test resources from
    // overriding the resources of the APK under test.
    try (InputStream stream = Files.newInputStream(processedManifest)) {
      for (Instrumentation instrumentation :
          AndroidManifestParser.parse(stream).getInstrumentations()) {
        if (packageUnderTest.equals(instrumentation.getTargetPackage())) {
          return true;
        }
      }
    }

    logger.fine(
        String.format(
            "Custom package and instrumentation verification finished at %sms",
            timer.elapsed(TimeUnit.MILLISECONDS)));
    return false;
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
