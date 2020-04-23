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

import static com.google.common.collect.Streams.concat;
import static java.util.stream.Collectors.toList;

import com.android.builder.core.VariantType;
import com.android.utils.StdLogger;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.android.Converters.DependencyAndroidDataListConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.build.android.Converters.SerializedAndroidDataListConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDataConverter;
import com.google.devtools.build.android.Converters.VariantTypeConverter;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.PackagedResources;
import com.google.devtools.build.android.aapt2.ProtoApk;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import com.google.devtools.common.options.TriState;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Provides an entry point for the resource processing using the AOSP build tools.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/Aapt2ResourcePackagingAction\
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
public class Aapt2ResourcePackagingAction {

  private static final StdLogger STD_LOGGER = new StdLogger(StdLogger.Level.WARNING);

  private static Aapt2ConfigOptions aaptConfigOptions;
  private static Options options;

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
        name = "additionalApksToLinkAgainst",
        defaultValue = "null",
        category = "input",
        converter = PathListConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "List of APKs used during linking.")
    public List<Path> additionalApksToLinkAgainst;

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
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        help = "Unused/deprecated option.")
    public String packageUnderTest;

    @Option(
        name = "isTestWithResources",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        help = "Unused/deprecated option.")
    public boolean isTestWithResources;

  }

  public static void main(String[] args) throws Exception {
    Profiler profiler = InMemoryProfiler.createAndStart("setup");
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(
                Options.class, Aapt2ConfigOptions.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    aaptConfigOptions = optionsParser.getOptions(Aapt2ConfigOptions.class);
    options = optionsParser.getOptions(Options.class);

    try (ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resources_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {
      final Path tmp = scopedTmp.getPath();
      final Path densityManifest = tmp.resolve("manifest-filtered/AndroidManifest.xml");
      final Path processedManifest = tmp.resolve("manifest-processed/AndroidManifest.xml");
      final Path symbols = tmp.resolve("symbols/symbols.bin");
      final Path databindingResourcesRoot =
          Files.createDirectories(tmp.resolve("android_data_binding_resources"));
      final Path compiledResources = Files.createDirectories(tmp.resolve("compiled"));
      final Path linkedOut = Files.createDirectories(tmp.resolve("linked"));
      final AndroidCompiledDataDeserializer dataDeserializer =
          AndroidCompiledDataDeserializer.create(/*includeFileContentsForValidation=*/ true);
      final ResourceCompiler compiler =
          ResourceCompiler.create(
              executorService,
              compiledResources,
              aaptConfigOptions.aapt2,
              aaptConfigOptions.buildToolsVersion,
              aaptConfigOptions.generatePseudoLocale);

      profiler.recordEndOf("setup").startTask("compile");
      CompiledResources compiled =
          options
              .primaryData
              .processDataBindings(
                  options.dataBindingInfoOut, options.packageForR, databindingResourcesRoot)
              .compile(compiler, compiledResources)
              .processManifest(
                  manifest ->
                      AndroidManifestProcessor.with(STD_LOGGER)
                          .processManifest(
                              options.applicationId,
                              options.versionCode,
                              options.versionName,
                              manifest,
                              processedManifest,
                              optionsParser.getOptions(ResourceProcessorCommonOptions.class)
                                  .logWarnings))
              .processManifest(
                  manifest ->
                      new DensitySpecificManifestProcessor(options.densities, densityManifest)
                          .process(manifest));

      profiler.recordEndOf("compile").startTask("merge");

      // Checks for merge conflicts, and write the merged data out.
      final Path symbolsBin =
          AndroidResourceMerger.mergeDataToSymbols(
              ParsedAndroidData.loadedFrom(
                  DependencyInfo.DependencyType.PRIMARY,
                  ImmutableList.of(SerializedAndroidData.from(compiled)),
                  executorService,
                  dataDeserializer),
              new DensitySpecificManifestProcessor(options.densities, densityManifest)
                  .process(options.primaryData.getManifest()),
              ImmutableList.<SerializedAndroidData>builder()
                  .addAll(options.directData)
                  .addAll(options.directAssets)
                  .build(),
              ImmutableList.<SerializedAndroidData>builder()
                  .addAll(options.transitiveData)
                  .addAll(options.transitiveAssets)
                  .build(),
              options.packageType,
              symbols,
              dataDeserializer,
              options.throwOnResourceConflict,
              executorService);
      if (options.symbolsOut != null) {
        Files.copy(symbolsBin, options.symbolsOut);
      }

      profiler.recordEndOf("merge").startTask("link");
      // Write manifestOutput now before the dummy manifest is created.
      if (options.manifestOutput != null) {
        AndroidResourceOutputs.copyManifestToOutput(compiled, options.manifestOutput);
      }

      List<CompiledResources> compiledResourceDeps =
          // Last defined dependencies will overwrite previous one, so always place direct
          // after transitive.
          concat(options.transitiveData.stream(), options.directData.stream())
              .map(DependencyAndroidData::getCompiledSymbols)
              .collect(toList());

      // NB: "-A" options are in *decreasing* precedence, while "-R" options are in *increasing*
      // precedence.  While this is internally inconsistent, it matches AAPTv1's treatment of "-A".
      List<Path> assetDirs =
          concat(
                  options.primaryData.assetDirs.stream(),
                  concat(
                          options.directData.stream(),
                          options.directAssets.stream(),
                          options.transitiveData.stream(),
                          options.transitiveAssets.stream())
                      .flatMap(dep -> dep.assetDirs.stream()))
              .collect(toList());

      List<StaticLibrary> dependencies =
          Lists.newArrayList(StaticLibrary.from(aaptConfigOptions.androidJar));
      if (options.additionalApksToLinkAgainst != null) {
        dependencies.addAll(
            options.additionalApksToLinkAgainst.stream()
                .map(StaticLibrary::from)
                .collect(toList()));
      }

      final PackagedResources packagedResources =
          ResourceLinker.create(aaptConfigOptions.aapt2, executorService, linkedOut)
              .profileUsing(profiler)
              .customPackage(options.packageForR)
              .outputAsProto(aaptConfigOptions.resourceTableAsProto)
              .dependencies(ImmutableList.copyOf(dependencies))
              .include(compiledResourceDeps)
              .withAssets(assetDirs)
              .buildVersion(aaptConfigOptions.buildToolsVersion)
              .conditionalKeepRules(aaptConfigOptions.conditionalKeepRules == TriState.YES)
              .filterToDensity(options.densities)
              .storeUncompressed(aaptConfigOptions.uncompressedExtensions)
              .debug(aaptConfigOptions.debug)
              .includeGeneratedLocales(aaptConfigOptions.generatePseudoLocale)
              .includeOnlyConfigs(aaptConfigOptions.resourceConfigs)
              .link(compiled);
      profiler.recordEndOf("link").startTask("validate");

      ValidateAndLinkResourcesAction.checkVisibilityOfResourceReferences(
          ProtoApk.readFrom(packagedResources.proto()).getManifest(),
          compiled,
          compiledResourceDeps);

      profiler.recordEndOf("validate");

      if (options.packagePath != null) {
        copy(packagedResources.apk(), options.packagePath);
      }
      if (options.proguardOutput != null) {
        copy(packagedResources.proguardConfig(), options.proguardOutput);
      }
      if (options.mainDexProguardOutput != null) {
        copy(packagedResources.mainDexProguard(), options.mainDexProguardOutput);
      }
      if (options.srcJarOutput != null) {
        AndroidResourceOutputs.createSrcJar(
            packagedResources.javaSourceDirectory(), options.srcJarOutput, /* staticIds= */ false);
      }
      if (options.rOutput != null) {
        copy(packagedResources.rTxt(), options.rOutput);
      }
      if (options.resourcesOutput != null) {
        packagedResources.asArchive().writeTo(options.resourcesOutput, /* compress= */ false);
      }
    }
  }

  private static void copy(Path from, Path out) throws IOException {
    Files.createDirectories(out.getParent());
    Files.copy(from, out);
  }
}
