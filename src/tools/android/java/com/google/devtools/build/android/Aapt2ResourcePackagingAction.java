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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Streams.concat;
import static java.util.stream.Collectors.toList;

import com.android.builder.core.VariantTypeImpl;
import com.android.utils.StdLogger;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.android.Converters.AmpersandSplitter;
import com.google.devtools.build.android.Converters.ColonSplitter;
import com.google.devtools.build.android.Converters.CompatDependencyAndroidDataConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.CompatSerializedAndroidDataConverter;
import com.google.devtools.build.android.Converters.CompatUnvalidatedAndroidDataConverter;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.PackagedResources;
import com.google.devtools.build.android.aapt2.ProtoApk;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

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

  /** Flag specifications for this action. */
  @Parameters(separators = "= ")
  public static final class Options {
    @Parameter(
        names = "--primaryData",
        converter = CompatUnvalidatedAndroidDataConverter.class,
        description =
            "The directory containing the primary resource directory. The contents will override "
                + "the contents of any other resource directories during merging. The expected "
                + "format is "
                + UnvalidatedAndroidData.EXPECTED_FORMAT)
    public UnvalidatedAndroidData primaryData;

    @Parameter(
        names = "--data",
        converter = CompatDependencyAndroidDataConverter.class,
        description =
            "Transitive Data dependencies. These values will be used if not defined in the "
                + "primary resources. The expected format is "
                + DependencyAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<DependencyAndroidData> transitiveData = ImmutableList.of();

    @Parameter(
        names = "--directData",
        converter = CompatDependencyAndroidDataConverter.class,
        description =
            "Direct Data dependencies. These values will be used if not defined in the "
                + "primary resources. The expected format is "
                + DependencyAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<DependencyAndroidData> directData = ImmutableList.of();

    @Parameter(
        names = "--assets",
        converter = CompatSerializedAndroidDataConverter.class,
        splitter = AmpersandSplitter.class,
        description =
            "Transitive asset dependencies. These can also be specified together with resources"
                + " using --data. Expected format: "
                + SerializedAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<SerializedAndroidData> transitiveAssets = ImmutableList.of();

    @Parameter(
        names = "--directAssets",
        converter = CompatSerializedAndroidDataConverter.class,
        splitter = AmpersandSplitter.class,
        description =
            "Direct asset dependencies. These can also be specified together with resources using "
                + "--directData. Expected format: "
                + SerializedAndroidData.EXPECTED_FORMAT
                + "[,...]")
    public List<SerializedAndroidData> directAssets = ImmutableList.of();

    @Parameter(
        names = "--additionalApksToLinkAgainst",
        converter = CompatPathConverter.class,
        splitter = ColonSplitter.class,
        description = "List of APKs used during linking.")
    public List<Path> additionalApksToLinkAgainst = ImmutableList.of();

    @Parameter(
        names = "--packageId",
        description = "Resource ID prefix; see AAPT2 documentation for --package-id.")
    public int packageId = -1;

    @Parameter(
        names = "--rOutput",
        converter = CompatPathConverter.class,
        description = "Path to where the R.txt should be written.")
    public Path rOutput;

    @Parameter(
        names = {
          "--symbolsOut",
          "--symbolsTxtOut" // The old name of the flag.
        },
        converter = CompatPathConverter.class,
        description = "Path to where the symbols should be written.")
    public Path symbolsOut;

    @Parameter(
        names = "--dataBindingInfoOut",
        converter = CompatPathConverter.class,
        description = "Path to where data binding's layout info output should be written.")
    public Path dataBindingInfoOut;

    @Parameter(
        names = "--packagePath",
        converter = CompatPathConverter.class,
        description = "Path to the write the archive.")
    public Path packagePath;

    @Parameter(
        names = "--resourcesOutput",
        converter = CompatPathConverter.class,
        description = "Path to the write merged resources archive.")
    public Path resourcesOutput;

    @Parameter(
        names = "--proguardOutput",
        converter = CompatPathConverter.class,
        description = "Path for the proguard file.")
    public Path proguardOutput;

    @Parameter(
        names = "--mainDexProguardOutput",
        converter = CompatPathConverter.class,
        description = "Path for the main dex proguard file.")
    public Path mainDexProguardOutput;

    @Parameter(
        names = "--manifestOutput",
        converter = CompatPathConverter.class,
        description = "Path for the modified manifest.")
    public Path manifestOutput;

    @Parameter(
        names = "--srcJarOutput",
        converter = CompatPathConverter.class,
        description = "Path for the generated java source jar.")
    public Path srcJarOutput;

    @Parameter(
        names = "--packageType",
        description =
            "Variant configuration type for packaging the resources."
                + " Acceptable values BASE_APK, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantTypeImpl packageType = VariantTypeImpl.BASE_APK;

    @Parameter(
        names = "--densities",
        description = "A list of densities to filter the resource drawables by.")
    public List<String> densities = ImmutableList.of();

    @Parameter(
        names = "--packageForR",
        description = "Custom java package to generate the R symbols files.")
    public String packageForR;

    @Parameter(
        names = "--applicationId",
        description = "Custom application id (package manifest) for the packaged manifest.")
    public String applicationId;

    @Parameter(
        names = "--versionName",
        description = "Version name to stamp into the packaged manifest.")
    public String versionName;

    @Parameter(
        names = "--versionCode",
        description = "Version code to stamp into the packaged manifest.")
    public int versionCode = -1;

    @Parameter(
        names = "--throwOnResourceConflict",
        arity = 1,
        description =
            "If passed, resource merge conflicts will be treated as errors instead of warnings")
    public boolean throwOnResourceConflict;

    @Parameter(names = "--packageUnderTest", arity = 1, description = "Unused/deprecated option.")
    public String packageUnderTest;

    @Parameter(
        names = "--isTestWithResources",
        arity = 1,
        description = "Unused/deprecated option.")
    public boolean isTestWithResources;

    @Parameter(
        names = "--includeProguardLocationReferences",
        arity = 1,
        description = "When generating proguard configurations, include location references.")
    public boolean includeProguardLocationReferences;

    @Parameter(
        names = "--resourceApks",
        converter = CompatPathConverter.class,
        splitter = ColonSplitter.class,
        description = "List of reource only APK files to link against.")
    public List<Path> resourceApks = ImmutableList.of();
  }

  public static void main(String[] args) throws Exception {
    Profiler profiler = InMemoryProfiler.createAndStart("setup");
    Options options = new Options();
    Aapt2ConfigOptions aaptConfigOptions = new Aapt2ConfigOptions();
    ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
    Object[] allOptions = new Object[] {options, aaptConfigOptions, resourceProcessorCommonOptions};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

    Preconditions.checkArgument(
        options.packageId == -1 || (options.packageId >= 2 && options.packageId <= 255),
        "packageId must be in the range [2,255]");

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
                  options.dataBindingInfoOut,
                  options.packageForR,
                  databindingResourcesRoot,
                  aaptConfigOptions.useDataBindingAndroidX)
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
                              resourceProcessorCommonOptions.logWarnings))
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

      ImmutableList<StaticLibrary> resourceApks = ImmutableList.of();
      if (options.resourceApks != null) {
        resourceApks =
            options.resourceApks.stream().map(StaticLibrary::from).collect(toImmutableList());
      }

      final PackagedResources packagedResources =
          ResourceLinker.create(aaptConfigOptions.aapt2, executorService, linkedOut)
              .profileUsing(profiler)
              .customPackage(options.packageForR)
              .packageId(
                  options.packageId != -1 ? Optional.of(options.packageId) : Optional.empty())
              .outputAsProto(aaptConfigOptions.resourceTableAsProto)
              .dependencies(ImmutableList.copyOf(dependencies))
              .resourceApks(resourceApks)
              .include(compiledResourceDeps)
              .withAssets(assetDirs)
              .buildVersion(aaptConfigOptions.buildToolsVersion)
              .conditionalKeepRules(aaptConfigOptions.conditionalKeepRules == TriState.YES)
              .filterToDensity(options.densities)
              .storeUncompressed(aaptConfigOptions.uncompressedExtensions)
              .debug(aaptConfigOptions.debug)
              .includeGeneratedLocales(aaptConfigOptions.generatePseudoLocale)
              .includeOnlyConfigs(aaptConfigOptions.resourceConfigs)
              .includeProguardLocationReferences(options.includeProguardLocationReferences)
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
