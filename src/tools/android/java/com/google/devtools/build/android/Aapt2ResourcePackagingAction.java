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

import com.android.utils.StdLogger;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.AndroidResourceProcessingAction.Options;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.common.options.OptionsParser;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
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
 *   java/com/google/build/android/Aapt2ResourcePackagingAction\
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
public class Aapt2ResourcePackagingAction {

  private static final StdLogger STD_LOGGER = new StdLogger(StdLogger.Level.WARNING);

  private static final Logger logger =
      Logger.getLogger(Aapt2ResourcePackagingAction.class.getName());

  private static Aapt2ConfigOptions aaptConfigOptions;
  private static Options options;

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, Aapt2ConfigOptions.class);
    optionsParser.enableParamsFileSupport(FileSystems.getDefault());
    optionsParser.parseAndExitUponError(args);
    aaptConfigOptions = optionsParser.getOptions(Aapt2ConfigOptions.class);
    options = optionsParser.getOptions(Options.class);

    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("android_resources_tmp")) {
      final Path tmp = scopedTmp.getPath();
      final Path mergedAssets = tmp.resolve("merged_assets");
      final Path mergedResources = tmp.resolve("merged_resources");

      final Path densityManifest = tmp.resolve("manifest-filtered/AndroidManifest.xml");

      final Path processedManifest = tmp.resolve("manifest-processed/AndroidManifest.xml");
      final Path dummyManifest = tmp.resolve("manifest-aapt-dummy/AndroidManifest.xml");
      final Path databindingResourcesRoot =
          Files.createDirectories(tmp.resolve("android_data_binding_resources"));
      final Path databindingMetaData =
          Files.createDirectories(tmp.resolve("android_data_binding_metadata"));
      final Path compiledResources = Files.createDirectories(tmp.resolve("compiled"));
      final Path staticLinkedOut = Files.createDirectories(tmp.resolve("static-linked"));
      final Path linkedOut = Files.createDirectories(tmp.resolve("linked"));

      Path generatedSources = null;
      if (options.srcJarOutput != null || options.rOutput != null || options.symbolsOut != null) {
        generatedSources = tmp.resolve("generated_resources");
      }

      logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      List<DependencyAndroidData> data =
          ImmutableSet.<DependencyAndroidData>builder()
              .addAll(options.directData)
              .addAll(options.transitiveData)
              .build()
              .asList();

      // Checks for merge conflicts.
      MergedAndroidData mergedAndroidData =
          AndroidResourceMerger.mergeData(
              options.primaryData,
              options.directData,
              options.transitiveData,
              mergedResources,
              mergedAssets,
              null /* cruncher. Aapt2 automatically chooses to crunch or not. */,
              options.packageType,
              options.symbolsOut,
              options.prefilteredResources,
              false /* throwOnResourceConflict */);

      logger.fine(String.format("Merging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      final List<String> densitiesToFilter =
          options.prefilteredResources.isEmpty()
              ? options.densities
              : Collections.<String>emptyList();
      final ListeningExecutorService executorService = ExecutorServiceCloser.createDefaultService();
      try (final Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
        final ResourceCompiler compiler =
            ResourceCompiler.create(
                executorService,
                compiledResources,
                aaptConfigOptions.aapt2,
                aaptConfigOptions.buildToolsVersion);

        CompiledResources compiled =
            options
                .primaryData
                .processDataBindings(options.dataBindingInfoOut, options.packageForR,
                    databindingResourcesRoot)
                .compile(compiler, compiledResources)
                .processManifest(
                    manifest ->
                        AndroidManifestProcessor.with(STD_LOGGER)
                            .processManifest(
                                options.applicationId,
                                options.versionCode,
                                options.versionName,
                                manifest,
                                processedManifest))
                .processManifest(
                    manifest ->
                        new DensitySpecificManifestProcessor(options.densities, densityManifest)
                            .process(manifest));

        // Write manifestOutput now before the dummy manifest is created.
        if (options.manifestOutput != null) {
          AndroidResourceOutputs.copyManifestToOutput(compiled, options.manifestOutput);
        }

        List<StaticLibrary> dependencies =
            // Last defined dependencies will overwrite previous one, so always place direct
            // after transitive.
            concat(options.transitiveData.stream(), options.directData.stream())
                .map(DependencyAndroidData::getStaticLibrary)
                .collect(toList());

        ResourceLinker.create(aaptConfigOptions.aapt2, linkedOut)
            .dependencies(ImmutableList.of(StaticLibrary.from(aaptConfigOptions.androidJar)))
            .include(dependencies)
            .buildVersion(aaptConfigOptions.buildToolsVersion)
            .filterToDensity(densitiesToFilter)
            .link(compiled)
            .copyPackageTo(options.packagePath)
            .copyProguardTo(options.proguardOutput)
            .copyMainDexProguardTo(options.mainDexProguardOutput)
            .createSourceJar(options.srcJarOutput)
            .copyRTxtTo(options.rOutput);
        logger.fine(String.format("aapt2 finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      }
      if (options.resourcesOutput != null) {
        // The compiled resources and the merged resources should be the same.
        // TODO(corysmith): Decompile or otherwise provide the exact resources in the apk.
        AndroidResourceOutputs.createResourcesZip(
            mergedAndroidData.getResourceDir(),
            mergedAndroidData.getAssetDir(),
            options.resourcesOutput,
            false /* compress */);
      }
      logger.fine(
          String.format("Packaging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (MergingException e) {
      logger.log(java.util.logging.Level.SEVERE, "Error during merging resources", e);
      throw e;
    } catch (IOException e) {
      logger.log(java.util.logging.Level.SEVERE, "Error during processing resources", e);
      throw e;
    } catch (Exception e) {
      logger.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      throw e;
    }
    logger.fine(String.format("Resources processed in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }
}
