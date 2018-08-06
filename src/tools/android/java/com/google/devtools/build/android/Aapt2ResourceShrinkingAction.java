// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.util.stream.Collectors.toSet;

import com.android.builder.core.VariantConfiguration;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.ResourceShrinkerAction.Options;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.File;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;

/**
 * An action to perform resource shrinking using the Gradle resource shrinker.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/ResourceShrinkerAction
 *       --aapt2 path to sdk/aapt2
 *       --androidJar path to sdk/androidJar
 *       --shrunkJar path to proguard dead code removal jar
 *       --resources path to processed resources zip
 *       --rTxt path to processed resources R.txt
 *       --primaryManifest path to processed resources AndroidManifest.xml
 *       --dependencyManifest path to dependency library manifest (repeated flag)
 *       --shrunkResourceApk path to write shrunk ap_
 *       --shrunkResources path to write shrunk resources zip
 * </pre>
 */
public class Aapt2ResourceShrinkingAction {

  /** Aapt2 shrinking specific options */
  public static final class Aapt2ShrinkOptions extends OptionsBase {
    @Option(
        name = "useProtoApk",
        defaultValue = "true",
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to the shrunk jar from a Proguard run with shrinking enabled.")
    public boolean useProtoApk;
  }

  public static void main(String[] args) throws Exception {
    final Profiler profiler = LoggingProfiler.createAndStart("shrink").startTask("flags");
    // Parse arguments.
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(
            ImmutableList.of(Options.class, Aapt2ConfigOptions.class, Aapt2ShrinkOptions.class));
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    optionsParser.parseAndExitUponError(args);
    Aapt2ConfigOptions aapt2ConfigOptions = optionsParser.getOptions(Aapt2ConfigOptions.class);
    Options options = optionsParser.getOptions(Options.class);
    Aapt2ShrinkOptions aapt2ShrinkOptions = optionsParser.getOptions(Aapt2ShrinkOptions.class);
    profiler.recordEndOf("flags").startTask("setup");

    try (ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resources_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {

      final ResourcesZip resourcesZip =
          ResourcesZip.createFrom(
              options.resourcesZip, scopedTmp.subDirectoryOf("merged-resources"));
      Path workingResourcesDirectory = scopedTmp.subDirectoryOf("resources");
      final ResourceLinker linker =
          ResourceLinker.create(
                  aapt2ConfigOptions.aapt2, executorService, scopedTmp.subDirectoryOf("linking"))
              .profileUsing(profiler)
              .dependencies(ImmutableList.of(StaticLibrary.from(aapt2ConfigOptions.androidJar)));

      final Set<String> packages =
          options
              .dependencyManifests
              .stream()
              .map(Path::toFile)
              .map(manifestToPackageUsing(executorService))
              .map(futureToString())
              .collect(toSet());

      if (aapt2ShrinkOptions.useProtoApk) {
        resourcesZip
            .shrinkUsingProto(
                packages,
                options.shrunkJar,
                options.proguardMapping,
                options.log,
                scopedTmp.subDirectoryOf("shrunk-resources"))
            .writeBinaryTo(linker, options.shrunkApk, aapt2ConfigOptions.resourceTableAsProto)
            .writeReportTo(options.log)
            .writeResourcesToZip(options.shrunkResources);
        if (options.rTxtOutput != null) {
          // Fufill the contract -- however, we do not generate an R.txt from the shrunk resources.
          Files.copy(options.rTxt, options.rTxtOutput);
        }
      } else {
        final ResourceCompiler resourceCompiler =
            ResourceCompiler.create(
                executorService,
                workingResourcesDirectory,
                aapt2ConfigOptions.aapt2,
                aapt2ConfigOptions.buildToolsVersion,
                aapt2ConfigOptions.generatePseudoLocale);
        profiler.recordEndOf("setup").startTask("compile");

        final CompiledResources compiled =
            resourcesZip
                .shrink(
                    packages,
                    options.rTxt,
                    options.shrunkJar,
                    options.primaryManifest,
                    options.proguardMapping,
                    options.log,
                    scopedTmp.subDirectoryOf("shrunk-resources"))
                .writeArchiveTo(options.shrunkResources, false)
                .compile(resourceCompiler, workingResourcesDirectory);
        profiler.recordEndOf("compile");
        linker
            .dependencies(ImmutableList.of(StaticLibrary.from(aapt2ConfigOptions.androidJar)))
            .profileUsing(profiler)
            .outputAsProto(aapt2ConfigOptions.resourceTableAsProto)
            .buildVersion(aapt2ConfigOptions.buildToolsVersion)
            .includeOnlyConfigs(aapt2ConfigOptions.resourceConfigs)
            .debug(aapt2ConfigOptions.debug)
            .link(compiled)
            .copyPackageTo(options.shrunkApk)
            .copyRTxtTo(options.rTxtOutput);
      }
      profiler.recordEndOf("shrink");
    }
  }

  static Function<File, ListenableFuture<String>> manifestToPackageUsing(
      ListeningExecutorService executor) {
    return f -> executor.submit(() -> VariantConfiguration.getManifestPackage(f));
  }

  static Function<ListenableFuture<String>, String> futureToString() {
    return f -> {
      try {
        return f.get();
      } catch (InterruptedException | ExecutionException e) {
        throw new RuntimeException(e);
      }
    };
  }
}
