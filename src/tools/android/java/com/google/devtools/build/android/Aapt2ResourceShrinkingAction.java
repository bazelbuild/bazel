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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.ResourceShrinkerAction.Options;
import com.google.devtools.build.android.ResourcesZip.ShrunkProtoApk;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.Set;

/**
 * An action that will analyze and remove unused resources from a ResourcesZip.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/Aapt2ResourceShrinkingAction
 *       --aapt2 path to sdk/aapt2
 *       --androidJar path to sdk/androidJar
 *       --shrunkJar path to proguard dead code removal jar
 *       --resources path to processed resources zip
 *       --rTxt path to processed resources R.txt
 *       --shrunkResourceApk path to write shrunk ap_
 *       --shrunkResources path to write shrunk resources zip
 * </pre>
 */
public class Aapt2ResourceShrinkingAction {

  public static void main(String[] args) throws Exception {
    final Profiler profiler = LoggingProfiler.createAndStart("shrink").startTask("flags");
    // Parse arguments.
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(ImmutableList.of(Options.class, Aapt2ConfigOptions.class));
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    optionsParser.parseAndExitUponError(args);
    Aapt2ConfigOptions aapt2ConfigOptions = optionsParser.getOptions(Aapt2ConfigOptions.class);
    Options options = optionsParser.getOptions(Options.class);
    profiler.recordEndOf("flags").startTask("setup");

    try (ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resources_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {

      final ResourcesZip resourcesZip =
          ResourcesZip.createFrom(
              options.resourcesZip, scopedTmp.subDirectoryOf("merged-resources"));
      final ResourceLinker linker =
          ResourceLinker.create(
                  aapt2ConfigOptions.aapt2, executorService, scopedTmp.subDirectoryOf("linking"))
              .profileUsing(profiler)
              .dependencies(ImmutableList.of(StaticLibrary.from(aapt2ConfigOptions.androidJar)));

      final Set<String> packages = new HashSet<>(resourcesZip.asPackages());

      profiler.recordEndOf("setup").startTask("resourceShrinker");

      try (final ShrunkProtoApk shrunk =
          resourcesZip.shrinkUsingProto(
              packages,
              options.shrunkJar,
              options.proguardMapping,
              options.log,
              scopedTmp.subDirectoryOf("shrunk-resources"))) {
        shrunk
            .writeBinaryTo(linker, options.shrunkApk, aapt2ConfigOptions.resourceTableAsProto)
            .writeReportTo(options.log)
            .writeResourcesToZip(options.shrunkResources);
        if (options.rTxtOutput != null) {
          // Fulfill the contract -- however, we do not generate an R.txt from the shrunk
          // resources.
          Files.copy(options.rTxt, options.rTxtOutput);
        }
      }
      profiler.recordEndOf("resourceShrinker").recordEndOf("shrink");
    }
  }
}
