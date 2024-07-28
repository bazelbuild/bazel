// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.ProtoApk;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * An action that will take a ResourcesZip and convert it into a proto APK.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/ConvertResourceZipToApkAction
 *       --aapt2 path to sdk/aapt2
 *       --androidJar path to sdk/androidJar
 *       --resources path to processed resources zip
 *       --outputApk path to write shrunk ap_
 * </pre>
 */
public final class ConvertResourceZipToApkAction {
  public static void main(String[] args) throws Exception {
    final Profiler profiler =
        LoggingProfiler.createAndStart("convert_proto_apk").startTask("flags");
    // Parse arguments.
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(
                Options.class, Aapt2ConfigOptions.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    Aapt2ConfigOptions aapt2ConfigOptions = optionsParser.getOptions(Aapt2ConfigOptions.class);
    Options options = optionsParser.getOptions(Options.class);
    System.out.println("WTF: " + options.resourcesZip);
    Preconditions.checkArgument(options.resourcesZip != null, "Missing input resource zip.");
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
      profiler.recordEndOf("setup").startTask("convert");
      ProtoApk inputApk = ProtoApk.readFrom(resourcesZip.asApk());
      Files.copy(
          aapt2ConfigOptions.resourceTableAsProto
              ? inputApk.asApkPath()
              : linker.convertProtoApkToBinary(inputApk),
          options.outputApk,
          StandardCopyOption.REPLACE_EXISTING);
      profiler.recordEndOf("convert");
    }
    profiler.recordEndOf("convert_proto_apk");
  }

  /** Extra options specific to {@link ConvertResourceZipToApkAction}. */
  public static class Options extends OptionsBase {
    @Option(
        name = "resources",
        defaultValue = "null",
        category = "input",
        converter = ExistingPathConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to the resources zip to be shrunk.")
    public Path resourcesZip;

    @Option(
        name = "outputApk",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to the output resource APK.")
    public Path outputApk;
  }
}
