// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.android.AndroidDataMerger.ContentComparingChecker;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.SerializedAndroidDataConverter;
import com.google.devtools.build.android.Converters.SerializedAndroidDataListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Logger;

/** An action that merges a library's assets (without using resources or manifests). */
public class AndroidAssetMergingAction extends AbstractBusyBoxAction {
  private static final Logger logger = Logger.getLogger(AndroidAssetMergingAction.class.getName());

  public static void main(String[] args) throws Exception {
    create().invoke(args);
  }

  @VisibleForTesting
  static void testingMain(String... args) throws Exception {
    create().invokeWithoutExit(args);
  }

  private static AndroidAssetMergingAction create() {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    return new AndroidAssetMergingAction(optionsParser);
  }

  private AndroidAssetMergingAction(OptionsParser optionsParser) {
    super(optionsParser, "Merge assets");
  }

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      name = "primaryData",
      defaultValue = "null",
      converter = SerializedAndroidDataConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "The assets of the current target. The expected format is "
              + SerializedAndroidData.EXPECTED_FORMAT
    )
    public SerializedAndroidData primary;

    @Option(
      name = "directData",
      defaultValue = "",
      converter = SerializedAndroidDataListConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Direct asset dependencies. These values will be used if not defined in the "
              + "primary assets. The expected format is "
              + SerializedAndroidData.EXPECTED_FORMAT
              + "[&...]"
    )
    public List<SerializedAndroidData> directData;

    @Option(
      name = "data",
      defaultValue = "",
      converter = SerializedAndroidDataListConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Transitive Data dependencies. These values will be used if not defined in the "
              + "primary assets. The expected format is "
              + SerializedAndroidData.EXPECTED_FORMAT
              + "[&...]"
    )
    public List<SerializedAndroidData> transitiveData;

    @Option(
      name = "assetsOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the write merged asset archive."
    )
    public Path assetsOutput;

    @Option(
      name = "throwOnAssetConflict",
      defaultValue = "true",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "If passed, asset merge conflicts will be treated as errors instead of warnings"
    )
    public boolean throwOnAssetConflict;
  }

  @Override
  void run(Path tmp, ExecutorServiceCloser executorService) throws Exception {
    Options options = getOptions(Options.class);
    Path mergedAssets = tmp.resolve("merged_assets");
    Path ignored = tmp.resolve("ignored");

    Preconditions.checkNotNull(options.primary);

    final ParsedAndroidData.Builder primaryBuilder = ParsedAndroidData.Builder.newBuilder();
    final AndroidParsedDataDeserializer deserializer = AndroidParsedDataDeserializer.create();
    options.primary.deserialize(
        DependencyInfo.DependencyType.PRIMARY, deserializer, primaryBuilder.consumers());
    ParsedAndroidData primaryData = primaryBuilder.build();

    UnwrittenMergedAndroidData unwrittenMergedData = AndroidResourceMerger.mergeData(
        executorService,
        options.transitiveData,
        options.directData,
        primaryData,
        /* primaryManifest = */ null,
        /* allowPrimaryOverrideAll = */ false,
        deserializer,
        options.throwOnAssetConflict,
        ContentComparingChecker.create());

    logCompletion("Merging");

    if (options.assetsOutput != null) {
      MergedAndroidData writtenMergedData =
          AndroidResourceMerger.writeMergedData(
              ignored,
              mergedAssets,
              /* cruncher = */ null,
              /* symbolsOut = */ null,
              /* rclassWriter = */ null,
              executorService,
              unwrittenMergedData);

      logCompletion("Writing");

      Preconditions.checkState(
          !Files.exists(ignored),
          "The asset merging action should not produce non-asset merge results!");

      ResourcesZip.from(ignored, writtenMergedData.getAssetDir())
          .writeTo(options.assetsOutput, /* compress= */ true);
      logCompletion("Create assets zip");
    }
  }

  @Override
  Logger getLogger() {
    return logger;
  }
}
