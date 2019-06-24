// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDirectoriesConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Parses android resource XML files from a directory and serializes the results to a protobuf.
 * Other actions that depend on the same resource directory can then load the data from a single
 * protobuf instead of reparsing multiple tiny XML files from source (proto loading is faster).
 */
public class AndroidResourceParsingAction {

  private static final Logger logger =
      Logger.getLogger(AndroidResourceParsingAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      name = "primaryData",
      defaultValue = "null",
      converter = UnvalidatedAndroidDirectoriesConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      category = "input",
      help =
          "The resource and asset directories to parse and summarize in a symbols file."
              + " The expected format is "
              + UnvalidatedAndroidDirectories.EXPECTED_FORMAT
    )
    public UnvalidatedAndroidDirectories primaryData;

    @Option(
      name = "output",
      defaultValue = "null",
      converter = PathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      category = "output",
      help = "Path to write the output protobuf."
    )
    public Path output;
  }

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);

    Preconditions.checkNotNull(options.primaryData);
    Preconditions.checkNotNull(options.output);

    final Stopwatch timer = Stopwatch.createStarted();
    ParsedAndroidData parsedPrimary = ParsedAndroidData.from(options.primaryData);
    logger.fine(String.format("Walked XML tree at %dms", timer.elapsed(TimeUnit.MILLISECONDS)));
    UnwrittenMergedAndroidData unwrittenData =
        UnwrittenMergedAndroidData.of(
            null, parsedPrimary, ParsedAndroidData.from(ImmutableList.<DependencyAndroidData>of()));
    AndroidDataSerializer serializer = AndroidDataSerializer.create();
    unwrittenData.serializeTo(serializer);
    serializer.flushTo(options.output);
    logger.fine(
        String.format("Finished parse + serialize in %dms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }
}
