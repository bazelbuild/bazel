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

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.CompatUnvalidatedAndroidDirectoriesConverter;
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
  @Parameters(separators = "= ")
  public static final class Options {

    @Parameter(
        names = "--primaryData",
        converter = CompatUnvalidatedAndroidDirectoriesConverter.class,
        description =
            "The resource and asset directories to parse and summarize in a symbols file."
                + " The expected format is "
                + UnvalidatedAndroidDirectories.EXPECTED_FORMAT)
    public UnvalidatedAndroidDirectories primaryData;

    @Parameter(
        names = "--output",
        converter = CompatPathConverter.class,
        description = "Path to write the output protobuf.")
    public Path output;
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
    Object[] allOptions = new Object[] {options, resourceProcessorCommonOptions};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(options, preprocessedArgs);
    jc.parse(normalizedArgs);

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
