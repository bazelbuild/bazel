// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.xcode.plmerge;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import com.dd.plist.NSObject;

import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Entry point for the {@code plmerge} tool, which merges the data from one or more plists into a
 * single binary plist. This tool's functionality is similar to that of the
 * {@code builtin-infoPlistUtility} in Xcode.
 */
public class PlMerge {
  /**
   * Options for {@link PlMerge}.
   */
  public static class PlMergeOptions extends OptionsBase {
    @Option(
        name = "source_file",
        help = "Paths to the plist files to merge. These can be binary, XML, or ASCII format. "
            + "Repeat this flag to specify multiple files. Required.",
        allowMultiple = true,
        defaultValue = "null")
    public List<String> sourceFiles;

    @Option(
        name = "out_file",
        help = "Path to the output file. Required.",
        defaultValue = "null")
    public String outFile;

    @Option(
        name = "primary_bundle_id",
        help = "A reverse-DNS string identifier for this bundle associated with output binary "
            + "plist. This flag overrides the bundle id specified in field CFBundleIdentifier in "
            + "the associated plist file.",
        defaultValue = "null")
    public String primaryBundleId;

    @Option(
        name = "fallback_bundle_id",
        help = "A fallback reverse-DNS string identifier for this bundle when the bundle "
            + "identifier is not specified in flag primary_bundle_id or associated plist file",
        defaultValue = "null")
    public String fallbackBundleId;
  }

  public static void main(String[] args) throws IOException, OptionsParsingException {
    OptionsParser parser = OptionsParser.newOptionsParser(PlMergeOptions.class);
    parser.parse(args);
    PlMergeOptions options = parser.getOptions(PlMergeOptions.class);
    if (options.sourceFiles.isEmpty()) {
      missingArg("At least one --source_file");
    }
    if (options.outFile == null) {
      missingArg("--out_file");
    }
    FileSystem fileSystem = FileSystems.getDefault();

    List<Path> sourceFilePaths = new ArrayList<>();
    for (String sourceFile : options.sourceFiles) {
      sourceFilePaths.add(fileSystem.getPath(sourceFile));
    }

    PlistMerging merging = PlistMerging.from(sourceFilePaths, ImmutableMap.<String, NSObject>of(),
        ImmutableMap.<String, String>of(), new KeysToRemoveIfEmptyString());
    if (options.primaryBundleId != null || options.fallbackBundleId != null) {
      // Only set the bundle identifier if we were passed arguments to do so.
      // This prevents CFBundleIdentifiers being put into strings files.
      merging.setBundleIdentifier(options.primaryBundleId, options.fallbackBundleId);
    }
    merging.writePlist(fileSystem.getPath(options.outFile));
  }

  private static void missingArg(String flag) {
    throw new IllegalArgumentException(flag + " is required:\n"
        + Options.getUsage(PlMergeOptions.class));
  }
}
