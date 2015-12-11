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

import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos.Control;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.List;

/**
 * Entry point for the {@code plmerge} tool, which merges the data from one or more plists into a
 * single binary plist. This tool's functionality is similar to that of the
 * {@code builtin-infoPlistUtility} in Xcode.
 *
 * <p>For backwards compatibility, PlMerge can consume either a control protobuf, passed using
 * --control, or the command line arguments --source_file, --out_file, --primary_bundle_id,
 * and --fallback_bundle_id.  If a --control is not provided, PlMerge will fall back on the other
 * command line arguments.  If --control is provided, all other command line arguments are ignored.
 */
public class PlMerge {
  /**
   * Options for {@link PlMerge}.
   */
  public static class PlMergeOptions extends OptionsBase {
    @Option(
      name = "source_file",
      help =
          "Paths to the plist files to merge. These can be binary, XML, or ASCII format. "
              + "Repeat this flag to specify multiple files. Required.",
      allowMultiple = true,
      defaultValue = "null"
    )
    public List<String> sourceFiles;

    @Option(name = "out_file", help = "Path to the output file. Required.", defaultValue = "null")
    public String outFile;

    @Option(
      name = "primary_bundle_id",
      help =
          "A reverse-DNS string identifier for this bundle associated with output binary "
              + "plist. This flag overrides the bundle id specified in field CFBundleIdentifier in "
              + "the associated plist file.",
      defaultValue = "null"
    )
    public String primaryBundleId;

    @Option(
      name = "fallback_bundle_id",
      help =
          "A fallback reverse-DNS string identifier for this bundle when the bundle "
              + "identifier is not specified in flag primary_bundle_id or associated plist file",
      defaultValue = "null"
    )
    public String fallbackBundleId;

    @Option(
      name = "control",
      help =
          "Absolute path of the Control protobuf. Data can be passed to plmerge through this "
              + "protobuf or through source_file, out_file, primary_bundle_id and "
              + "fallback_bundle_id.",
      defaultValue = "null"
    )
    public String controlPath;
  }

  public static void main(String[] args) throws OptionsParsingException, IOException {
    FileSystem fileSystem = FileSystems.getDefault();
    OptionsParser parser = OptionsParser.newOptionsParser(PlMergeOptions.class);
    parser.parse(args);
    PlMergeOptions options = parser.getOptions(PlMergeOptions.class);

    MergingArguments data = null;

    if (usingControlProtobuf(options)) {
      InputStream in = Files.newInputStream(fileSystem.getPath(options.controlPath));
      Control control = Control.parseFrom(in);
      validateControl(control);
      data = new MergingArguments(control);
    } else if (usingCommandLineArgs(options)) {
      data = new MergingArguments(options);
    } else {
      missingArg("Either --control or --out_file and at least one --source_file");
    }

    PlistMerging merging =
        PlistMerging.from(
            data, new KeysToRemoveIfEmptyString("CFBundleIconFile", "NSPrincipalClass"));
    if (data.getPrimaryBundleId() != null || data.getFallbackBundleId() != null) {
      // Only set the bundle identifier if we were passed arguments to do so.
      // This prevents CFBundleIdentifiers being put into strings files.
      merging.setBundleIdentifier(data.getPrimaryBundleId(), data.getFallbackBundleId());
    }
    merging.writePlist(fileSystem.getPath(data.getOutFile()));
  }

  private static void validateControl(Control control) {
    if (control.getSourceFileList().isEmpty()) {
      missingArg("At least one source_file");
    } else if (!control.hasOutFile()) {
      missingArg("out_file");
    }
  }

  private static void missingArg(String flag) {
    throw new IllegalArgumentException(
        flag + " is required:\n" + Options.getUsage(PlMergeOptions.class));
  }

  private static boolean usingControlProtobuf(PlMergeOptions options) {
    return options.controlPath != null;
  }

  private static boolean usingCommandLineArgs(PlMergeOptions options) {
    return (!options.sourceFiles.isEmpty()) && (options.outFile != null);
  }
}
