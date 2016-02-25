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

import com.google.common.base.Strings;
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

/**
 * Entry point for the {@code plmerge} tool, which merges the data from one or more plists into a
 * single binary plist. This tool's functionality is similar to that of the
 * {@code builtin-infoPlistUtility} in Xcode.
 *
 * <p>--control is a control protobuf.
 */
public class PlMerge {
  /**
   * Options for {@link PlMerge}.
   */
  public static class PlMergeOptions extends OptionsBase {
    @Option(
      name = "control",
      help = "Absolute path of the Control protobuf.",
      defaultValue = "null"
    )
    public String controlPath;
  }

  public static void main(String[] args) throws OptionsParsingException, IOException {
    FileSystem fileSystem = FileSystems.getDefault();
    OptionsParser parser = OptionsParser.newOptionsParser(PlMergeOptions.class);
    parser.parse(args);
    PlMergeOptions options = parser.getOptions(PlMergeOptions.class);

    if (options.controlPath == null) {
      missingArg("control");
    }

    InputStream in = Files.newInputStream(fileSystem.getPath(options.controlPath));
    Control control = Control.parseFrom(in);
    validateControl(control);

    PlistMerging merging =
        PlistMerging.from(
            control, new KeysToRemoveIfEmptyString("CFBundleIconFile", "NSPrincipalClass"));

    String primaryBundleId = Strings.emptyToNull(control.getPrimaryBundleId());
    String fallbackBundleId = Strings.emptyToNull(control.getFallbackBundleId());


    if (primaryBundleId != null || fallbackBundleId != null) {
      // Only set the bundle identifier if we were passed arguments to do so.
      // This prevents CFBundleIdentifiers being put into strings files.
      merging.setBundleIdentifier(primaryBundleId, fallbackBundleId);
    }
    merging.writePlist(fileSystem.getPath(control.getOutFile()));
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
}
