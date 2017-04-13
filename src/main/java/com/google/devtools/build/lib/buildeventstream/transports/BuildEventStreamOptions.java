// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream.transports;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

/** Options used to configure BuildEventStreamer and its BuildEventTransports. */
public class BuildEventStreamOptions extends OptionsBase {

  @Option(
    name = "experimental_build_event_text_file",
    defaultValue = "",
    category = "hidden",
    help = "If non-empty, write a textual representation of the build event protocol to that file"
  )
  public String buildEventTextFile;

  @Option(
    name = "experimental_build_event_binary_file",
    defaultValue = "",
    category = "hidden",
    help =
        "If non-empty, write a varint delimited binary representation of representation of the"
            + " build event protocol to that file."
  )
  public String buildEventBinaryFile;

  @Option(
    name = "experimental_build_event_text_file_path_conversion",
    defaultValue = "true",
    category = "hidden",
    help = "Convert paths in the text file representation of the build event protocol to more"
        + " globally valid URIs whenever possible; if disabled, the file:// uri scheme will always"
        + " be used"
  )
  public boolean buildEventTextFilePathConversion;

  @Option(
    name = "experimental_build_event_binary_file_path_conversion",
    defaultValue = "true",
    category = "hidden",
    help = "Convert paths in the binary file representation of the build event protocol to more"
        + " globally valid URIs whenever possible; if disabled, the file:// uri scheme will always"
        + " be used"
  )
  public boolean buildEventBinaryFilePathConversion;

  public String getBuildEventTextFile() {
    return buildEventTextFile;
  }

  public String getBuildEventBinaryFile() {
    return buildEventBinaryFile;
  }

  public boolean getBuildEventTextFilePathConversion() {
    return buildEventTextFilePathConversion;
  }

  public boolean getBuildEventBinaryFilePathConversion() {
    return buildEventBinaryFilePathConversion;
  }
}
