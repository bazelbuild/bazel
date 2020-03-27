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

package com.google.devtools.build.lib.rules.apple.swift;

import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import java.util.List;

/** Command-line options for building with Swift tools. */
public class SwiftCommandLineOptions extends FragmentOptions {
  @Option(
      name = "swiftcopt",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help = "Additional options to pass to Swift compilation.")
  public List<String> copts;

  @Option(
      name = "host_swiftcopt",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Additional options to pass to swiftc for host tools.")
  public List<String> hostSwiftcoptList;

  @Override
  public FragmentOptions getHost() {
    SwiftCommandLineOptions host = (SwiftCommandLineOptions) super.getHost();
    host.copts = this.hostSwiftcoptList;

    // Save host options in case of a further exec->host transition.
    host.hostSwiftcoptList = hostSwiftcoptList;

    return host;
  }
}
