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

package com.google.devtools.build.lib.starlarkdebug.module;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;

/** Configuration options for Starlark debugging. */
public final class StarlarkDebuggerOptions extends OptionsBase {
  @Option(
      name = "experimental_starlark_debug",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If true, Blaze will open the Starlark debug server at the start of the build "
              + "invocation, and wait for a debugger to attach before running the build.")
  public boolean debugStarlark;

  @Option(
      name = "experimental_starlark_debug_server_port",
      defaultValue = "7300",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "The port on which the Starlark debug server will listen for connections.")
  public int debugServerPort;

  @Option(
      name = "experimental_starlark_debug_verbose_logging",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Show verbose logs for the debugger.")
  public boolean verboseLogs;
}
