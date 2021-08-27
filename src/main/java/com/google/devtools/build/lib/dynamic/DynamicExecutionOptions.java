// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.dynamic;

import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import java.util.Map;

/**
 * Options related to dynamic spawn execution.
 */
public class DynamicExecutionOptions extends OptionsBase {

  @Option(
      name = "experimental_spawn_scheduler",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      help =
          "Enable dynamic execution by running actions locally and remotely in parallel. Bazel "
              + "spawns each action locally and remotely and picks the one that completes first. "
              + "If an action supports workers, the local action will be run in the persistent "
              + "worker mode. To enable dynamic execution for an individual action mnemonic, use "
              + "the `--internal_spawn_scheduler` and `--strategy=<mnemonic>=dynamic` flags "
              + "instead.",
      expansion = {
        "--internal_spawn_scheduler",
        "--spawn_strategy=dynamic",
      })
  public Void experimentalSpawnScheduler;

  @Option(
      name = "internal_spawn_scheduler",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "false",
      help =
          "Placeholder option so that we can tell in Blaze whether the spawn scheduler was "
              + "enabled.")
  public boolean internalSpawnScheduler;

  @Option(
      name = "dynamic_local_strategy",
      converter = Converters.StringToStringListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      allowMultiple = true,
      help =
          "The local strategies, in order, to use for the given mnemonic. Passing"
              + " 'local' as the mnemonic sets the default for unspecified mnemonics. Takes"
              + " [mnemonic=]local_strategy[,local_strategy,...]")
  public List<Map.Entry<String, List<String>>> dynamicLocalStrategy;

  @Option(
      name = "dynamic_remote_strategy",
      converter = Converters.StringToStringListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      allowMultiple = true,
      help =
          "The remote strategies to use for the given mnemonic. Passing 'remote'"
              + " as the mnemonic sets the default for unspecified mnemonics. Takes"
              + " [mnemonic=]remote_strategy[,remote_strategy,...]")
  public List<Map.Entry<String, List<String>>> dynamicRemoteStrategy;

  @Option(
      name = "dynamic_worker_strategy",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.DEPRECATED},
      defaultValue = "",
      help = "Deprecated. Please use --dynamic_local_strategy=worker_strategy,local_strategy.")
  public String dynamicWorkerStrategy;

  @Option(
    name = "experimental_local_execution_delay",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    defaultValue = "1000",
    help =
        "How many milliseconds should local execution be delayed, if remote execution was faster"
            + " during a build at least once?"
  )
  public int localExecutionDelay;

  @Option(
    name = "experimental_debug_spawn_scheduler",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    defaultValue = "false"
  )
  public boolean debugSpawnScheduler;

  @Option(
      name = "experimental_require_availability_info",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "false",
      help =
          "If true, fail the build if there are actions that set requires-darwin but do not have"
              + "Xcode availability-related execution requirements set.")
  public boolean requireAvailabilityInfo;

  @Option(
      name = "experimental_availability_info_exempt",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "Genrule,TestRunner",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      help =
          "A comma-separated list of mnemonics that are not required to have Xcode-related "
              + "execution info if --experimental_require_availability_info=true. No-op if "
              + "--experimental_require_availability_info=false.")
  public List<String> availabilityInfoExempt;
}
