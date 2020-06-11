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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;

/**
 * Options that affect the <i>mechanism</i> of analysis. These are distinct from {@link
 * com.google.devtools.build.lib.analysis.config.BuildOptions}, which affect the <i>value</i> of a
 * BuildConfiguration.
 */
public class AnalysisOptions extends OptionsBase {
  @Option(
    name = "analysis_warnings_as_errors",
    deprecationWarning =
        "analysis_warnings_as_errors is now a no-op and will be removed in"
            + " an upcoming Blaze release",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.NO_OP},
    help = "Treat visible analysis warnings as errors."
  )
  public boolean analysisWarningsAsErrors;

  @Option(
    name = "discard_analysis_cache",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Discard the analysis cache immediately after the analysis phase completes."
            + " Reduces memory usage by ~10%, but makes further incremental builds slower."
  )
  public boolean discardAnalysisCache;

  @Option(
    name = "max_config_changes_to_show",
    defaultValue = "3",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "When discarding the analysis cache due to a change in the build options, "
        + "displays up to the given number of changed option names. "
        + "If the number given is -1, all changed options will be displayed."
  )
  public int maxConfigChangesToShow;

  @Option(
    name = "experimental_extra_action_filter",
    defaultValue = "",
    converter = RegexFilter.RegexFilterConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Filters set of targets to schedule extra_actions for."
  )
  public RegexFilter extraActionFilter;

  @Option(
    name = "experimental_extra_action_top_level_only",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Only schedules extra_actions for top level targets."
  )
  public boolean extraActionTopLevelOnly;

  @Option(
    name = "version_window_for_dirty_node_gc",
    defaultValue = "0",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Nodes that have been dirty for more than this many versions will be deleted"
            + " from the graph upon the next update. Values must be non-negative long integers,"
            + " or -1 indicating the maximum possible window."
  )
  public long versionWindowForDirtyNodeGc;

  @Deprecated
  @Option(
    name = "experimental_interleave_loading_and_analysis",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "No-op."
  )
  public boolean interleaveLoadingAndAnalysis;

  @Option(
    name = "experimental_skyframe_prepare_analysis",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    help = "Switches analysis preparation to a new code path based on Skyframe."
  )
  public boolean skyframePrepareAnalysis;

  @Option(
      name = "experimental_strict_conflict_checks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = OptionMetadataTag.INCOMPATIBLE_CHANGE,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "Check for action prefix file path conflicts, regardless of action-specific overrides.")
  public boolean strictConflictChecks;
}
