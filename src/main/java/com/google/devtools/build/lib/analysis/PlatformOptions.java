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

package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.EmptyToNullLabelConverter;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelListConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;
import java.util.Map;

/** Command-line options for platform-related configuration. */
public class PlatformOptions extends FragmentOptions {

  /**
   * Main workspace-relative location to use when the user does not explicitly set {@code
   * --platform_mappings}.
   */
  public static final PathFragment DEFAULT_PLATFORM_MAPPINGS =
      PathFragment.create("platform_mappings");

  private static final ImmutableSet<String> DEFAULT_PLATFORM_NAMES =
      ImmutableSet.of("host", "host_platform", "target_platform", "default_host", "default_target");

  public static boolean platformIsDefault(Label platform) {
    return DEFAULT_PLATFORM_NAMES.contains(platform.getName());
  }

  @Option(
      name = "host_platform",
      oldName = "experimental_host_platform",
      converter = EmptyToNullLabelConverter.class,
      defaultValue = "@bazel_tools//tools:host_platform",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help = "The label of a platform rule that describes the host system.")
  public Label hostPlatform;

  @Option(
      name = "extra_execution_platforms",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "The platforms that are available as execution platforms to run actions. "
              + "Platforms can be specified by exact target, or as a target pattern. "
              + "These platforms will be considered before those declared in the WORKSPACE file by "
              + "register_execution_platforms(). This option may only be set once; later "
              + "instances will override earlier flag settings.")
  public List<String> extraExecutionPlatforms;

  @Option(
      name = "platforms",
      oldName = "experimental_platforms",
      converter = LabelListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help =
          "The labels of the platform rules describing the target platforms for the current "
              + "command.")
  public List<Label> platforms;

  @Option(
      name = "extra_toolchains",
      defaultValue = "null",
      converter = CommaSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      allowMultiple = true,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help =
          "The toolchain rules to be considered during toolchain resolution. "
              + "Toolchains can be specified by exact target, or as a target pattern. "
              + "These toolchains will be considered before those declared in the WORKSPACE file "
              + "by register_toolchains().")
  public List<String> extraToolchains;

  @Option(
      name = "toolchain_resolution_debug",
      defaultValue = "-.*", // By default, exclude everything.
      converter = RegexFilter.RegexFilterConverter.class,
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Print debug information during toolchain resolution. The flag takes a regex, which is"
              + " checked against toolchain types and specific targets to see which to debug. "
              + "Multiple regexes may be  separated by commas, and then each regex is checked "
              + "separately. Note: The output of this flag is very complex and will likely only be "
              + "useful to experts in toolchain resolution.")
  public RegexFilter toolchainResolutionDebug;


  @Option(
      name = "incompatible_use_toolchain_resolution_for_java_rules",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "No-op. Kept here for backwards compatibility.")
  public boolean useToolchainResolutionForJavaRules;

  @Option(
      name = "platform_mappings",
      converter = OptionsUtils.EmptyToNullRelativePathFragmentConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help =
          "The location of a mapping file that describes which platform to use if none is set or "
              + "which flags to set when a platform already exists. Must be relative to the main "
              + "workspace root. Defaults to 'platform_mappings' (a file directly under the "
              + "workspace root).")
  public PathFragment platformMappings;

  @Option(
      name = "experimental_add_exec_constraints_to_targets",
      converter = RegexFilterToLabelListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      allowMultiple = true,
      help =
          "List of comma-separated regular expressions, each optionally prefixed by - (negative"
              + " expression), assigned (=) to a list of comma-separated constraint value targets."
              + " If a target matches no negative expression and at least one positive expression"
              + " its toolchain resolution will be performed as if it had declared the constraint"
              + " values as execution constraints. Example: //demo,-test=@platforms//cpus:x86_64"
              + " will add 'x86_64' to any target under //demo except for those whose name contains"
              + " 'test'.")
  public List<Map.Entry<RegexFilter, List<Label>>> targetFilterToAdditionalExecConstraints;

  /**
   * Deduplicate the given list, keeping the last copy of any duplicates.
   *
   * <p>Example: [a, b, a, c, b] -> [a, c, b]
   */
  protected static ImmutableList<String> dedupeKeepingLast(ImmutableList<String> values) {
    // Check common cases.
    if (values.size() <= 1) {
      return values;
    }

    // Reverse the list and then deduplicate.
    ImmutableList<String> reversedResult =
        values.reverse().stream().distinct().collect(toImmutableList());

    // If there were no duplicates, return the exact same instance we got.
    if (reversedResult.size() == values.size()) {
      return values;
    }

    // Reverse the result to get back to the original order.
    return reversedResult.reverse();
  }

  @Override
  public PlatformOptions getExec() {
    PlatformOptions exec = (PlatformOptions) getDefault();
    exec.platforms =
        this.hostPlatform == null ? ImmutableList.of() : ImmutableList.of(this.hostPlatform);
    exec.hostPlatform = this.hostPlatform;
    exec.platformMappings = this.platformMappings;
    exec.extraExecutionPlatforms = this.extraExecutionPlatforms;
    exec.extraToolchains = this.extraToolchains;
    exec.toolchainResolutionDebug = this.toolchainResolutionDebug;
    exec.useToolchainResolutionForJavaRules = this.useToolchainResolutionForJavaRules;
    return exec;
  }

  @Override
  public PlatformOptions getNormalized() {
    PlatformOptions result = (PlatformOptions) clone();
    result.extraToolchains =
        dedupeKeepingLast(
            result.extraToolchains == null
                ? ImmutableList.of()
                : ImmutableList.copyOf(result.extraToolchains));
    // Only the first entry of platforms is used (it should have been Label and not List<Label>)
    // So drop all but the first entry.
    if (result.platforms.size() > 1) {
      result.platforms = ImmutableList.of(result.platforms.get(0));
    }
    return result;
  }

  /** Returns the intended target platform value based on options defined in this fragment. */
  public Label computeTargetPlatform() {
    if (!platforms.isEmpty()) {
      return Iterables.getFirst(platforms, null);
    } else {
      // Default to the host platform, whatever it is.
      return hostPlatform;
    }
  }

  /** Converter of filter to label list valued flags. */
  public static final class RegexFilterToLabelListConverter
      extends Converters.AssignmentToListOfValuesConverter<RegexFilter, Label> {

    public RegexFilterToLabelListConverter() {
      super(
          new RegexFilter.RegexFilterConverter(),
          new CoreOptionConverters.LabelConverter(),
          AllowEmptyKeys.NO);
    }

    @Override
    public String getTypeDescription() {
      return "a '<RegexFilter>=<label1>[,<label2>,...]' assignment";
    }
  }
}
