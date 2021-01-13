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

import com.google.common.collect.ImmutableList;
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

  // TODO(https://github.com/bazelbuild/bazel/issues/6849): After migration, set the defaults
  // directly.
  public static final Label LEGACY_DEFAULT_HOST_PLATFORM =
      Label.parseAbsoluteUnchecked("@bazel_tools//platforms:host_platform");
  public static final Label DEFAULT_HOST_PLATFORM =
      Label.parseAbsoluteUnchecked("@local_config_platform//:host");

  /**
   * Main workspace-relative location to use when the user does not explicitly set {@code
   * --platform_mappings}.
   */
  public static final PathFragment DEFAULT_PLATFORM_MAPPINGS =
      PathFragment.create("platform_mappings");

  @Option(
      name = "host_platform",
      oldName = "experimental_host_platform",
      converter = EmptyToNullLabelConverter.class,
      defaultValue = "",
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
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      allowMultiple = true,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "The platforms that are available as execution platforms to run actions. "
              + "Platforms can be specified by exact target, or as a target pattern. "
              + "These platforms will be considered before those declared in the WORKSPACE file by "
              + "register_execution_platforms().")
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
      name = "target_platform_fallback",
      converter = EmptyToNullLabelConverter.class,
      defaultValue = "@bazel_tools//platforms:target_platform",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help =
          "The label of a platform rule that should be used if no target platform is set and no"
              + " platform mapping matches the current set of flags.")
  public Label targetPlatformFallback;

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
      name = "toolchain_resolution_override",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      deprecationWarning =
          "toolchain_resolution_override is now a no-op and will be removed in"
              + " an upcoming release",
      help =
          "Override toolchain resolution for a toolchain type with a specific toolchain. "
              + "Example: --toolchain_resolution_override=@io_bazel_rules_go//:toolchain="
              + "@io_bazel_rules_go//:linux-arm64-toolchain")
  public List<String> toolchainResolutionOverrides;

  @Option(
      name = "toolchain_resolution_debug",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Print debug information while finding toolchains for a rule. This might help developers "
              + "of Bazel or Starlark rules with debugging failures due to missing toolchains.")
  public boolean toolchainResolutionDebug;

  @Option(
      name = "incompatible_auto_configure_host_platform",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If true, the host platform will be inherited from @local_config_platform//:host, "
              + "instead of being based on the --cpu (and --host_cpu) flags.")
  public boolean autoConfigureHostPlatform;

  @Option(
      name = "incompatible_use_toolchain_resolution_for_java_rules",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, toolchain resolution will be used to resolve java_toolchain and"
              + " java_runtime.")
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

  @Option(
      name = "incompatible_override_toolchain_transition",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, all rules will use the toolchain transition for toolchain dependencies.")
  public boolean overrideToolchainTransition;

  @Override
  public PlatformOptions getHost() {
    PlatformOptions host = (PlatformOptions) getDefault();
    host.platforms =
        this.hostPlatform == null ? ImmutableList.of() : ImmutableList.of(this.hostPlatform);
    host.hostPlatform = this.hostPlatform;
    host.platformMappings = this.platformMappings;
    host.extraExecutionPlatforms = this.extraExecutionPlatforms;
    host.extraToolchains = this.extraToolchains;
    host.toolchainResolutionDebug = this.toolchainResolutionDebug;
    host.toolchainResolutionOverrides = this.toolchainResolutionOverrides;
    host.autoConfigureHostPlatform = this.autoConfigureHostPlatform;
    host.useToolchainResolutionForJavaRules = this.useToolchainResolutionForJavaRules;
    host.targetPlatformFallback = this.targetPlatformFallback;
    host.overrideToolchainTransition = this.overrideToolchainTransition;
    return host;
  }

  /** Returns the intended target platform value based on options defined in this fragment. */
  public Label computeTargetPlatform() {
    // Handle default values for the host and target platform.
    // TODO(https://github.com/bazelbuild/bazel/issues/6849): After migration, set the defaults
    // directly.

    if (!platforms.isEmpty()) {
      return Iterables.getFirst(platforms, null);
    } else if (autoConfigureHostPlatform) {
      // Default to the host platform, whatever it is.
      return computeHostPlatform();
    } else {
      // Use the legacy target platform
      return targetPlatformFallback;
    }
  }

  /** Returns the intended host platform value based on options defined in this fragment. */
  public Label computeHostPlatform() {
    // Handle default values for the host and target platform.
    // TODO(https://github.com/bazelbuild/bazel/issues/6849): After migration, set the defaults
    // directly.

    if (this.hostPlatform != null) {
      return this.hostPlatform;
    } else if (autoConfigureHostPlatform) {
      // Use the auto-configured host platform.
      return DEFAULT_HOST_PLATFORM;
    } else {
      // Use the legacy host platform.
      return LEGACY_DEFAULT_HOST_PLATFORM;
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
