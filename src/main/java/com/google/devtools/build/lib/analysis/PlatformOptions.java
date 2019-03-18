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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelListConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;

/** Command-line options for platform-related configuration. */
public class PlatformOptions extends FragmentOptions {

  // TODO(https://github.com/bazelbuild/bazel/issues/6849): After migration, set the defaults
  // directly.
  public static final Label LEGACY_DEFAULT_HOST_PLATFORM =
      Label.parseAbsoluteUnchecked("@bazel_tools//platforms:host_platform");
  public static final Label DEFAULT_HOST_PLATFORM =
      Label.parseAbsoluteUnchecked("@local_config_platform//:host");
  public static final Label LEGACY_DEFAULT_TARGET_PLATFORM =
      Label.parseAbsoluteUnchecked("@bazel_tools//platforms:target_platform");

  @Option(
      name = "host_platform",
      oldName = "experimental_host_platform",
      converter = BuildConfiguration.EmptyToNullLabelConverter.class,
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
      defaultValue = "",
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
      name = "extra_toolchains",
      defaultValue = "",
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
      defaultValue = "",
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
      name = "enabled_toolchain_types",
      defaultValue = "",
      converter = LabelListConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      deprecationWarning =
          "Use --incompatible_enable_cc_toolchain_resolution to enable toolchain for cc rules. "
              + "Other rules will define separate flags as needed.",
      help =
          "Enable toolchain resolution for the given toolchain type, if the rules used support "
              + "that. This does not directly change the core Blaze machinery, but is a signal to "
              + "participating rule implementations that toolchain resolution should be used.")
  public List<Label> enabledToolchainTypes;

  @Option(
      name = "incompatible_auto_configure_host_platform",
      defaultValue = "false",
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
      name = "experimental_use_toolchain_resolution_for_java_rules",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "If set to true, toolchain resolution will be used to resolve java_toolchain and"
              + " java_runtime.")
  public boolean useToolchainResolutionForJavaRules;

  @Override
  public PlatformOptions getHost() {
    PlatformOptions host = (PlatformOptions) getDefault();
    host.platforms =
        this.hostPlatform == null ? ImmutableList.of() : ImmutableList.of(this.hostPlatform);
    host.hostPlatform = this.hostPlatform;
    host.extraExecutionPlatforms = this.extraExecutionPlatforms;
    host.extraToolchains = this.extraToolchains;
    host.enabledToolchainTypes = this.enabledToolchainTypes;
    host.toolchainResolutionDebug = this.toolchainResolutionDebug;
    host.toolchainResolutionOverrides = this.toolchainResolutionOverrides;
    host.autoConfigureHostPlatform = this.autoConfigureHostPlatform;
    host.useToolchainResolutionForJavaRules = this.useToolchainResolutionForJavaRules;
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
      return LEGACY_DEFAULT_TARGET_PLATFORM;
    }
  }

  /** Returns the intended host platform value based on options defined in this fragment. */
  public Label computeHostPlatform() {
    // Handle default values for the host and target platform.
    // TODO(https://github.com/bazelbuild/bazel/issues/6849): After migration, set the defaults
    // directly.

    Label hostPlatform;
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
}
