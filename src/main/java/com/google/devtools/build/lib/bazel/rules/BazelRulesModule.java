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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.rules.cpp.CcSkyframeFdoSupportFunction;
import com.google.devtools.build.lib.rules.cpp.CcSkyframeFdoSupportValue;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/** Module implementing the rule set of Bazel. */
public class BazelRulesModule extends BlazeModule {
  /** This is where deprecated options go to die. */
  public static class GraveyardOptions extends OptionsBase {
    @Option(
        name = "incompatible_disallow_legacy_java_provider",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean incompatibleDisallowLegacyJavaProvider;

    @Option(
        name = "incompatible_disable_legacy_proto_provider",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
        },
        help = "Deprecated no-op.")
    public boolean disableLegacyProtoProvider;

    @Option(
        name = "incompatible_disable_proto_source_root",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
        },
        help = "Deprecated no-op.")
    public boolean disableProtoSourceRoot;

    @Option(
        name = "incompatible_do_not_emit_buggy_external_repo_import",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean doNotUseBuggyImportPath;

    @Option(
        name = "incompatible_disable_crosstool_file",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.INCOMPATIBLE_CHANGE
        },
        help = "Deprecated no-op.")
    public boolean disableCrosstool;

    @Option(
        name = "incompatible_disable_legacy_crosstool_fields",
        oldName = "experimental_disable_legacy_crosstool_fields",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean disableLegacyCrosstoolFields;

    @Option(
        name = "incompatible_require_feature_configuration_for_pic",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean requireFeatureConfigurationForPic;

    @Option(
        name = "incompatible_disable_depset_in_cc_user_flags",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean disableDepsetInUserFlags;

    @Option(
        name = "incompatible_dont_emit_static_libgcc",
        oldName = "experimental_dont_emit_static_libgcc",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean disableEmittingStaticLibgcc;

    @Option(
        name = "incompatible_linkopts_in_user_link_flags",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean enableLinkoptsInUserLinkFlags;

    @Option(
        name = "incompatible_disable_runtimes_filegroups",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean disableRuntimesFilegroups;

    @Option(
        name = "incompatible_disable_tools_defaults_package",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.INCOMPATIBLE_CHANGE
        },
        help = "Deprecated no-op.")
    public boolean incompatibleDisableInMemoryToolsDefaultsPackage;

    @Option(
        name = "experimental_enable_cc_toolchain_config_info",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean enableCcToolchainConfigInfoFromStarlark;

    @Option(
        name = "output_symbol_counts",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
        metadataTags = {OptionMetadataTag.HIDDEN, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean symbolCounts;

    @Option(
        name = "incompatible_disable_sysroot_from_configuration",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean disableSysrootFromConfiguration;

    @Option(
        name = "incompatible_provide_cc_toolchain_info_from_cc_toolchain_suite",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean provideCcToolchainInfoFromCcToolchainSuite;

    @Option(
        name = "incompatible_disable_cc_toolchain_label_from_crosstool_proto",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EAGERNESS_TO_EXIT},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "Deprecated no-op.")
    public boolean disableCcToolchainFromCrosstool;

    @Option(
        name = "incompatible_disable_cc_configuration_make_variables",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.DEPRECATED,
        },
        help = "Deprecated no-op.")
    public boolean disableMakeVariables;

    @Option(
        name = "make_variables_source",
        defaultValue = "configuration",
        metadataTags = {OptionMetadataTag.HIDDEN, OptionMetadataTag.DEPRECATED},
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN})
    public String makeVariableSource;

    @Option(
        name = "incompatible_disable_legacy_flags_cc_toolchain_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.DEPRECATED
        },
        help =
            "Flag for disabling the legacy cc_toolchain Starlark API for accessing legacy "
                + "CROSSTOOL fields.")
    public boolean disableLegacyFlagsCcToolchainApi;

    @Option(
        name = "incompatible_enable_legacy_cpp_toolchain_skylark_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.DEPRECATED
        },
        help = "Obsolete, no effect.")
    public boolean enableLegacyToolchainStarlarkApi;

    @Option(
        name = "incompatible_disable_legacy_cpp_toolchain_skylark_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.DEPRECATED
        },
        help = "Obsolete, no effect.")
    public boolean disableLegacyToolchainStarlarkApi;

    @Option(
        name = "incompatible_cc_coverage",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.UNKNOWN,
        },
        oldName = "experimental_cc_coverage",
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES,
          OptionMetadataTag.DEPRECATED
        },
        help = "Obsolete, no effect.")
    public boolean useGcovCoverage;

    @Deprecated
    @Option(
        name = "direct_run",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean directRun;

    @Deprecated
    @Option(
        name = "glibc",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public String glibc;

    @Deprecated
    @Option(
        name = "experimental_shortened_obj_file_path",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.EXECUTION},
        defaultValue = "true",
        help = "This option is deprecated and has no effect.")
    public boolean shortenObjFilePath;

    @Option(
        name = "force_ignore_dash_static",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
        help = "noop")
    public boolean forceIgnoreDashStatic;

    @Option(
        name = "incompatible_disable_late_bound_option_defaults",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "This option is deprecated and has no effect.")
    public boolean incompatibleDisableLateBoundOptionDefaults;

    @Option(
        name = "incompatible_use_native_patch",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.DEPRECATED,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "This option is deprecated and has no effect.")
    public boolean useNativePatch;

    @Deprecated
    @Option(
        name = "ui",
        oldName = "experimental_ui",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "No-op.")
    public boolean experimentalUi;

    @Option(
        name = "experimental_profile_action_counts",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "No-op.")
    public boolean enableActionCountProfile;

    @Option(
        name = "incompatible_remove_binary_profile",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "No-op.")
    public boolean removeBinaryProfile;

    @Option(
        name = "experimental_post_profile_started_event",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "No-op.")
    public boolean postProfileStartedEvent;

    @Option(
        name = "incompatible_enable_profile_by_default",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help = "No-op.")
    public boolean enableProfileByDefault;
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    builder.setToolsRepository(BazelRuleClassProvider.TOOLS_REPOSITORY);
    BazelRuleClassProvider.setup(builder);

    try {
      // Load auto-configuration files, it is made outside of the rule class provider so that it
      // will not be loaded for our Java tests.
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelRulesModule.class, "xcode_configure.WORKSPACE"));
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelShRuleClasses.class, "sh_configure.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    validateRemoteOutputsMode(env);
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.addSkyFunction(
        CcSkyframeFdoSupportValue.SKYFUNCTION, new CcSkyframeFdoSupportFunction(directories));
  }

  @Override
  public BuildOptions getDefaultBuildOptions(BlazeRuntime blazeRuntime) {
    return BuildOptions.getDefaultBuildOptionsForFragments(
        blazeRuntime.getRuleClassProvider().getConfigurationOptions());
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(GraveyardOptions.class)
        : ImmutableList.of();
  }

  private static void validateRemoteOutputsMode(CommandEnvironment env) throws AbruptExitException {
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    if (remoteOptions == null) {
      return;
    }
    if (!remoteOptions.remoteOutputsMode.downloadAllOutputs()) {
      JavaOptions javaOptions = env.getOptions().getOptions(JavaOptions.class);
      if (javaOptions != null && !javaOptions.inmemoryJdepsFiles) {
        throw createRemoteExecutionExitException(
            "--experimental_remote_download_outputs=minimal requires"
                + " --experimental_inmemory_jdeps_files to be enabled",
            Code.REMOTE_DOWNLOAD_OUTPUTS_MINIMAL_WITHOUT_INMEMORY_JDEPS);
      }
      CppOptions cppOptions = env.getOptions().getOptions(CppOptions.class);
      if (cppOptions != null && !cppOptions.inmemoryDotdFiles) {
        throw createRemoteExecutionExitException(
            "--experimental_remote_download_outputs=minimal requires"
                + " --experimental_inmemory_dotd_files to be enabled",
            Code.REMOTE_DOWNLOAD_OUTPUTS_MINIMAL_WITHOUT_INMEMORY_DOTD);
      }
    }
  }

  private static AbruptExitException createRemoteExecutionExitException(
      String message, Code remoteExecutionCode) {
    return new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setRemoteExecution(RemoteExecution.newBuilder().setCode(remoteExecutionCode))
                .build()));
  }
}
