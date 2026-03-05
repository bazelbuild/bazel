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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static java.util.Map.Entry.comparingByKey;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelListConverter;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelToStringEntryConverter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.BooleanConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionSetConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.TriState;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeSet;
import net.starlark.java.eval.StarlarkValue;

/**
 * Core options affecting a {@link BuildConfigurationValue} that don't belong in domain-specific
 * {@link FragmentOptions}. All options defined here should be universal in that they affect
 * configuration regardless of which languages a build uses. In other words, this should only
 * contain options that aren't suitable for Starlark configuration.
 *
 * <p>(Note: any client that creates a view will also need to declare BuildView.Options, which
 * affect the <i>mechanism</i> of view construction, even if they don't affect the value of the
 * BuildConfigurationValue instances.)
 *
 * <p>IMPORTANT: when adding new options, be sure to consider whether those values should be
 * propagated to the exec configuration or not.
 *
 * <p>ALSO IMPORTANT: all option types MUST define a toString method that gives identical results
 * for semantically identical option values. The simplest way to ensure that is to return the input
 * string.
 */
public class CoreOptions extends FragmentOptions implements Cloneable {

  @Option(
      name = "incompatible_filegroup_runfiles_for_data",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, runfiles of targets listed in the srcs attribute are available to targets"
              + " that consume the filegroup as a data dependency.")
  public boolean filegroupRunfilesForData;

  @Option(
      name = "scl_config",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Name of the scl config defined in PROJECT.scl. Note that this feature is still under"
              + " development b/324119879.")
  public String sclConfig;

  @Option(
      name = "incompatible_merge_genfiles_directory",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If true, the genfiles directory is folded into the bin directory.")
  public boolean mergeGenfilesDirectory;

  @Option(
      name = "experimental_exec_config",
      defaultValue = "@_builtins//:common/builtin_exec_platforms.bzl%bazel_exec_transition",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          If set to `//some:label:my.bzl%my_transition`, uses my_transition for `cfg = "exec"`
          semantics instead of Bazel's internal exec transition logic. Else uses Bazel's
          internal logic.
          """)
  public String starlarkExecConfig;

  @Option(
      name = "incompatible_disable_select_on",
      defaultValue = "",
      converter = CommaSeparatedOptionSetConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.NON_CONFIGURABLE},
      help = "List of flags for which the use in `select()` is disabled.")
  public ImmutableList<String> disabledSelectOptions;

  @Option(
      name = "experimental_propagate_custom_flag",
      defaultValue = "null",
      allowMultiple = true,
      converter = CoreOptionConverters.CustomFlagConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          Which custom flags (starlark flags or defines) to propagate to the exec transition, by
          key. e.g. if `--define=a=b` should be propagated, set
          `--experimental_propagate_custom_flag=a`
          """)
  public List<String> customFlagsToPropagate;

  @Option(
      name = "experimental_exclude_defines_from_exec_config",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          If true, don't propagate `--define`s to the exec transition at default; only propagate
          defines specified by `--experimental_propagate_custom_flag`.
          """)
  public boolean excludeDefinesFromExecConfig;

  @Option(
      name = "incompatible_exclude_starlark_flags_from_exec_config",
      oldName = "experimental_exclude_starlark_flags_from_exec_config",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, don't propagate starlark flags to the exec transition at default; only"
              + " propagate starlark flags specified in `--experimental_propagate_custom_flag`.")
  public boolean excludeStarlarkFlagsFromExecConfig;

  @Option(
      name = "experimental_platform_in_output_dir",
      defaultValue = "Auto",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          If true, a shortname for the target platform is used in the output directory name
          instead of the CPU. The exact scheme is experimental and subject to change:
          1. First, in the rare case the `--platforms` option does not have exactly one value, a
             hash of the platforms option is used.
          2. Next, if any shortname for the current platform was registered by
             `--experimental_override_name_platform_in_output_dir`, then that shortname is used.
          3. Then, if `--experimental_use_platforms_in_output_dir_legacy_heuristic` is set, use a
             shortname based off the current platform Label.
          4. Finally, a hash of the platform option is used as a last resort.
          """)
  public TriState platformInOutputDir;

  @Option(
      name = "experimental_use_platforms_in_output_dir_legacy_heuristic",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          Please only use this flag as part of a suggested migration or testing strategy. Note
          that the heuristic has known deficiencies and it is suggested to migrate to
          relying on just `--experimental_override_name_platform_in_output_dir`.
          """)
  public boolean usePlatformsInOutputDirLegacyHeuristic;

  @Option(
      name = "experimental_override_platform_cpu_name",
      oldName = "experimental_override_name_platform_in_output_dir",
      oldNameWarning = false,
      converter = LabelToStringEntryConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          Each entry should be of the form `label=value` where label refers to a platform and values
          is the desired shortname to override the platform's CPU name in `$(TARGET_CPU)`
          make variable and output path. Only used when
          `--experimental_platform_in_output_dir`, `--incompatible_target_cpu_from_platform` or
          `--incompatible_bep_cpu_from_platform` is true. Has highest naming priority.
          """)
  public List<Map.Entry<Label, String>> overridePlatformCpuName;

  @Option(
      name = "incompatible_limit_platforms_in_output_dir_to",
      converter = LabelListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "Added for gradual rollout of --experimental_platform_in_output_dir to non-exec"
              + " configurations. Takes a comma-separated list of platform labels. If set,"
              + " --experimental_platform_in_output_dir is only enabled for platforms in this list."
              + " Otherwise, --experimental_platform_in_output_dir applies to all platforms.")
  public List<Label> limitOutputDirToPlatforms;

  @Option(
      name = "incompatible_target_cpu_from_platform",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          """
          If specified, the value of the cpu constraint (`@platforms//cpu:cpu`) of
          the target platform is used to set the `$(TARGET_CPU)` make variable.
          """)
  public boolean incompatibleTargetCpuFromPlatform;

  @Option(
      name = "incompatible_bep_cpu_from_platform",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If specified, the value of the cpu constraint (@platforms//cpu:cpu) of"
              + " the target platform is used to set the Configuration.cpu and"
              + " Configuration.platform_name fields in the BEP.")
  public boolean incompatibleBepCpuFromPlatform;

  @Option(
      name = "define",
      converter = Converters.AssignmentConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          """
          Each `--define` option specifies an assignment for a build variable.
          In case of multiple values for a variable, the last one wins.
          """)
  public List<Map.Entry<String, String>> commandLineBuildVariables;

  // TODO: blaze-configurability-team - Remove this when --cpu is fully deprecated.
  @Option(
      name = "allowed_cpu_values",
      defaultValue = "",
      converter = CommaSeparatedOptionSetConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Allowed values for the `--cpu` flag.")
  public ImmutableList<String> allowedCpuValues;

  @Option(
      name = "cpu",
      defaultValue = "",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      // Don't set an actual deprecation notice: this is still heavily used and will be ignored.
      help =
          """
          Deprecated: this flag is not used internally by Blaze although there are legacy platform
          mappings to allow for backwards compatibility. Do not use this flag, instead use
          `--platforms` with an appropriate platform definition.
          """)
  public String cpu;

  @Option(
      name = "min_param_file_size",
      defaultValue = "32768",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.EXECUTION,
        OptionEffectTag.ACTION_COMMAND_LINES
      },
      help = "Minimum command line length before creating a parameter file.")
  public int minParamFileSize;

  @Option(
      name = "experimental_extended_sanity_checks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Enables internal validation checks to make sure that configured target "
              + "implementations only access things they should. Causes a performance hit.")
  public boolean extendedSanityChecks;

  // This option is only used during execution. However, it is a required input to the analysis
  // phase, as otherwise flipping this flag would not invalidate already-executed actions.
  @Option(
      name = "experimental_writable_outputs",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "If true, the file permissions of action outputs are set to `0755` instead of `0555`")
  public boolean experimentalWritableOutputs;

  @Option(
      name = "stamp",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Stamp binaries with the date, username, hostname, workspace information, etc.")
  public boolean stampBinaries;

  // This default value is always overwritten in the case of "bazel coverage" by
  // a value returned by InstrumentationFilterSupport.computeInstrumentationFilter.
  @Option(
      name = "instrumentation_filter",
      converter = RegexFilter.RegexFilterConverter.class,
      defaultValue = "-/javatests[/:],-/test/java[/:]",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          """
          When coverage is enabled, only rules with names included by the
          specified regex-based filter will be instrumented. Rules prefixed
          with '-' are excluded instead. Note that only non-test rules are
          instrumented unless `--instrument_test_targets` is enabled.
          """)
  public RegexFilter instrumentationFilter;

  @Option(
      name = "instrument_test_targets",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          """
          When coverage is enabled, specifies whether to consider instrumenting test rules.
          When set, test rules included by `--instrumentation_filter` are instrumented.
          Otherwise, test rules are always excluded from coverage instrumentation.
          """)
  public boolean instrumentTestTargets;

  @Option(
      name = "host_cpu",
      defaultValue = "",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "The host CPU.")
  public String hostCpu;

  @Option(
      name = "compilation_mode",
      abbrev = 'c',
      converter = CompilationMode.Converter.class,
      defaultValue = "fastbuild",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES},
      help = "Specify the mode the binary will be built in. Values: `fastbuild`, `dbg`, `opt`.")
  public CompilationMode compilationMode;

  @Option(
      name = "host_compilation_mode",
      converter = CompilationMode.Converter.class,
      defaultValue = "opt",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          """
          Specify the mode the tools used during the build will be built in. Values:
          `fastbuild`, `dbg`, `opt`.
          """)
  public CompilationMode hostCompilationMode;

  @Option(
      name = "incompatible_auto_exec_groups",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          """
          When enabled, an exec groups is automatically created for each toolchain used by a rule.
          For this to work rule needs to specify `toolchain` parameter on its actions. For
          more information, see [GH-17134].

          [GH-17134]: https://github.com/bazelbuild/bazel/issues/17134
          """)
  public boolean useAutoExecGroups;

  /* At the moment, EXPLICIT_IN_OUTPUT_PATH is not being set here because platform_suffix
   * is being used as a configuration distinguisher for the exec transition. */
  @Option(
      name = "platform_suffix",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help = "Specifies a suffix to be added to the configuration directory.")
  public String platformSuffix;

  // TODO(bazel-team): The set of available variables from the client environment for actions
  // is computed independently in CommandEnvironment to inject a more restricted client
  // environment to skyframe.
  @Option(
      name = "action_env",
      converter = Converters.EnvVarsConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          """
          Specifies the set of environment variables available to actions with target
          configuration. Variables can be either specified by `name`, in which case
          the value will be taken from the invocation environment, by the `name=value`
          pair which sets the value independent of the invocation environment, or by
          `=name`, which unsets the variable of that name. This option can be used
          multiple times; for options given for the same variable, the latest wins,
          options for different variables accumulate.

          Note that unless `--incompatible_repo_env_ignores_action_env` is true, all `name=value`
          pairs will be available to repository rules.
          """)
  public List<Converters.EnvVar> actionEnvironment;

  @Option(
      name = "host_action_env",
      converter = Converters.EnvVarsConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          """
          Specifies the set of environment variables available to actions with execution
          configurations. Variables can be either specified by `name`, in which case the
          value will be taken from the invocation environment, by the `name=value` pair
          which sets the value independent of the invocation environment, or by
          `=name`, which unsets the variable of that name. This option can
          be used multiple times; for options given for the same variable, the latest
          wins, options for different variables accumulate.
          """)
  public List<Converters.EnvVar> hostActionEnvironment;

  @Option(
      name = "collect_code_coverage",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          """
          If specified, Bazel will instrument code (using offline instrumentation where
          possible) and will collect coverage information during tests. Only targets that
          match `--instrumentation_filter` will be affected. Usually this option should
          not be specified directly - `bazel coverage` command should be used instead.
          """)
  public boolean collectCodeCoverage;

  @Option(
      name = "experimental_collect_code_coverage_for_generated_files",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If specified, Bazel will also generate collect coverage information for generated"
              + " files.")
  public boolean collectCodeCoverageForGeneratedFiles;

  @Option(
      name = "build_runfile_manifests",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, write runfiles manifests for all targets. If false, omit them. Local tests will"
              + " fail to run when false.")
  public boolean buildRunfileManifests;

  @Option(
      name = "build_runfile_links",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, build runfiles symlink forests for all targets.  "
              + "If false, write them only when required by a local action, test or run command.")
  public boolean buildRunfileLinks;

  @Option(
      name = "incompatible_always_include_files_in_data",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          """
          If true, native rules add `DefaultInfo.files` of data dependencies to their runfiles,
          which matches the recommended behavior for Starlark rules ([runfiles features to avoid]).

          [runfiles features to avoid]: https://bazel.build/extending/rules#runfiles_features_to_avoid
          """)
  public boolean alwaysIncludeFilesToBuildInData;

  @Option(
      name = "incompatible_compact_repo_mapping_manifest",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          """
          If enabled, the `{binary}.repo_mapping` file emits a module extension's repo mapping
          only once instead of once for each repo generated by the extension that
          contributes runfiles.
          """)
  public boolean compactRepoMapping;

  @Option(
      name = "run_under",
      defaultValue = "null",
      converter = RunUnderConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          """
          Prefix to insert before the executables for the `test` and `run` commands.
          If the value is `foo -bar`, and the execution command line is `test_binary -baz`,
          then the final command line is `foo -bar test_binary -baz`.
          This can also be a label to an executable target. Some examples are:
          - `valgrind`
          - `strace`
          - `strace -c`
          - `valgrind --quiet --num-callers=20`
          - `//package:target`
          - `//package:target --options`
          """)
  public RunUnder runUnder;

  @Option(
      name = "check_visibility",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.NON_CONFIGURABLE},
      help = "If disabled, visibility errors in target dependencies are demoted to warnings.")
  public boolean checkVisibility;

  @Option(
      name = "experimental_enforce_transitive_visibility",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If true, enable package()s to set the transitive_visibility attribute to restrict which"
              + " packages may depend on them.")
  public boolean enforceTransitiveVisibility;

  @Option(
      name = "verbose_visibility_errors",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.NON_CONFIGURABLE},
      help = "If enabled, visibility errors include additional diagnostic information.")
  public boolean verboseVisibilityErrors;

  @Option(
      name = "incompatible_check_testonly_for_output_files",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If enabled, check testonly for prerequisite targets that are output files by"
              + " looking up the testonly of the generating rule. This matches visibility"
              + " checking.")
  public boolean checkTestonlyForOutputFiles;

  // Moved from viewOptions to here because license information is very expensive to serialize.
  // Having it here allows us to skip computation of transitive license information completely
  // when the setting is disabled.
  @Option(
      name = "check_licenses",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help = "Deprecated in favor of `--config=check_licenses`.")
  public boolean checkLicenses;

  @Option(
      name = "enforce_constraints",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help =
          "Checks the environments each target is compatible with and reports errors if any "
              + "target has dependencies that don't support the same environments",
      oldName = "experimental_enforce_constraints")
  public boolean enforceConstraints;

  @Option(
      name = "experimental_action_listener",
      allowMultiple = true,
      defaultValue = "null",
      converter = LabelListConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          Deprecated in favor of aspects. Use `action_listener` to attach an `extra_action` to
          existing build actions.
          """)
  public List<Label> actionListeners;

  @Option(
      name = "is exec configuration",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.INTERNAL},
      help = "Shows whether these options are set for an execution configuration.")
  public boolean isExec;

  @Option(
      name = "allow_analysis_failures",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          If true, an analysis failure of a rule target results in the target's propagation
          of an instance of `AnalysisFailureInfo` containing the error description, instead
          of resulting in a build failure.
          """)
  public boolean allowAnalysisFailures;

  @Option(
      name = "evaluating for analysis test",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.INTERNAL},
      help =
          """
          If true, targets in the current configuration are being analyzed only for purposes
          of an analysis test. This, for example, imposes the restriction described by
          `--analysis_testing_deps_limit`.
          """)
  public boolean evaluatingForAnalysisTest;

  @Option(
      name = "analysis_testing_deps_limit",
      defaultValue = "2000",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          """
          Sets the maximum number of transitive dependencies through a rule attribute with
          a `for_analysis_testing` configuration transition.
          Exceeding this limit will result in a rule error.
          """)
  public int analysisTestingDepsLimit;

  @Option(
      name = "features",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          """
          The given features will be enabled or disabled by default for targets built in the target configuration.
          Specifying `-{feature}` will disable the feature. Negative features always override positive ones.
          See also `--host_features`.
          """)
  public List<String> defaultFeatures;

  @Option(
      name = "host_features",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          """
          The given features will be enabled or disabled by default for targets built in the exec configuration.
          Specifying `-{feature}` will disable the feature. Negative features always override positive ones.
          """)
  public List<String> hostFeatures;

  @Option(
      name = "target_environment",
      converter = LabelListConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          """
          Declares this build's target environment. Must be a label reference to an
          [`environment` rule]. If specified, all top-level targets must be compatible with this
          environment.

          See also `--platforms`.

          [`environment` rule]: https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/analysis/constraints/EnvironmentRule.java
          """)
  public List<Label> targetEnvironments;

  @Option(
      name = "allow_unresolved_symlinks",
      oldName = "experimental_allow_unresolved_symlinks",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          """
          If enabled, Bazel allows the use of `ctx.action.declare_symlink()` and the use of
          `ctx.actions.symlink()` without a target file, thus allowing the creation of
          unresolved symlinks.

          Unresolved symlinks inside tree artifacts are not currently supported.
          """)
  public boolean allowUnresolvedSymlinks;

  @Option(
      name = "experimental_allow_map_directory",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.EXECUTION,
      },
      metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.NON_CONFIGURABLE},
      help =
          "If enabled, Bazel allows the use of ctx.actions.map_directory(). This allows for the"
              + " creation of actions based on the files in a directory through a user defined"
              + " Starlark function and a <code>template_ctx</code> that supports basic action"
              + " generation APIs.")
  public boolean allowMapDirectory;

  /** Values for --experimental_output_paths. */
  public enum OutputPathsMode implements StarlarkValue {
    /** Use the production output path model. */
    OFF,
    /**
     * Strip the config prefix (i.e. {@code /x86-fastbuild/} from output paths for actions that are
     * registered to support this feature.
     *
     * <p>See {@link com.google.devtools.build.lib.actions.PathStripper} for details.
     */
    STRIP,
  }

  /** Converter for --experimental_output_paths. */
  public static class OutputPathsConverter extends EnumConverter<OutputPathsMode> {
    public OutputPathsConverter() {
      super(OutputPathsMode.class, "output path mode");
    }
  }

  @Option(
      name = "experimental_output_paths",
      converter = OutputPathsConverter.class,
      defaultValue = "off",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.EXECUTION
      },
      help =
          """
          Which model to use for where in the output tree rules write their outputs, particularly
          for multi-platform / multi-configuration builds. This is highly experimental. See
          [GH-6526] for details. Starlark actions can opt into path mapping by adding the key
          `supports-path-mapping` to the `execution_requirements` dict.

          [GH-6526]: https://github.com/bazelbuild/bazel/issues/6526
          """)
  public OutputPathsMode outputPathsMode;

  @Option(
      name = "enable_runfiles",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Enable runfiles symlink tree; By default, it's off on Windows, on on other platforms.")
  public TriState enableRunfiles;

  @Option(
      name = "modify_execution_info",
      allowMultiple = true,
      converter = ExecutionInfoModifier.Converter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {
        OptionEffectTag.EXECUTION,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          """
          Add or remove keys from an action's execution info based on action mnemonic.
          Applies only to actions which support execution info. Many common actions
          support execution info, e.g. Genrule, CppCompile, Javac, StarlarkAction,
          TestRunner. When specifying multiple values, order matters because
          many regexes may apply to the same mnemonic.

          Syntax: `regex=[+-]key,regex=[+-]key,...`.

          Examples:
          - `.*=+x,.*=-y,.*=+z` adds `x` and `z` to, and removes `y` from, the execution info for all actions.
          - `Genrule=+requires-x` adds `requires-x` to the execution info for all Genrule actions.
          - `(?!Genrule).*=-requires-x` removes `requires-x` from the execution info for all non-Genrule actions.
          """)
  public List<ExecutionInfoModifier> executionInfoModifier;

  @Option(
      name = "incompatible_modify_execution_info_additive",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {
        OptionEffectTag.EXECUTION,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          """
          When enabled, passing multiple `--modify_execution_info` flags is additive.
          When disabled, only the last flag is taken into account.
          """)
  public boolean additiveModifyExecutionInfo;

  @Option(
      name = "incompatible_bazel_test_exec_run_under",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
      },
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          """
          If enabled, `bazel test --run_under=//:runner` builds `//:runner` in the exec
          configuration. If disabled, it builds `//:runner` in the target configuration.
          Bazel executes tests on exec machines, so the former is more correct. This
          doesn't affect `bazel run`, which always builds `--run_under=//foo` in the
          target configuration.
          """)
  public boolean bazelTestExecRunUnder;

  @Option(
      name = "include_config_fragments_provider",
      defaultValue = "off",
      converter = IncludeConfigFragmentsEnumConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = OptionMetadataTag.HIDDEN,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          """
          INTERNAL BLAZE DEVELOPER FEATURE: If `direct`, all configured targets expose
          RequiredConfigFragmentsProvider with the configuration fragments they directly
          require.
          If `transitive`, they do the same but also include the fragments their
          transitive dependencies require. If `off`, the provider is omitted.

          If not `off`, this also populates `config_setting`'s
          `ConfigMatchingProvider.requiredFragmentOptions` with the fragment options the
          `config_setting` requires.

          Be careful using this feature: it adds memory to every configured target in the
          build.
          """)
  public IncludeConfigFragmentsEnum includeRequiredConfigFragmentsProvider;

  @Option(
      name = "experimental_remotable_source_manifests",
      defaultValue = "false",
      converter = BooleanConverter.class,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
      help = "Whether to make source manifest actions remotable")
  public boolean remotableSourceManifestActions;

  @Option(
      name = "flag_alias",
      converter = CoreOptionConverters.FlagAliasConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      metadataTags = {OptionMetadataTag.NON_CONFIGURABLE},
      help =
          """
          Sets a shorthand name for a Starlark flag. It takes a single key-value pair in the form
          `{key}={value}` as an argument.
          """)
  public List<Map.Entry<String, Label>> commandLineFlagAliases;

  @Option(
      name = "archived_tree_artifact_mnemonics_filter",
      defaultValue = "-.*", // disabled by default
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
      converter = RegexFilter.RegexFilterConverter.class,
      help =
          "Regex filter for mnemonics of actions for which we should create archived tree"
              + " artifacts. This option is a no-op for actions which do not generate tree"
              + " artifacts.")
  public RegexFilter archivedArtifactsMnemonicsFilter;

  @Option(
      name = "experimental_debug_selects_always_succeed",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          """
          When set, `select` functions with no matching clause will return an empty value, instead
          of failing. This is to help use `cquery` diagnose failures in `select`.
          """)
  public boolean debugSelectsAlwaysSucceed;

  @Option(
      name = "experimental_throttle_action_cache_check",
      defaultValue = "true",
      converter = BooleanConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "Whether to throttle the check whether an action is cached.")
  public boolean throttleActionCacheCheck;

  // This cannot be in TestOptions since the default test toolchain needs to be enabled
  // conditionally based on its value and test trimming would drop it when evaluating the toolchain
  // target.
  @Option(
      name = "use_target_platform_for_tests",
      deprecationWarning =
          "Tests select an execution platform matching all constraints of the target platform by"
              + " default. Instead of using this flag, make sure that all test target platform are"
              + " registered as execution platforms.",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "If true, use the target platform for running tests rather than the test exec group.")
  public boolean useTargetPlatformForTests;

  @Option(
      name = "exec_aspects",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      allowMultiple = true,
      help =
          "Comma-separated list of aspects to be applied to exec-configured targets, regardless of"
              + " whether or not they are top-level targets. This is an experimental feature and"
              + " is subject to change.")
  public List<String> execAspects;

  public Optional<String> getPlatformCpuNameOverride(Label platform) {
    // As highest priority, use the last entry that matches in name override option.
    return Streams.findLast(
        overridePlatformCpuName.stream()
            .filter(e -> e.getKey().equals(platform))
            .map(Map.Entry::getValue));
  }

  public boolean usePlatformInOutputDir(Label platform) {
    if (isExec) {
      return platformInOutputDir == TriState.YES || platformInOutputDir == TriState.AUTO;
    }

    if (platformInOutputDir == TriState.YES) {
      return limitOutputDirToPlatforms.isEmpty() || limitOutputDirToPlatforms.contains(platform);
    }

    return false;
  }

  private static final LoadingCache<List<Map.Entry<String, Label>>, ImmutableMap<String, Label>>
      ALIAS_MAP_CACHE = Caffeine.newBuilder().weakKeys().build(ImmutableMap::copyOf);

  public ImmutableMap<String, Label> getCommandLineFlagAliases() {
    return ALIAS_MAP_CACHE.get(commandLineFlagAliases);
  }

  /** Ways configured targets may provide the {@link Fragment}s they require. */
  public enum IncludeConfigFragmentsEnum implements StarlarkValue {
    /**
     * Don't offer the provider at all. This is best for most builds, which don't use this
     * information and don't need the extra memory hit over every configured target.
     */
    OFF,
    /** Provide the fragments required <em>directly</em> by this rule. */
    DIRECT,
    /** Provide the fragments required by this rule and its transitive dependencies. */
    TRANSITIVE
  }

  /** Enum converter for --include_config_fragments_provider. */
  public static class IncludeConfigFragmentsEnumConverter
      extends EnumConverter<IncludeConfigFragmentsEnum> {
    public IncludeConfigFragmentsEnumConverter() {
      super(IncludeConfigFragmentsEnum.class, "include config fragments provider option");
    }
  }

  /// Normalizes --define flags, preserving the last one to appear in the event of conflicts.
  public ImmutableMap<String, String> getNormalizedCommandLineBuildVariables() {
    return sortEntries(normalizeEntries(commandLineBuildVariables)).stream()
        .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
  }

  // Sort the map entries by key.
  private static <V> List<Map.Entry<String, V>> sortEntries(List<Map.Entry<String, V>> entries) {
    ImmutableList<Map.Entry<String, V>> sortedEntries =
        entries.stream().sorted(comparingByKey()).collect(toImmutableList());
    // If we made no changes, return the same instance we got to reduce churn.
    if (sortedEntries.equals(entries)) {
      return entries;
    }
    return sortedEntries;
  }

  /// Normalizes --features flags by sorting the values and having disables win over enables.
  private static List<String> getNormalizedFeatures(List<String> features) {
    // Parse out the features into a Map<String, boolean>, where the boolean represents whether
    // the feature is enabled or disabled.
    Map<String, Boolean> featureToState = new HashMap<>();
    for (String feature : features) {
      if (feature.startsWith("-")) {
        // disable always wins.
        featureToState.put(feature.substring(1), false);
      } else if (!featureToState.containsKey(feature)) {
        // enable feature only if it does not already have a state.
        // If existing state is enabled, no need to do extra work.
        // If existing state is disabled, it wins.
        featureToState.put(feature, true);
      }
    }
    // Partition into enabled/disabled features.
    TreeSet<String> enabled = new TreeSet<>();
    TreeSet<String> disabled = new TreeSet<>();
    for (Map.Entry<String, Boolean> entry : featureToState.entrySet()) {
      if (entry.getValue()) {
        enabled.add(entry.getKey());
      } else {
        disabled.add(entry.getKey());
      }
    }
    // Rebuild the set of features.
    // Since we used TreeSet the features come out in a deterministic order.
    List<String> result = new ArrayList<>(enabled);
    disabled.stream().map(x -> "-" + x).forEach(result::add);
    // If we made no changes, return the same instance we got to reduce churn.
    return result.equals(features) ? features : result;
  }

  @Override
  public CoreOptions getNormalized() {
    CoreOptions result = (CoreOptions) clone();
    result.allowedCpuValues = dedupAndSort(allowedCpuValues);
    result.commandLineBuildVariables = sortEntries(normalizeEntries(commandLineBuildVariables));

    // Normalize features.
    result.defaultFeatures = getNormalizedFeatures(defaultFeatures);

    result.actionEnvironment = normalizeEnvVars(actionEnvironment);
    result.hostActionEnvironment = normalizeEnvVars(hostActionEnvironment);
    result.commandLineFlagAliases = sortEntries(normalizeEntries(commandLineFlagAliases));

    return result;
  }

  @Override
  public CoreOptions clone() {
    return (CoreOptions) super.clone();
  }
}
