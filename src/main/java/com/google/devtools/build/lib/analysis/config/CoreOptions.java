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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.EmptyToNullLabelConverter;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelListConverter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.BooleanConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import java.util.AbstractMap.SimpleEntry;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Core options affecting a {@link BuildConfiguration} that don't belong in domain-specific {@link
 * FragmentOptions}. All options defined here should be universal in that they affect configuration
 * regardless of which languages a build uses. In other words, this should only contain options that
 * aren't suitable for Starlark configuration.
 *
 * <p>(Note: any client that creates a view will also need to declare BuildView.Options, which
 * affect the <i>mechanism</i> of view construction, even if they don't affect the value of the
 * BuildConfiguration instances.)
 *
 * <p>IMPORTANT: when adding new options, be sure to consider whether those values should be
 * propagated to the host configuration or not.
 *
 * <p>ALSO IMPORTANT: all option types MUST define a toString method that gives identical results
 * for semantically identical option values. The simplest way to ensure that is to return the input
 * string.
 */
public class CoreOptions extends FragmentOptions implements Cloneable {
  public static final OptionDefinition CPU =
      OptionsParser.getOptionDefinitionByName(CoreOptions.class, "cpu");

  @Option(
      name = "incompatible_merge_genfiles_directory",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If true, the genfiles directory is folded into the bin directory.")
  public boolean mergeGenfilesDirectory;

  @Option(
      name = "incompatible_use_platforms_repo_for_constraints",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If true, constraint settings from @bazel_tools are removed.")
  public boolean usePlatformsRepoForConstraints;

  @Option(
      name = "define",
      converter = Converters.AssignmentConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Each --define option specifies an assignment for a build variable.")
  public List<Map.Entry<String, String>> commandLineBuildVariables;

  @Option(
      name = "collapse_duplicate_defines",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help =
          "When enabled, redundant --defines will be removed early in the build. This avoids"
              + " unnecessary loss of the analysis cache for certain types of equivalent"
              + " builds.")
  public boolean collapseDuplicateDefines;

  @Option(
      name = "cpu",
      defaultValue = "",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "The target CPU.")
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
      name = "defer_param_files",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.EXECUTION,
        OptionEffectTag.ACTION_COMMAND_LINES
      },
      help = "This option is deprecated and has no effect and will be removed in the future.")
  public boolean deferParamFiles;

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

  @Option(
      name = "strict_filesets",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
      help =
          "If this option is enabled, filesets crossing package boundaries are reported "
              + "as errors. It does not work when check_fileset_dependencies_recursively is "
              + "disabled.")
  public boolean strictFilesets;

  @Option(
      name = "experimental_strict_fileset_output",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "If this option is enabled, filesets will treat all output artifacts as regular files. "
              + "They will not traverse directories or be sensitive to symlinks.")
  public boolean strictFilesetOutput;

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
          "When coverage is enabled, only rules with names included by the "
              + "specified regex-based filter will be instrumented. Rules prefixed "
              + "with '-' are excluded instead. Note that only non-test rules are "
              + "instrumented unless --instrument_test_targets is enabled.")
  public RegexFilter instrumentationFilter;

  @Option(
      name = "instrument_test_targets",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "When coverage is enabled, specifies whether to consider instrumenting test rules. "
              + "When set, test rules included by --instrumentation_filter are instrumented. "
              + "Otherwise, test rules are always excluded from coverage instrumentation.")
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
      help = "Specify the mode the binary will be built in. Values: 'fastbuild', 'dbg', 'opt'.")
  public CompilationMode compilationMode;

  @Option(
      name = "host_compilation_mode",
      converter = CompilationMode.Converter.class,
      defaultValue = "opt",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          "Specify the mode the tools used during the build will be built in. Values: "
              + "'fastbuild', 'dbg', 'opt'.")
  public CompilationMode hostCompilationMode;

  /**
   * This option is used internally to set output directory name of the <i>host</i> configuration to
   * a constant, so that the output files for the host are completely independent of those for the
   * target, no matter what options are in force (k8/piii, opt/dbg, etc).
   */
  @Option(
      name = "output directory name",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      metadataTags = {OptionMetadataTag.INTERNAL})
  public String outputDirectoryName;

  /**
   * This option is used by starlark transitions to add a distinguishing element to the output
   * directory name, in order to avoid name clashing.
   */
  @Option(
      name = "transition directory name fragment",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      metadataTags = {OptionMetadataTag.INTERNAL})
  public String transitionDirectoryNameFragment;

  /** Regardless of input, converts to an empty list. For use with affectedByStarlarkTransition */
  public static class EmptyListConverter implements Converter<List<String>> {
    @Override
    public List<String> convert(String input) throws OptionsParsingException {
      return ImmutableList.of();
    }

    @Override
    public String getTypeDescription() {
      return "Regardless of input, converts to an empty list. For use with"
          + " affectedByStarlarkTransition";
    }
  }

  /**
   * This internal option is a *set* of names (e.g. "cpu") of *native* options that have been
   * changed by starlark transitions at any point in the build at the time of accessing. This is
   * used to regenerate {@code transitionDirectoryNameFragment} after each starlark transition.
   */
  @Option(
      name = "affected by starlark transition",
      defaultValue = "",
      converter = EmptyListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      metadataTags = {OptionMetadataTag.INTERNAL})
  public List<String> affectedByStarlarkTransition;

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

  // TODO(bazel-team): The test environment is actually computed in BlazeRuntime and this option
  // is not read anywhere else. Thus, it should be in a different options class, preferably one
  // specific to the "test" command or maybe in its own configuration fragment.
  @Option(
      name = "test_env",
      converter = Converters.OptionalAssignmentConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.TEST_RUNNER},
      help =
          "Specifies additional environment variables to be injected into the test runner "
              + "environment. Variables can be either specified by name, in which case its value "
              + "will be read from the Bazel client environment, or by the name=value pair. "
              + "This option can be used multiple times to specify several variables. "
              + "Used only by the 'bazel test' command.")
  public List<Map.Entry<String, String>> testEnvironment;

  // TODO(bazel-team): The set of available variables from the client environment for actions
  // is computed independently in CommandEnvironment to inject a more restricted client
  // environment to skyframe.
  @Option(
      name = "action_env",
      converter = Converters.OptionalAssignmentConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          "Specifies the set of environment variables available to actions. "
              + "Variables can be either specified by name, in which case the value will be "
              + "taken from the invocation environment, or by the name=value pair which sets "
              + "the value independent of the invocation environment. This option can be used "
              + "multiple times; for options given for the same variable, the latest wins, options "
              + "for different variables accumulate.")
  public List<Map.Entry<String, String>> actionEnvironment;

  @Option(
      name = "repo_env",
      converter = Converters.OptionalAssignmentConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          "Specifies additional environment variables to be available only for repository rules."
              + " Note that repository rules see the full environment anyway, but in this way"
              + " configuration information can be passed to repositories through options without"
              + " invalidating the action graph.")
  public List<Map.Entry<String, String>> repositoryEnvironment;

  @Option(
      name = "collect_code_coverage",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If specified, Bazel will instrument code (using offline instrumentation where "
              + "possible) and will collect coverage information during tests. Only targets that "
              + " match --instrumentation_filter will be affected. Usually this option should "
              + " not be specified directly - 'bazel coverage' command should be used instead.")
  public boolean collectCodeCoverage;

  @Option(
      name = "experimental_forward_instrumented_files_info_by_default",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If specified, rules that don't configure InstrumentedFilesInfo will still forward the "
              + "contents of InstrumentedFilesInfo from transitive dependencies.")
  public boolean experimentalForwardInstrumentedFilesInfoByDefault;

  @Option(
      name = "build_runfile_manifests",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "If true, write runfiles manifests for all targets.  " + "If false, omit them.")
  public boolean buildRunfilesManifests;

  @Option(
      name = "build_runfile_links",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, build runfiles symlink forests for all targets.  "
              + "If false, write only manifests when possible.")
  public boolean buildRunfiles;

  @Option(
      name = "legacy_external_runfiles",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, build runfiles symlink forests for external repositories under "
              + ".runfiles/wsname/external/repo (in addition to .runfiles/repo).")
  public boolean legacyExternalRunfiles;

  @Option(
      name = "check_fileset_dependencies_recursively",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      deprecationWarning =
          "This flag is a no-op and fileset dependencies are always checked "
              + "to ensure correctness of builds.",
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS})
  public boolean checkFilesetDependenciesRecursively;

  @Option(
      name = "experimental_skyframe_native_filesets",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      deprecationWarning = "This flag is a no-op and skyframe-native-filesets is always true.")
  public boolean skyframeNativeFileset;

  @Option(
      name = "run_under",
      defaultValue = "null",
      converter = RunUnderConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          "Prefix to insert before the executables for the 'test' and 'run' commands. "
              + "If the value is 'foo -bar', and the execution command line is 'test_binary -baz', "
              + "then the final command line is 'foo -bar test_binary -baz'."
              + "This can also be a label to an executable target. Some examples are: "
              + "'valgrind', 'strace', 'strace -c', "
              + "'valgrind --quiet --num-callers=20', '//package:target', "
              + " '//package:target --options'.")
  public RunUnder runUnder;

  @Option(
      name = "distinct_host_configuration",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          "Build all the tools used during the build for a distinct configuration from that used "
              + "for the target program. When this is disabled, the same configuration is used for "
              + "host and target programs. This may cause undesirable rebuilds of tools such as "
              + "the protocol compiler (and then everything downstream) whenever a minor change "
              + "is made to the target configuration, such as setting the linker options. When "
              + "this is enabled (the default), a distinct configuration will be used to build the "
              + "tools, preventing undesired rebuilds. However, certain libraries will then need "
              + "to be compiled twice, once for each configuration, which may cause some builds "
              + "to be slower. As a rule of thumb, this option is likely to benefit users that "
              + "make frequent changes in configuration (e.g. opt/dbg).  "
              + "Please read the user manual for the full explanation.")
  public boolean useDistinctHostConfiguration;

  @Option(
      name = "check_visibility",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help = "If disabled, visibility errors are demoted to warnings.")
  public boolean checkVisibility;

  // Moved from viewOptions to here because license information is very expensive to serialize.
  // Having it here allows us to skip computation of transitive license information completely
  // when the setting is disabled.
  @Option(
      name = "check_licenses",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help =
          "Check that licensing constraints imposed by dependent packages "
              + "do not conflict with distribution modes of the targets being built. "
              + "By default, licenses are not checked.")
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
      help = "Use action_listener to attach an extra_action to existing build actions.")
  public List<Label> actionListeners;

  @Option(
      name = "is host configuration",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.INTERNAL},
      help = "Shows whether these options are set for host configuration.")
  public boolean isHost;

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
          "If true, an analysis failure of a rule target results in the target's propagation "
              + "of an instance of AnalysisFailureInfo containing the error description, instead "
              + "of resulting in a build failure.")
  public boolean allowAnalysisFailures;

  @Option(
      name = "evaluating for analysis test",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.INTERNAL},
      help =
          "If true, targets in the current configuration are being analyzed only for purposes "
              + "of an analysis test. This, for example, imposes the restriction described by "
              + "--analysis_testing_deps_limit.")
  public boolean evaluatingForAnalysisTest;

  @Option(
      name = "analysis_testing_deps_limit",
      defaultValue = "600",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Sets the maximum number of transitive dependencies through a rule attribute with "
              + "a for_analysis_testing configuration transition. "
              + "Exceeding this limit will result in a rule error.")
  public int analysisTestingDepsLimit;

  @Option(
      name = "features",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "The given features will be enabled or disabled by default for all packages. "
              + "Specifying -<feature> will disable the feature globally. "
              + "Negative features always override positive ones. "
              + "This flag is used to enable rolling out default feature changes without a "
              + "Bazel release.")
  public List<String> defaultFeatures;

  @Option(
      name = "target_environment",
      converter = LabelListConverter.class,
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "Declares this build's target environment. Must be a label reference to an "
              + "\"environment\" rule. If specified, all top-level targets must be "
              + "compatible with this environment.")
  public List<Label> targetEnvironments;

  @Option(
      name = "auto_cpu_environment_group",
      converter = EmptyToNullLabelConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Declare the environment_group to use for automatically mapping cpu values to "
              + "target_environment values.")
  public Label autoCpuEnvironmentGroup;

  /** Values for --experimental_dynamic_configs. */
  public enum ConfigsMode {
    /**
     * Deprecated mode: Each configured target is evaluated with only the configuration fragments it
     * needs by loading the target graph and examining the transitive requirements for each target
     * before analysis begins.
     *
     * <p>To become a no-op soon: b/129289764
     */
    ON,
    /** Default mode: Each configured target is evaluated with all fragments known to Blaze. */
    NOTRIM,
    /**
     * Experimental mode: Each configured target is evaluated with only the configuration fragments
     * it needs by visiting them with a full configuration to begin with and collapsing the
     * configuration down to the fragments which were actually used.
     */
    RETROACTIVE;
  }

  /** Converter for --experimental_dynamic_configs. */
  public static class ConfigsModeConverter extends EnumConverter<ConfigsMode> {
    public ConfigsModeConverter() {
      super(ConfigsMode.class, "configurations mode");
    }
  }

  @Option(
      name = "experimental_dynamic_configs",
      defaultValue = "notrim",
      converter = ConfigsModeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Instantiates build configurations with the specified properties")
  public ConfigsMode configsMode;

  /** Values for --experimental_output_paths. */
  public enum OutputPathsMode {
    /** Use the production output path model. */
    OFF,
    /**
     * Use <a href="https://github.com/bazelbuild/bazel/issues/6526#issuecomment-488103473">
     * content-based paths</a>.
     *
     * <p>Rule implementations also have to individually opt into this. So this setting doesn't mean
     * all outputs follow this. Non-opted-in outputs continue to use the production model.
     *
     * <p>Follow the above link for latest details on exact scope.
     */
    CONTENT,
  }

  /** Converter for --experimental_output_paths. */
  public static class OutputPathsConverter extends EnumConverter<OutputPathsMode> {
    public OutputPathsConverter() {
      super(OutputPathsMode.class, "output path mode");
    }
  }

  @Option(
      name = "experimental_allow_unresolved_symlinks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          "If enabled, Bazel allows the use of ctx.action.{declare_symlink,symlink}, thus "
              + "allowing the user to create symlinks (resolved and unresolved)")
  public boolean allowUnresolvedSymlinks;

  @Option(
      name = "experimental_output_paths",
      converter = OutputPathsConverter.class,
      defaultValue = "off",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.EXECUTION
      },
      help =
          "Which model to use for where in the output tree rules write their outputs, particularly "
              + "for multi-platform / multi-configuration builds. This is highly experimental. See "
              + "https://github.com/bazelbuild/bazel/issues/6526 for details.")
  public OutputPathsMode outputPathsMode;

  @Option(
      name = "enable_runfiles",
      oldName = "experimental_enable_runfiles",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Enable runfiles symlink tree; By default, it's off on Windows, on on other platforms.")
  public TriState enableRunfiles;

  @Option(
      name = "modify_execution_info",
      converter = ExecutionInfoModifier.Converter.class,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {
        OptionEffectTag.EXECUTION,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      defaultValue = "",
      help =
          "Add or remove keys from an action's execution info based on action mnemonic.  "
              + "Applies only to actions which support execution info. Many common actions "
              + "support execution info, e.g. Genrule, CppCompile, Javac, StarlarkAction, "
              + "TestRunner. When specifying multiple values, order matters because "
              + "many regexes may apply to the same mnemonic.\n\n"
              + "Syntax: \"regex=[+-]key,[+-]key,...\".\n\n"
              + "Examples:\n"
              + "  '.*=+x,.*=-y,.*=+z' adds 'x' and 'z' to, and removes 'y' from, "
              + "the execution info for all actions.\n"
              + "  'Genrule=+requires-x' adds 'requires-x' to the execution info for "
              + "all Genrule actions.\n"
              + "  '(?!Genrule).*=-requires-x' removes 'requires-x' from the execution info for "
              + "all non-Genrule actions.\n")
  public ExecutionInfoModifier executionInfoModifier;

  @Option(
      name = "experimental_genquery_use_graphless_query",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help = "Whether to use graphless query and disable output ordering.")
  public TriState useGraphlessQuery;

  @Option(
      name = "include_config_fragments_provider",
      defaultValue = "off",
      converter = IncludeConfigFragmentsEnumConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = OptionMetadataTag.HIDDEN,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "INTERNAL BLAZE DEVELOPER FEATURE: If \"direct\", all configured targets expose "
              + "RequiredConfigFragmentsProvider with the configuration fragments they directly "
              + "require (use \"direct_host_only\" to limit to targets in the host configuration). "
              + "If \"transitive\", they do the same but also include the fragments their "
              + "transitive dependencies require. If \"off\", the provider is omitted. "
              + ""
              + "If not \"off\", this also populates config_setting's "
              + " ConfigMatchingProvider.requiredFragmentOptions with the fragment options "
              + " the config_setting requires."
              + ""
              + "Be careful using this feature: it adds memory to every configured target in the "
              + "build")
  public IncludeConfigFragmentsEnum includeRequiredConfigFragmentsProvider;

  @Option(
      name = "experimental_inprocess_symlink_creation",
      defaultValue = "false",
      converter = BooleanConverter.class,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
      help = "Whether to make direct file system calls to create symlink trees")
  public boolean inprocessSymlinkCreation;

  @Option(
      name = "experimental_skip_runfiles_manifests",
      defaultValue = "false",
      converter = BooleanConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = OptionMetadataTag.HIDDEN,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.EXECUTION,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS
      },
      help =
          "If enabled, Bazel does not create runfiles symlink manifests. This flag is ignored "
              + "if --experimental_enable_runfiles is set to false.")
  public boolean skipRunfilesManifests;

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
      name = "experimental_enable_aggregating_middleman",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Whether to enable the use of AggregatingMiddleman in rules.")
  public boolean enableAggregatingMiddleman;

  // TODO(b/132346407): Remove when all usages are gone.
  @Option(
      name = "experimental_enable_shorthand_aliases",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "When enabled, alternate names can be assigned to Starlark-defined flags.")
  public boolean enableShorthandAliases;

  /** Ways configured targets may provide the {@link Fragment}s they require. */
  public enum IncludeConfigFragmentsEnum {
    /**
     * Don't offer the provider at all. This is best for most builds, which don't use this
     * information and don't need the extra memory hit over every configured target.
     */
    OFF,
    /**
     * Provide the fragments required <em>directly</em> by this rule if it is being analyzed in the
     * host configuration.
     */
    DIRECT_HOST_ONLY,
    /** Provide the fragments required <em>directly</em> by this rule. */
    DIRECT,
    /** Provide the fragments required by this rule and its transitive dependencies. */
    TRANSITIVE;
  }

  /** Enum converter for --include_config_fragments_provider. */
  public static class IncludeConfigFragmentsEnumConverter
      extends EnumConverter<IncludeConfigFragmentsEnum> {
    public IncludeConfigFragmentsEnumConverter() {
      super(IncludeConfigFragmentsEnum.class, "include config fragments provider option");
    }
  }

  /** Used to specify which sanitizer is enabled in the current APK split. */
  public enum FatApkSplitSanitizer {
    NONE(null, ""),
    HWASAN("hwasan", "-hwasan");

    private FatApkSplitSanitizer(String feature, String androidLibDirSuffix) {
      this.feature = feature;
      this.androidLibDirSuffix = androidLibDirSuffix;
    }

    public final String feature;
    public final String androidLibDirSuffix;
  }

  /** Converter for {@link FatApkSplitSanitizer}. */
  public static class FatApkSplitSanitizerConverter extends EnumConverter<FatApkSplitSanitizer> {
    public FatApkSplitSanitizerConverter() {
      super(FatApkSplitSanitizer.class, "fat apk split sanitizer");
    }
  }

  @Option(
      name = "fat_apk_split_sanitizer",
      defaultValue = "NONE",
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = {OptionMetadataTag.INTERNAL},
      converter = FatApkSplitSanitizerConverter.class)
  public FatApkSplitSanitizer fatApkSplitSanitizer;

  @Override
  public FragmentOptions getHost() {
    CoreOptions host = (CoreOptions) getDefault();

    host.outputDirectoryName = "host";
    host.transitionDirectoryNameFragment = transitionDirectoryNameFragment;
    host.compilationMode = hostCompilationMode;
    host.isHost = true;
    host.isExec = false;
    host.configsMode = configsMode;
    host.outputPathsMode = outputPathsMode;
    host.enableRunfiles = enableRunfiles;
    host.executionInfoModifier = executionInfoModifier;
    host.commandLineBuildVariables = commandLineBuildVariables;
    host.enforceConstraints = enforceConstraints;
    host.mergeGenfilesDirectory = mergeGenfilesDirectory;
    host.cpu = hostCpu;
    host.includeRequiredConfigFragmentsProvider = includeRequiredConfigFragmentsProvider;
    host.enableAggregatingMiddleman = enableAggregatingMiddleman;

    // === Runfiles ===
    host.buildRunfilesManifests = buildRunfilesManifests;
    host.buildRunfiles = buildRunfiles;
    host.legacyExternalRunfiles = legacyExternalRunfiles;
    host.remotableSourceManifestActions = remotableSourceManifestActions;
    host.skipRunfilesManifests = skipRunfilesManifests;

    // === Filesets ===
    host.strictFilesetOutput = strictFilesetOutput;
    host.strictFilesets = strictFilesets;

    // === Linkstamping ===
    // Disable all link stamping for the host configuration, to improve action
    // cache hit rates for tools.
    host.stampBinaries = false;

    // === Visibility ===
    host.checkVisibility = checkVisibility;

    // === Licenses ===
    host.checkLicenses = checkLicenses;

    // === Pass on C++ compiler features.
    host.defaultFeatures = ImmutableList.copyOf(defaultFeatures);

    // Save host options in case of a further exec->host transition.
    host.hostCpu = hostCpu;
    host.hostCompilationMode = hostCompilationMode;

    return host;
  }

  @Override
  public CoreOptions getNormalized() {
    CoreOptions result = (CoreOptions) clone();

    if (collapseDuplicateDefines) {
      LinkedHashMap<String, String> flagValueByName = new LinkedHashMap<>();
      for (Map.Entry<String, String> entry : result.commandLineBuildVariables) {
        // If the same --define flag is passed multiple times we keep the last value.
        flagValueByName.put(entry.getKey(), entry.getValue());
      }

      // This check is an optimization to avoid creating a new list if the normalization was a
      // no-op.
      if (flagValueByName.size() != result.commandLineBuildVariables.size()) {
        result.commandLineBuildVariables =
            flagValueByName.entrySet().stream()
                // The entries in the transformed list must be serializable.
                .map(SimpleEntry::new)
                .collect(toImmutableList());
      }
    }

    return result;
  }
}
