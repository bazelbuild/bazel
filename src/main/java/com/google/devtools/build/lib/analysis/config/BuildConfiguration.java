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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Splitter;
import com.google.common.base.Verify;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.MutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Instances of BuildConfiguration represent a collection of context
 * information which may affect a build (for example: the target platform for
 * compilation, or whether or not debug tables are required).  In fact, all
 * "environmental" information (e.g. from the tool's command-line, as opposed
 * to the BUILD file) that can affect the output of any build tool should be
 * explicitly represented in the BuildConfiguration instance.
 *
 * <p>A single build may require building tools to run on a variety of
 * platforms: when compiling a server application for production, we must build
 * the build tools (like compilers) to run on the host platform, but cross-compile
 * the application for the production environment.
 *
 * <p>There is always at least one BuildConfiguration instance in any build:
 * the one representing the host platform. Additional instances may be created,
 * in a cross-compilation build, for example.
 *
 * <p>Instances of BuildConfiguration are canonical:
 * <pre>c1.equals(c2) <=> c1==c2.</pre>
 */
@SkylarkModule(name = "configuration",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Data required for the analysis of a target that comes from targets that "
        + "depend on it and not targets that it depends on.")
public final class BuildConfiguration implements BuildEvent {
  /**
   * An interface for language-specific configurations.
   *
   * <p>All implementations must be immutable and communicate this as clearly as possible
   * (e.g. declare {@link ImmutableList} signatures on their interfaces vs. {@link List}).
   * This is because fragment instances may be shared across configurations.
   */
  public abstract static class Fragment {
    /**
     * Validates the options for this Fragment. Issues warnings for the
     * use of deprecated options, and warnings or errors for any option settings
     * that conflict.
     */
    @SuppressWarnings("unused")
    public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    }

    /**
     * Adds mapping of names to values of "Make" variables defined by this configuration.
     */
    @SuppressWarnings("unused")
    public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {
    }

    /**
     * Returns a fragment of the output directory name for this configuration. The output
     * directory for the whole configuration contains all the short names by all fragments.
     */
    @Nullable
    public String getOutputDirectoryName() {
      return null;
    }

    /**
     * The platform name is a concatenation of fragment platform names.
     */
    public String getPlatformName() {
      return "";
    }

    /**
     * Add items to the action environment.
     *
     * @param builder the map to add environment variables to
     */
    public void setupActionEnvironment(Map<String, String> builder) {
    }

    /**
     * Returns the shell to be used.
     *
     * <p>Each configuration instance must have at most one fragment that returns non-null.
     */
    @SuppressWarnings("unused")
    public PathFragment getShellExecutable() {
      return null;
    }

    /**
     * Returns { 'option name': 'alternative default' } entries for options where the
     * "real default" should be something besides the default specified in the {@link Option}
     * declaration.
     */
    public Map<String, Object> lateBoundOptionDefaults() {
      return ImmutableMap.of();
    }

    /**
     * Return set of features enabled by this configuration.
     */
    public ImmutableSet<String> configurationEnabledFeatures(RuleContext ruleContext) {
      return ImmutableSet.of();
    }

    /**
     * @return false if a Fragment understands that it won't be able to work with a given strategy,
     *     or true otherwise.
     */
    public boolean compatibleWithStrategy(String strategyName) {
      return true;
    }

    /**
     * Returns the transition that produces the "artifact owner" for this configuration, or null
     * if this configuration is its own owner.
     *
     * <p>If multiple fragments return the same transition, that transition is only applied
     * once. Multiple fragments may not return different non-null transitions.
     */
    @Nullable
    public PatchTransition getArtifactOwnerTransition() {
      return null;
    }

    /**
     * Returns an extra transition that should apply to top-level targets in this
     * configuration. Returns null if no transition is needed.
     *
     * <p>Overriders should not change {@link FragmentOptions} not associated with their fragment.
     *
     * <p>If multiple fragments specify a transition, they're composed together in a
     * deterministic but undocumented order (so don't write code expecting a specific order).
     */
    @Nullable
    public PatchTransition topLevelConfigurationHook(Target toTarget) {
      return null;
    }

    /** Returns a reserved set of action mnemonics. These cannot be used from a Skylark action. */
    public ImmutableSet<String> getReservedActionMnemonics() {
      return ImmutableSet.of();
    }
  }

  public static final Label convertOptionsLabel(String input) throws OptionsParsingException {
    try {
      // Check if the input starts with '/'. We don't check for "//" so that
      // we get a better error message if the user accidentally tries to use
      // an absolute path (starting with '/') for a label.
      if (!input.startsWith("/") && !input.startsWith("@")) {
        input = "//" + input;
      }
      return Label.parseAbsolute(input);
    } catch (LabelSyntaxException e) {
      throw new OptionsParsingException(e.getMessage());
    }
  }

  /**
   * A converter from strings to Labels.
   */
  public static class LabelConverter implements Converter<Label> {
    @Override
    public Label convert(String input) throws OptionsParsingException {
      return convertOptionsLabel(input);
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /** A converter from comma-separated strings to Label lists. */
  public static class LabelListConverter implements Converter<List<Label>> {
    @Override
    public List<Label> convert(String input) throws OptionsParsingException {
      ImmutableList.Builder result = ImmutableList.builder();
      for (String label : Splitter.on(",").omitEmptyStrings().split(input)) {
        result.add(convertOptionsLabel(label));
      }
      return result.build();
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /**
   * A converter that returns null if the input string is empty, otherwise it converts
   * the input to a label.
   */
  public static class EmptyToNullLabelConverter implements Converter<Label> {
    @Override
    public Label convert(String input) throws OptionsParsingException {
      return input.isEmpty() ? null : convertOptionsLabel(input);
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /**
   * A label converter that returns a default value if the input string is empty.
   */
  public static class DefaultLabelConverter implements Converter<Label> {
    private final Label defaultValue;

    protected DefaultLabelConverter(String defaultValue) {
      this.defaultValue = defaultValue.equals("null")
          ? null
          : Label.parseAbsoluteUnchecked(defaultValue);
    }

    @Override
    public Label convert(String input) throws OptionsParsingException {
      return input.isEmpty() ? defaultValue : convertOptionsLabel(input);
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /** Flag converter for a map of unique keys with optional labels as values. */
  public static class LabelMapConverter implements Converter<Map<String, Label>> {
    @Override
    public Map<String, Label> convert(String input) throws OptionsParsingException {
      // Use LinkedHashMap so we can report duplicate keys more easily while preserving order
      Map<String, Label> result = new LinkedHashMap<>();
      for (String entry : Splitter.on(",").omitEmptyStrings().trimResults().split(input)) {
        String key;
        Label label;
        int sepIndex = entry.indexOf('=');
        if (sepIndex < 0) {
          key = entry;
          label = null;
        } else {
          key = entry.substring(0, sepIndex);
          String value = entry.substring(sepIndex + 1);
          label = value.isEmpty() ? null : convertOptionsLabel(value);
        }
        if (result.containsKey(key)) {
          throw new OptionsParsingException("Key '" + key + "' appears twice");
        }
        result.put(key, label);
      }
      return Collections.unmodifiableMap(result);
    }

    @Override
    public String getTypeDescription() {
      return "a comma-separated list of keys optionally followed by '=' and a label";
    }
  }

  /** TODO(bazel-team): document this */
  public static class PluginOptionConverter implements Converter<Map.Entry<String, String>> {
    @Override
    public Map.Entry<String, String> convert(String input) throws OptionsParsingException {
      int index = input.indexOf('=');
      if (index == -1) {
        throw new OptionsParsingException("Plugin option not in the plugin=option format");
      }
      String option = input.substring(0, index);
      String value = input.substring(index + 1);
      return Maps.immutableEntry(option, value);
    }

    @Override
    public String getTypeDescription() {
      return "An option for a plugin";
    }
  }

  /** TODO(bazel-team): document this */
  public static class RunsPerTestConverter extends PerLabelOptions.PerLabelOptionsConverter {
    @Override
    public PerLabelOptions convert(String input) throws OptionsParsingException {
      try {
        return parseAsInteger(input);
      } catch (NumberFormatException ignored) {
        return parseAsRegex(input);
      }
    }

    private PerLabelOptions parseAsInteger(String input)
        throws NumberFormatException, OptionsParsingException {
      int numericValue = Integer.parseInt(input);
      if (numericValue <= 0) {
        throw new OptionsParsingException("'" + input + "' should be >= 1");
      } else {
        RegexFilter catchAll = new RegexFilter(Collections.singletonList(".*"),
            Collections.<String>emptyList());
        return new PerLabelOptions(catchAll, Collections.singletonList(input));
      }
    }

    private PerLabelOptions parseAsRegex(String input) throws OptionsParsingException {
      PerLabelOptions testRegexps = super.convert(input);
      if (testRegexps.getOptions().size() != 1) {
        throw new OptionsParsingException(
            "'" + input + "' has multiple runs for a single pattern");
      }
      String runsPerTest = Iterables.getOnlyElement(testRegexps.getOptions());
      try {
        int numericRunsPerTest = Integer.parseInt(runsPerTest);
        if (numericRunsPerTest <= 0) {
          throw new OptionsParsingException("'" + input + "' has a value < 1");
        }
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' has a non-numeric value", e);
      }
      return testRegexps;
    }

    @Override
    public String getTypeDescription() {
      return "a positive integer or test_regex@runs. This flag may be passed more than once";
    }
  }

  /**
   * Values for the --strict_*_deps option
   */
  public static enum StrictDepsMode {
    /** Silently allow referencing transitive dependencies. */
    OFF,
    /** Warn about transitive dependencies being used directly. */
    WARN,
    /** Fail the build when transitive dependencies are used directly. */
    ERROR,
    /** Transition to strict by default. */
    STRICT,
    /** When no flag value is specified on the command line. */
    DEFAULT
  }

  /**
   * Converter for the --strict_*_deps option.
   */
  public static class StrictDepsConverter extends EnumConverter<StrictDepsMode> {
    public StrictDepsConverter() {
      super(StrictDepsMode.class, "strict dependency checking level");
    }
  }

  /**
   * Options that affect the value of a BuildConfiguration instance.
   *
   * <p>(Note: any client that creates a view will also need to declare
   * BuildView.Options, which affect the <i>mechanism</i> of view construction,
   * even if they don't affect the value of the BuildConfiguration instances.)
   *
   * <p>IMPORTANT: when adding new options, be sure to consider whether those
   * values should be propagated to the host configuration or not.
   *
   * <p>ALSO IMPORTANT: all option types MUST define a toString method that
   * gives identical results for semantically identical option values. The
   * simplest way to ensure that is to return the input string.
   */
  public static class Options extends FragmentOptions implements Cloneable {
    @Option(
      name = "experimental_separate_genfiles_directory",
      defaultValue = "true",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help = "Whether to have a separate genfiles directory or fold it into the bin directory"
    )
    public boolean separateGenfilesDirectory;

    @Option(
      name = "define",
      converter = Converters.AssignmentConverter.class,
      defaultValue = "",
      category = "semantics",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS },
      help = "Each --define option specifies an assignment for a build variable."
    )
    public List<Map.Entry<String, String>> commandLineBuildVariables;

    @Option(
      name = "cpu",
      defaultValue = "",
      category = "semantics",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS },
      help = "The target CPU."
    )
    public String cpu;

    @Option(
      name = "min_param_file_size",
      defaultValue = "32768",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
          OptionEffectTag.LOADING_AND_ANALYSIS,
          OptionEffectTag.EXECUTION,
          OptionEffectTag.ACTION_COMMAND_LINES},
      help = "Minimum command line length before creating a parameter file."
    )
    public int minParamFileSize;

    @Option(
      name = "experimental_extended_sanity_checks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = { OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help =
          "Enables internal validation checks to make sure that configured target "
              + "implementations only access things they should. Causes a performance hit."
    )
    public boolean extendedSanityChecks;

    @Option(
      name = "strict_filesets",
      defaultValue = "false",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = { OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT },
      help =
          "If this option is enabled, filesets crossing package boundaries are reported "
              + "as errors. It does not work when check_fileset_dependencies_recursively is "
              + "disabled."
    )
    public boolean strictFilesets;

    @Option(
      name = "stamp",
      defaultValue = "false",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help = "Stamp binaries with the date, username, hostname, workspace information, etc."
    )
    public boolean stampBinaries;

    // This default value is always overwritten in the case of "bazel coverage" by
    // a value returned by InstrumentationFilterSupport.computeInstrumentationFilter.
    @Option(
      name = "instrumentation_filter",
      converter = RegexFilter.RegexFilterConverter.class,
      defaultValue = "-/javatests[/:],-/test/java[/:]",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "When coverage is enabled, only rules with names included by the "
              + "specified regex-based filter will be instrumented. Rules prefixed "
              + "with '-' are excluded instead. Note that only non-test rules are "
              + "instrumented unless --instrument_test_targets is enabled."
    )
    public RegexFilter instrumentationFilter;

    @Option(
      name = "instrument_test_targets",
      defaultValue = "false",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "When coverage is enabled, specifies whether to consider instrumenting test rules. "
              + "When set, test rules included by --instrumentation_filter are instrumented. "
              + "Otherwise, test rules are always excluded from coverage instrumentation."
    )
    public boolean instrumentTestTargets;

    @Option(
      name = "host_cpu",
      defaultValue = "",
      category = "semantics",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS },
      help = "The host CPU."
    )
    public String hostCpu;

    @Option(
      name = "compilation_mode",
      abbrev = 'c',
      converter = CompilationMode.Converter.class,
      defaultValue = "fastbuild",
      category = "semantics", // Should this be "flags"?
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES },
      help = "Specify the mode the binary will be built in. " + "Values: 'fastbuild', 'dbg', 'opt'."
    )
    public CompilationMode compilationMode;

    /**
     * This option is used internally to set output directory name of the <i>host</i> configuration
     * to a constant, so that the output files for the host are completely independent of those for
     * the target, no matter what options are in force (k8/piii, opt/dbg, etc).
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
      metadataTags = { OptionMetadataTag.INTERNAL }
    )
    public String outputDirectoryName;

    @Option(
      name = "platform_suffix",
      defaultValue = "null",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
          OptionEffectTag.LOSES_INCREMENTAL_STATE,
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help = "Specifies a suffix to be added to the configuration directory."
    )
    public String platformSuffix;

    // TODO(bazel-team): The test environment is actually computed in BlazeRuntime and this option
    // is not read anywhere else. Thus, it should be in a different options class, preferably one
    // specific to the "test" command or maybe in its own configuration fragment.
    @Option(
      name = "test_env",
      converter = Converters.OptionalAssignmentConverter.class,
      allowMultiple = true,
      defaultValue = "",
      category = "testing",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = { OptionEffectTag.TEST_RUNNER },
      help =
          "Specifies additional environment variables to be injected into the test runner "
              + "environment. Variables can be either specified by name, in which case its value "
              + "will be read from the Bazel client environment, or by the name=value pair. "
              + "This option can be used multiple times to specify several variables. "
              + "Used only by the 'bazel test' command."
    )
    public List<Map.Entry<String, String>> testEnvironment;

    // TODO(bazel-team): The set of available variables from the client environment for actions
    // is computed independently in CommandEnvironment to inject a more restricted client
    // environment to skyframe.
    @Option(
      name = "action_env",
      converter = Converters.OptionalAssignmentConverter.class,
      allowMultiple = true,
      defaultValue = "",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          "Specifies the set of environment variables available to actions. "
              + "Variables can be either specified by name, in which case the value will be "
              + "taken from the invocation environment, or by the name=value pair which sets "
              + "the value independent of the invocation environment. This option can be used "
              + "multiple times; for options given for the same variable, the latest wins, options "
              + "for different variables accumulate."
    )
    public List<Map.Entry<String, String>> actionEnvironment;

    @Option(
      name = "collect_code_coverage",
      defaultValue = "false",
      category = "testing",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "If specified, Bazel will instrument code (using offline instrumentation where "
              + "possible) and will collect coverage information during tests. Only targets that "
              + " match --instrumentation_filter will be affected. Usually this option should "
              + " not be specified directly - 'bazel coverage' command should be used instead."
    )
    public boolean collectCodeCoverage;

    @Option(
      name = "coverage_support",
      converter = LabelConverter.class,
      defaultValue = "@bazel_tools//tools/test:coverage_support",
      category = "testing",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help =
          "Location of support files that are required on the inputs of every test action "
              + "that collects code coverage. Defaults to '//tools/test:coverage_support'."
    )
    public Label coverageSupport;

    @Option(
      name = "coverage_report_generator",
      converter = LabelConverter.class,
      defaultValue = "@bazel_tools//tools/test:coverage_report_generator",
      category = "testing",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS
      },
      help =
          "Location of the binary that is used to generate coverage reports. This must "
              + "currently be a filegroup that contains a single file, the binary. Defaults to "
              + "'//tools/test:coverage_report_generator'."
    )
    public Label coverageReportGenerator;

    @Option(
      name = "experimental_use_llvm_covmap",
      defaultValue = "false",
      category = "experimental",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS
      },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help =
          "If specified, Bazel will generate llvm-cov coverage map information rather than "
              + "gcov when collect_code_coverage is enabled."
    )
    public boolean useLLVMCoverageMapFormat;

    @Option(
      name = "build_runfile_manifests",
      defaultValue = "true",
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "If true, write runfiles manifests for all targets.  "
              + "If false, omit them."
    )
    public boolean buildRunfilesManifests;

    @Option(
      name = "build_runfile_links",
      defaultValue = "true",
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "If true, build runfiles symlink forests for all targets.  "
              + "If false, write only manifests when possible."
    )
    public boolean buildRunfiles;

    @Option(
      name = "legacy_external_runfiles",
      defaultValue = "true",
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "If true, build runfiles symlink forests for external repositories under "
              + ".runfiles/wsname/external/repo (in addition to .runfiles/repo)."
    )
    public boolean legacyExternalRunfiles;

    @Option(
      name = "check_fileset_dependencies_recursively",
      defaultValue = "true",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "If false, fileset targets will, whenever possible, create "
              + "symlinks to directories instead of creating one symlink for each "
              + "file inside the directory. Disabling this will significantly "
              + "speed up fileset builds, but targets that depend on filesets will "
              + "not be rebuilt if files are added, removed or modified in a "
              + "subdirectory which has not been traversed."
    )
    public boolean checkFilesetDependenciesRecursively;

    @Option(
      name = "experimental_skyframe_native_filesets",
      defaultValue = "false",
      category = "experimental",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = { OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help =
          "If true, Blaze will use the skyframe-native implementation of the Fileset rule."
              + " This offers improved performance in incremental builds of Filesets as well as"
              + " correct incremental behavior, but is not yet stable. The default is false,"
              + " meaning Blaze uses the legacy impelementation of Fileset."
    )
    public boolean skyframeNativeFileset;

    @Option(
      name = "run_under",
      category = "run",
      defaultValue = "null",
      converter = RunUnderConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.ACTION_COMMAND_LINES},
      help =
          "Prefix to insert in front of command before running. "
              + "Examples:\n"
              + "\t--run_under=valgrind\n"
              + "\t--run_under=strace\n"
              + "\t--run_under='strace -c'\n"
              + "\t--run_under='valgrind --quiet --num-callers=20'\n"
              + "\t--run_under=//package:target\n"
              + "\t--run_under='//package:target --options'\n"
    )
    public RunUnder runUnder;

    @Option(
      name = "distinct_host_configuration",
      defaultValue = "true",
      category = "strategy",
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
              + "Please read the user manual for the full explanation."
    )
    public boolean useDistinctHostConfiguration;

    @Option(
      name = "check_visibility",
      defaultValue = "true",
      category = "checking",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = { OptionEffectTag.BUILD_FILE_SEMANTICS },
      help = "If disabled, visibility errors are demoted to warnings."
    )
    public boolean checkVisibility;

    // Moved from viewOptions to here because license information is very expensive to serialize.
    // Having it here allows us to skip computation of transitive license information completely
    // when the setting is disabled.
    @Option(
      name = "check_licenses",
      defaultValue = "false",
      category = "checking",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = { OptionEffectTag.BUILD_FILE_SEMANTICS },
      help =
          "Check that licensing constraints imposed by dependent packages "
              + "do not conflict with distribution modes of the targets being built. "
              + "By default, licenses are not checked."
    )
    public boolean checkLicenses;

    @Option(
      name = "enforce_constraints",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = { OptionEffectTag.BUILD_FILE_SEMANTICS },
      help =
          "Checks the environments each target is compatible with and reports errors if any "
              + "target has dependencies that don't support the same environments",
      oldName = "experimental_enforce_constraints"
    )
    public boolean enforceConstraints;

    @Option(
      name = "experimental_action_listener",
      allowMultiple = true,
      defaultValue = "",
      category = "experimental",
      converter = LabelListConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.EXECUTION },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help = "Use action_listener to attach an extra_action to existing build actions."
    )
    public List<Label> actionListeners;

    // TODO(bazel-team): Either remove this flag once transparent compression is shown to not
    // noticeably affect running time, or keep this flag and move it into a new configuration
    // fragment.
    @Option(
      name = "experimental_transparent_compression",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = { OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help =
          "Enables gzip compression for the contents of FileWriteActions, which reduces "
              + "memory usage in the analysis phase at the expense of additional time overhead."
    )
    public boolean transparentCompression;

    @Option(
      name = "is host configuration",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = { OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION },
      metadataTags = { OptionMetadataTag.INTERNAL },
      help = "Shows whether these options are set for host configuration."
    )
    public boolean isHost;

    @Option(
      name = "features",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "The given features will be enabled or disabled by default for all packages. "
              + "Specifying -<feature> will disable the feature globally. "
              + "Negative features always override positive ones. "
              + "This flag is used to enable rolling out default feature changes without a "
              + "Blaze release."
    )
    public List<String> defaultFeatures;

    @Option(
      name = "target_environment",
      converter = LabelListConverter.class,
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = { OptionEffectTag.CHANGES_INPUTS },
      help =
          "Declares this build's target environment. Must be a label reference to an "
              + "\"environment\" rule. If specified, all top-level targets must be "
              + "compatible with this environment."
    )
    public List<Label> targetEnvironments;

    @Option(
      name = "auto_cpu_environment_group",
      // TODO(b/67853005): Remove when all usage of experimental_auto_cpu_environment_group is
      // removed
      oldName = "experimental_auto_cpu_environment_group",
      converter = EmptyToNullLabelConverter.class,
      defaultValue = "",
      category = "flags",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Declare the environment_group to use for automatically mapping cpu values to "
              + "target_environment values."
    )
    public Label autoCpuEnvironmentGroup;

    /**
     * Values for --experimental_dynamic_configs.
     */
    public enum ConfigsMode {
      /** Only include the configuration fragments each rule needs. */
      ON,
      /** Always including all fragments known to Blaze. */
      NOTRIM,
    }

    /**
     * Converter for --experimental_dynamic_configs.
     */
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
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help =
          "Instantiates build configurations with the specified properties"
    )
    public ConfigsMode configsMode;

    @Option(
      name = "experimental_enable_runfiles",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS },
      metadataTags = { OptionMetadataTag.EXPERIMENTAL },
      help = "Enable runfiles; off on Windows, on on other platforms"
    )
    public TriState enableRunfiles;

    @Option(
      name = "windows_exe_launcher",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = { OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS },
      help =
          "Build a Windows exe launcher for sh_binary rule, "
              + "it has no effect on other platforms than Windows"
    )
    public boolean windowsExeLauncher;

    @Override
    public FragmentOptions getHost() {
      Options host = (Options) getDefault();

      host.outputDirectoryName = "host";
      host.compilationMode = CompilationMode.OPT;
      host.isHost = true;
      host.configsMode = configsMode;
      host.enableRunfiles = enableRunfiles;
      host.windowsExeLauncher = windowsExeLauncher;
      host.commandLineBuildVariables = commandLineBuildVariables;
      host.enforceConstraints = enforceConstraints;
      host.separateGenfilesDirectory = separateGenfilesDirectory;
      host.cpu = hostCpu;

      // === Runfiles ===
      host.buildRunfilesManifests = buildRunfilesManifests;
      host.buildRunfiles = buildRunfiles;

      // === Linkstamping ===
      // Disable all link stamping for the host configuration, to improve action
      // cache hit rates for tools.
      host.stampBinaries = false;

      // === Visibility ===
      host.checkVisibility = checkVisibility;

      // === Licenses ===
      host.checkLicenses = checkLicenses;

      // === Fileset ===
      host.skyframeNativeFileset = skyframeNativeFileset;

      // === Pass on C++ compiler features.
      host.defaultFeatures = ImmutableList.copyOf(defaultFeatures);

      return host;
    }

    @Override
    public Map<String, Set<Label>> getDefaultsLabels(BuildConfiguration.Options commonOptions) {
      return ImmutableMap.<String, Set<Label>>of(
          "coverage_support", ImmutableSet.of(coverageSupport),
          "coverage_report_generator", ImmutableSet.of(coverageReportGenerator));
    }
  }

  private final String checksum;

  private final ImmutableMap<Class<? extends Fragment>, Fragment> fragments;
  private final ImmutableMap<String, Class<? extends Fragment>> skylarkVisibleFragments;
  private final RepositoryName mainRepositoryName;
  private final ImmutableSet<String> reservedActionMnemonics;

  /**
   * Directories in the output tree.
   *
   * <p>The computation of the output directory should be a non-injective mapping from
   * BuildConfiguration instances to strings. The result should identify the aspects of the
   * configuration that should be reflected in the output file names.  Furthermore the
   * returned string must not contain shell metacharacters.
   *
   * <p>For configuration settings which are NOT part of the output directory name,
   * rebuilding with a different value of such a setting will build in
   * the same output directory.  This means that any actions whose
   * keys (see Action.getKey()) have changed will be rerun.  That
   * may result in a lot of recompilation.
   *
   * <p>For configuration settings which ARE part of the output directory name,
   * rebuilding with a different value of such a setting will rebuild
   * in a different output directory; this will result in higher disk
   * usage and more work the <i>first</i> time you rebuild with a different
   * setting, but will result in less work if you regularly switch
   * back and forth between different settings.
   *
   * <p>With one important exception, it's sound to choose any subset of the
   * config's components for this string, it just alters the dimensionality
   * of the cache.  In other words, it's a trade-off on the "injectiveness"
   * scale: at one extreme (output directory name contains all data in the config, and is
   * thus injective) you get extremely precise caching (no competition for the
   * same output-file locations) but you have to rebuild for even the
   * slightest change in configuration.  At the other extreme (the output
   * (directory name is a constant) you have very high competition for
   * output-file locations, but if a slight change in configuration doesn't
   * affect a particular build step, you're guaranteed not to have to
   * rebuild it. The important exception has to do with multiple configurations: every
   * configuration in the build must have a different output directory name so that
   * their artifacts do not conflict.
   *
   * <p>The host configuration is special-cased: in order to guarantee that its output directory
   * is always separate from that of the target configuration, we simply pin it to "host". We do
   * this so that the build works even if the two configurations are too close (which is common)
   * and so that the path of artifacts in the host configuration is a bit more readable.
   */
  private enum OutputDirectory {
    BIN("bin"),
    GENFILES("genfiles"),
    MIDDLEMAN(true),
    TESTLOGS("testlogs"),
    COVERAGE("coverage-metadata"),
    INCLUDE(BlazeDirectories.RELATIVE_INCLUDE_DIR),
    OUTPUT(false);

    private final PathFragment nameFragment;
    private final boolean middleman;

    /**
     * This constructor is for roots without suffixes, e.g.,
     * [[execroot/repo]/bazel-out/local-fastbuild].
     * @param isMiddleman whether the root should be a middleman root or a "normal" derived root.
     */
    OutputDirectory(boolean isMiddleman) {
      this.nameFragment = PathFragment.EMPTY_FRAGMENT;
      this.middleman = isMiddleman;
    }

    OutputDirectory(String name) {
      this.nameFragment = PathFragment.create(name);
      this.middleman = false;
    }

    Root getRoot(
        RepositoryName repositoryName, String outputDirName, BlazeDirectories directories,
        RepositoryName mainRepositoryName) {
      // e.g., execroot/repo1
      Path execRoot = directories.getExecRoot(mainRepositoryName.strippedName());
      // e.g., execroot/repo1/bazel-out/config/bin
      Path outputDir = execRoot.getRelative(directories.getRelativeOutputPath())
          .getRelative(outputDirName);
      if (middleman) {
        return INTERNER.intern(Root.middlemanRoot(execRoot, outputDir,
            repositoryName.equals(mainRepositoryName)));
      }
      // e.g., [[execroot/repo1]/bazel-out/config/bin]
      return INTERNER.intern(
          Root.asDerivedRoot(execRoot, outputDir.getRelative(nameFragment),
              repositoryName.equals(mainRepositoryName)));
    }
  }

  private final BlazeDirectories directories;
  private final String outputDirName;

  // We intern the roots for non-main repositories, so we don't keep around thousands of copies of
  // the same root.
  private static Interner<Root> INTERNER = Interners.newWeakInterner();

  // We precompute the roots for the main repository, since that's the common case.
  private final Root outputDirectoryForMainRepository;
  private final Root binDirectoryForMainRepository;
  private final Root includeDirectoryForMainRepository;
  private final Root genfilesDirectoryForMainRepository;
  private final Root coverageDirectoryForMainRepository;
  private final Root testlogsDirectoryForMainRepository;
  private final Root middlemanDirectoryForMainRepository;

  private final boolean separateGenfilesDirectory;

  // Cache this value for quicker access. We don't cache it inside BuildOptions because BuildOptions
  // is mutable, so a cached value there could fall out of date when it's updated.
  private final boolean actionsEnabled;

  // TODO(bazel-team): Move this to a configuration fragment.
  private final PathFragment shellExecutable;

  /**
   * The global "make variables" such as "$(TARGET_CPU)"; these get applied to all rules analyzed in
   * this configuration.
   */
  private final ImmutableMap<String, String> globalMakeEnv;

  private final ActionEnvironment actionEnv;
  private final ActionEnvironment testEnv;

  private final BuildOptions buildOptions;
  private final Options options;

  private final String mnemonic;
  private final String platformName;

  private final ImmutableMap<String, String> commandLineBuildVariables;

  private final int hashCode; // We can precompute the hash code as all its inputs are immutable.

  /** Data for introspecting the options used by this configuration. */
  private final TransitiveOptionDetails transitiveOptionDetails;

  /**
   * Returns true if this configuration is semantically equal to the other, with
   * the possible exception that the other has fewer fragments.
   *
   * <p>This is useful for trimming: as the same configuration gets "trimmed" while going down a
   * dependency chain, it's still the same configuration but loses some of its fragments. So we need
   * a more nuanced concept of "equality" than simple reference equality.
   */
  public boolean equalsOrIsSupersetOf(BuildConfiguration other) {
    return this.equals(other)
        || (other != null
        // TODO(gregce): add back in output root checking. This requires a better approach to
        // configuration-safe output paths. If the parent config has a fragment the child config
        // doesn't, it may inject $(FOO) into the output roots. So the child bindir might be
        // "bazel-out/arm-linux-fastbuild/bin" while the parent bindir is
        // "bazel-out/android-arm-linux-fastbuild/bin". That's pretty awkward to check here.
        //      && outputRoots.equals(other.outputRoots)
                && actionsEnabled == other.actionsEnabled
                && fragments.values().containsAll(other.fragments.values())
                && buildOptions.getOptions().containsAll(other.buildOptions.getOptions()));
  }

  /**
   * Returns {@code true} if this configuration is semantically equal to the other, including
   * checking that both have the same sets of fragments and options.
   */
  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof BuildConfiguration)) {
      return false;
    }
    BuildConfiguration otherConfig = (BuildConfiguration) other;
    return actionsEnabled == otherConfig.actionsEnabled
        && fragments.values().equals(otherConfig.fragments.values())
        && buildOptions.getOptions().equals(otherConfig.buildOptions.getOptions());
  }

  private int computeHashCode() {
    return Objects.hash(isActionsEnabled(), fragments, buildOptions.getOptions());
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  /**
   * Returns map of all the fragments for this configuration.
   */
  public ImmutableMap<Class<? extends Fragment>, Fragment> getAllFragments() {
    return fragments;
  }

  /**
   * Validates the options for this BuildConfiguration. Issues warnings for the
   * use of deprecated options, and warnings or errors for any option settings
   * that conflict.
   */
  public void reportInvalidOptions(EventHandler reporter) {
    for (Fragment fragment : fragments.values()) {
      fragment.reportInvalidOptions(reporter, this.buildOptions);
    }

    if (options.outputDirectoryName != null) {
      reporter.handle(Event.error(
          "The internal '--output directory name' option cannot be used on the command line"));
    }
  }

  /**
   * @return false if any of the fragments don't work well with the supplied strategy.
   */
  public boolean compatibleWithStrategy(final String strategyName) {
    return Iterables.all(
        fragments.values(),
        new Predicate<Fragment>() {
          @Override
          public boolean apply(@Nullable Fragment fragment) {
            return fragment.compatibleWithStrategy(strategyName);
          }
        });
  }

  /**
   * Compute the shell environment, which, at configuration level, is a pair consisting of the
   * statically set environment variables with their values and the set of environment variables to
   * be inherited from the client environment.
   */
  private ActionEnvironment setupActionEnvironment() {
    // We make a copy first to remove duplicate entries; last one wins.
    Map<String, String> actionEnv = new HashMap<>();
    // TODO(ulfjack): Remove all env variables from configuration fragments.
    for (Fragment fragment : fragments.values()) {
      fragment.setupActionEnvironment(actionEnv);
    }
    // Shell environment variables specified via options take precedence over the
    // ones inherited from the fragments. In the long run, these fragments will
    // be replaced by appropriate default rc files anyway.
    for (Map.Entry<String, String> entry : options.actionEnvironment) {
      actionEnv.put(entry.getKey(), entry.getValue());
    }
    return ActionEnvironment.split(actionEnv);
  }

  /**
   * Compute the test environment, which, at configuration level, is a pair consisting of the
   * statically set environment variables with their values and the set of environment variables to
   * be inherited from the client environment.
   */
  private ActionEnvironment setupTestEnvironment() {
    // We make a copy first to remove duplicate entries; last one wins.
    Map<String, String> testEnv = new HashMap<>();
    for (Map.Entry<String, String> entry : options.testEnvironment) {
      testEnv.put(entry.getKey(), entry.getValue());
    }
    return ActionEnvironment.split(testEnv);
  }

  /**
   * Sorts fragments by class name. This produces a stable order which, e.g., facilitates
   * consistent output from buildMneumonic.
   */
  private static final Comparator lexicalFragmentSorter =
      new Comparator<Class<? extends Fragment>>() {
        @Override
        public int compare(Class<? extends Fragment> o1, Class<? extends Fragment> o2) {
          return o1.getName().compareTo(o2.getName());
        }
      };

  /**
   * Constructs a new BuildConfiguration instance.
   */
  public BuildConfiguration(BlazeDirectories directories,
      Map<Class<? extends Fragment>, Fragment> fragmentsMap,
      BuildOptions buildOptions,
      String repositoryName) {
    this.directories = directories;
    this.fragments = ImmutableSortedMap.copyOf(fragmentsMap, lexicalFragmentSorter);

    this.skylarkVisibleFragments = buildIndexOfSkylarkVisibleFragments();

    this.buildOptions = buildOptions.clone();
    this.actionsEnabled = buildOptions.enableActions();
    this.options = buildOptions.get(Options.class);
    this.separateGenfilesDirectory = options.separateGenfilesDirectory;
    this.mainRepositoryName = RepositoryName.createFromValidStrippedName(repositoryName);

    // We can't use an ImmutableMap.Builder here; we need the ability to add entries with keys that
    // are already in the map so that the same define can be specified on the command line twice,
    // and ImmutableMap.Builder does not support that.
    Map<String, String> commandLineDefinesBuilder = new TreeMap<>();
    for (Map.Entry<String, String> define : options.commandLineBuildVariables) {
      commandLineDefinesBuilder.put(define.getKey(), define.getValue());
    }
    commandLineBuildVariables = ImmutableMap.copyOf(commandLineDefinesBuilder);

    this.mnemonic = buildMnemonic();
    this.outputDirName = (options.outputDirectoryName != null)
        ? options.outputDirectoryName : mnemonic;

    this.outputDirectoryForMainRepository =
        OutputDirectory.OUTPUT.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);
    this.binDirectoryForMainRepository =
        OutputDirectory.BIN.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);
    this.includeDirectoryForMainRepository =
        OutputDirectory.INCLUDE.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);
    this.genfilesDirectoryForMainRepository =
        OutputDirectory.GENFILES.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);
    this.coverageDirectoryForMainRepository =
        OutputDirectory.COVERAGE.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);
    this.testlogsDirectoryForMainRepository =
        OutputDirectory.TESTLOGS.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);
    this.middlemanDirectoryForMainRepository =
        OutputDirectory.MIDDLEMAN.getRoot(
            RepositoryName.MAIN, outputDirName, directories, mainRepositoryName);

    this.platformName = buildPlatformName();

    this.shellExecutable = computeShellExecutable();

    this.actionEnv = setupActionEnvironment();

    this.testEnv = setupTestEnvironment();

    this.transitiveOptionDetails = computeOptionsMap(buildOptions, fragments.values());

    ImmutableMap.Builder<String, String> globalMakeEnvBuilder = ImmutableMap.builder();
    for (Fragment fragment : fragments.values()) {
      fragment.addGlobalMakeVariables(globalMakeEnvBuilder);
    }

    globalMakeEnvBuilder.put("COMPILATION_MODE", options.compilationMode.toString());
    /*
     * Attention! Document these in the build-encyclopedia
     */
    // the bin directory and the genfiles directory
    // These variables will be used on Windows as well, so we need to make sure
    // that paths use the correct system file-separator.
    globalMakeEnvBuilder.put("BINDIR", getBinDirectory().getExecPath().getPathString());
    globalMakeEnvBuilder.put("GENDIR", getGenfilesDirectory().getExecPath().getPathString());
    globalMakeEnv = globalMakeEnvBuilder.build();

    checksum = Fingerprint.md5Digest(buildOptions.computeCacheKey());
    hashCode = computeHashCode();

    ImmutableSet.Builder<String> reservedActionMnemonics = ImmutableSet.builder();
    for (Fragment fragment : fragments.values()) {
      reservedActionMnemonics.addAll(fragment.getReservedActionMnemonics());
    }
    this.reservedActionMnemonics = reservedActionMnemonics.build();
  }

  /**
   * Returns a copy of this configuration only including the given fragments (which the current
   * configuration is assumed to have).
   */
  public BuildConfiguration clone(
      Set<Class<? extends BuildConfiguration.Fragment>> fragmentClasses,
      RuleClassProvider ruleClassProvider) {

    ClassToInstanceMap<Fragment> fragmentsMap = MutableClassToInstanceMap.create();
    for (Fragment fragment : fragments.values()) {
      if (fragmentClasses.contains(fragment.getClass())) {
        fragmentsMap.put(fragment.getClass(), fragment);
      }
    }
    BuildOptions options = buildOptions.trim(
        getOptionsClasses(fragmentsMap.keySet(), ruleClassProvider));
    BuildConfiguration newConfig =
        new BuildConfiguration(
            directories,
            fragmentsMap,
            options,
            mainRepositoryName.strippedName());
    return newConfig;
  }

  /**
   * Returns the config fragment options classes used by the given fragment types.
   */
  public static Set<Class<? extends FragmentOptions>> getOptionsClasses(
      Iterable<Class<? extends Fragment>> fragmentClasses, RuleClassProvider ruleClassProvider) {

    Multimap<Class<? extends BuildConfiguration.Fragment>, Class<? extends FragmentOptions>>
        fragmentToRequiredOptions = ArrayListMultimap.create();
    for (ConfigurationFragmentFactory fragmentLoader :
        ((ConfiguredRuleClassProvider) ruleClassProvider).getConfigurationFragments()) {
      fragmentToRequiredOptions.putAll(fragmentLoader.creates(),
          fragmentLoader.requiredOptions());
    }
    Set<Class<? extends FragmentOptions>> options = new HashSet<>();
    for (Class<? extends BuildConfiguration.Fragment> fragmentClass : fragmentClasses) {
      options.addAll(fragmentToRequiredOptions.get(fragmentClass));
    }
    return options;
  }



  private ImmutableMap<String, Class<? extends Fragment>> buildIndexOfSkylarkVisibleFragments() {
    ImmutableMap.Builder<String, Class<? extends Fragment>> builder = ImmutableMap.builder();

    for (Class<? extends Fragment> fragmentClass : fragments.keySet()) {
      String name = SkylarkModule.Resolver.resolveName(fragmentClass);
      if (name != null) {
        builder.put(name, fragmentClass);
      }
    }
    return builder.build();
  }

  /**
   * Retrieves the {@link TransitiveOptionDetails} containing data on this configuration's options.
   *
   * @see BuildConfigurationOptionDetails
   */
  TransitiveOptionDetails getTransitiveOptionDetails() {
    return transitiveOptionDetails;
  }

  /** Computes and returns the {@link TransitiveOptionDetails} for this configuration. */
  private static TransitiveOptionDetails computeOptionsMap(
      BuildOptions buildOptions, Iterable<Fragment> fragments) {
    // Collect from our fragments "alternative defaults" for options where the default
    // should be something other than what's specified in Option.defaultValue.
    Map<String, Object> lateBoundDefaults = Maps.newHashMap();
    for (Fragment fragment : fragments) {
      lateBoundDefaults.putAll(fragment.lateBoundOptionDefaults());
    }

    return TransitiveOptionDetails.forOptionsWithDefaults(
        buildOptions.getOptions(), lateBoundDefaults);
  }

  private String buildMnemonic() {
    // See explanation at declaration for outputRoots.
    String platformSuffix = (options.platformSuffix != null) ? options.platformSuffix : "";
    ArrayList<String> nameParts = new ArrayList<>();
    for (Fragment fragment : fragments.values()) {
      nameParts.add(fragment.getOutputDirectoryName());
    }
    nameParts.add(getCompilationMode() + platformSuffix);
    return Joiner.on('-').skipNulls().join(nameParts);
  }

  private String buildPlatformName() {
    StringBuilder platformNameBuilder = new StringBuilder();
    for (Fragment fragment : fragments.values()) {
      platformNameBuilder.append(fragment.getPlatformName());
    }
    return platformNameBuilder.toString();
  }

  /**
   * The platform string, suitable for use as a key into a MakeEnvironment.
   */
  public String getPlatformName() {
    return platformName;
  }

  /**
   * Returns the output directory for this build configuration.
   */
  public Root getOutputDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? outputDirectoryForMainRepository
        : OutputDirectory.OUTPUT.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns the bin directory for this build configuration.
   */
  @SkylarkCallable(name = "bin_dir", structField = true, documented = false)
  @Deprecated
  public Root getBinDirectory() {
    return getBinDirectory(RepositoryName.MAIN);
  }

  /**
   * TODO(kchodorow): This (and the other get*Directory functions) won't work with external
   * repositories without changes to how ArtifactFactory resolves derived roots. This is not an
   * issue right now because it only effects Blaze's include scanning (internal) and Bazel's
   * repositories (external) but will need to be fixed.
   */
  public Root getBinDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? binDirectoryForMainRepository
        : OutputDirectory.BIN.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns a relative path to the bin directory at execution time.
   */
  public PathFragment getBinFragment() {
    return getBinDirectory().getExecPath();
  }

  /**
   * Returns the include directory for this build configuration.
   */
  public Root getIncludeDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? includeDirectoryForMainRepository
        : OutputDirectory.INCLUDE.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns the genfiles directory for this build configuration.
   */
  @SkylarkCallable(name = "genfiles_dir", structField = true, documented = false)
  @Deprecated
  public Root getGenfilesDirectory() {
    return getGenfilesDirectory(RepositoryName.MAIN);
  }

  public Root getGenfilesDirectory(RepositoryName repositoryName) {
    if (!separateGenfilesDirectory) {
      return getBinDirectory(repositoryName);
    }

    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? genfilesDirectoryForMainRepository
        : OutputDirectory.GENFILES.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns the directory where coverage-related artifacts and metadata files
   * should be stored. This includes for example uninstrumented class files
   * needed for Jacoco's coverage reporting tools.
   */
  public Root getCoverageMetadataDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? coverageDirectoryForMainRepository
        : OutputDirectory.COVERAGE.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns the testlogs directory for this build configuration.
   */
  public Root getTestLogsDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? testlogsDirectoryForMainRepository
        : OutputDirectory.TESTLOGS.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns a relative path to the genfiles directory at execution time.
   */
  public PathFragment getGenfilesFragment() {
    return getGenfilesDirectory().getExecPath();
  }

  /**
   * Returns the path separator for the host platform. This is basically the same as {@link
   * java.io.File#pathSeparator}, except that that returns the value for this JVM, which may or may
   * not match the host platform. You should only use this when invoking tools that are known to use
   * the native path separator, i.e., the path separator for the machine that they run on.
   */
  @SkylarkCallable(name = "host_path_separator", structField = true,
      doc = "Returns the separator for PATH environment variable, which is ':' on Unix.")
  public String getHostPathSeparator() {
    // TODO(bazel-team): Maybe do this in the constructor instead? This isn't serialization-safe.
    return OS.getCurrent() == OS.WINDOWS ? ";" : ":";
  }

  /**
   * Returns the internal directory (used for middlemen) for this build configuration.
   */
  public Root getMiddlemanDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? middlemanDirectoryForMainRepository
        : OutputDirectory.MIDDLEMAN.getRoot(
            repositoryName, outputDirName, directories, mainRepositoryName);
  }

  public boolean isStrictFilesets() {
    return options.strictFilesets;
  }

  public String getMainRepositoryName() {
    return mainRepositoryName.strippedName();
  }

  /**
   * Returns the configuration-dependent string for this configuration. This is also the name of the
   * configuration's base output directory unless {@link Options#outputDirectoryName} overrides it.
   */
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  public String toString() {
    return checksum();
  }

  public ActionEnvironment getActionEnvironment() {
    return actionEnv;
  }

  @SkylarkCallable(
    name = "default_shell_env",
    structField = true,
    doc =
        "A dictionary representing the static local shell environment. It maps variables "
            + "to their values (strings)."
  )
  /**
   * Return the "fixed" part of the actions' environment variables.
   *
   * <p>An action's full set of environment variables consist of a "fixed" part and of a "variable"
   * part. The "fixed" variables are independent of the Bazel client's own environment, and are
   * returned by this function. The "variable" ones are inherited from the Bazel client's own
   * environment, and are returned by {@link getVariableShellEnvironment}.
   *
   * <p>Since values of the "fixed" variables are already known at analysis phase, it is returned
   * here as a map.
   */
  @Deprecated // Use getActionEnvironment instead.
  public ImmutableMap<String, String> getLocalShellEnvironment() {
    return actionEnv.getFixedEnv();
  }

  /**
   * Return the "variable" part of the actions' environment variables.
   *
   * <p>An action's full set of environment variables consist of a "fixed" part and of a "variable"
   * part. The "fixed" variables are independent of the Bazel client's own environment, and are
   * returned by {@link #getLocalShellEnvironment}. The "variable" ones are inherited from the Bazel
   * client's own environment, and are returned by this function.
   *
   * <p>The values of the "variable" variables are tracked in Skyframe via the {@link
   * com.google.devtools.build.lib.skyframe.SkyFunctions#CLIENT_ENVIRONMENT_VARIABLE} skyfunction.
   * This method only returns the names of those variables to be inherited, if set in the client's
   * environment. (Variables where the name is not returned in this set should not be taken from the
   * client environment.)
   */
  @Deprecated // Use getActionEnvironment instead.
  public ImmutableSet<String> getVariableShellEnvironment() {
    return actionEnv.getInheritedEnv();
  }

  /**
   * Returns the path to sh.
   */
  public PathFragment getShellExecutable() {
    return shellExecutable;
  }

  /**
   * Returns a regex-based instrumentation filter instance that used to match label
   * names to identify targets to be instrumented in the coverage mode.
   */
  public RegexFilter getInstrumentationFilter() {
    return options.instrumentationFilter;
  }

  /**
   * Returns a boolean of whether to include targets created by *_test rules in the set of targets
   * matched by --instrumentation_filter. If this is false, all test targets are excluded from
   * instrumentation.
   */
  public boolean shouldInstrumentTestTargets() {
    return options.instrumentTestTargets;
  }

  /**
   * Returns a new, unordered mapping of names to values of "Make" variables defined by this
   * configuration.
   *
   * <p>This does *not* include package-defined overrides (e.g. vardef)
   * and so should not be used by the build logic.  This is used only for
   * the 'info' command.
   *
   * <p>Command-line definitions of make enviroments override variables defined by
   * {@code Fragment.addGlobalMakeVariables()}.
   */
  public Map<String, String> getMakeEnvironment() {
    Map<String, String> makeEnvironment = new HashMap<>();
    makeEnvironment.putAll(globalMakeEnv);
    makeEnvironment.putAll(commandLineBuildVariables);
    return ImmutableMap.copyOf(makeEnvironment);
  }

  /**
   * Returns a new, unordered mapping of names that are set through the command lines.
   * (Fragments, in particular the Google C++ support, can set variables through the
   * command line.)
   */
  public ImmutableMap<String, String> getCommandLineBuildVariables() {
    return commandLineBuildVariables;
  }

  /**
   * Returns the global defaults for this configuration for the Make environment.
   */
  public ImmutableMap<String, String> getGlobalMakeEnvironment() {
    return globalMakeEnv;
  }

  /**
   * Returns the default value for the specified "Make" variable for this
   * configuration.  Returns null if no value was found.
   */
  public String getMakeVariableDefault(String var) {
    return globalMakeEnv.get(var);
  }

  /**
   * Returns a configuration fragment instances of the given class.
   */
  public <T extends Fragment> T getFragment(Class<T> clazz) {
    return clazz.cast(fragments.get(clazz));
  }

  /**
   * Returns true if the requested configuration fragment is present.
   */
  public <T extends Fragment> boolean hasFragment(Class<T> clazz) {
    return getFragment(clazz) != null;
  }

  /**
   * Returns true if all requested configuration fragment are present (this may be slow).
   */
  public boolean hasAllFragments(Set<Class<?>> fragmentClasses) {
    for (Class<?> fragmentClass : fragmentClasses) {
      if (!hasFragment(fragmentClass.asSubclass(Fragment.class))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Which fragments does this configuration contain?
   */
  public Set<Class<? extends Fragment>> fragmentClasses() {
    return fragments.keySet();
  }

  /**
   * Returns true if non-functional build stamps are enabled.
   */
  public boolean stampBinaries() {
    return options.stampBinaries;
  }

  /**
   * Returns true if extended sanity checks should be enabled.
   */
  public boolean extendedSanityChecks() {
    return options.extendedSanityChecks;
  }

  /**
   * Returns true if we are building runfiles manifests for this configuration.
   */
  public boolean buildRunfilesManifests() {
    return options.buildRunfilesManifests;
  }

  /**
   * Returns true if we are building runfiles symlinks for this configuration.
   */
  public boolean buildRunfiles() {
    return options.buildRunfiles;
  }

  /**
   * Returns if we are building external runfiles symlinks using the old-style structure.
   */
  public boolean legacyExternalRunfiles() {
    return options.legacyExternalRunfiles;
  }

  public boolean getCheckFilesetDependenciesRecursively() {
    return options.checkFilesetDependenciesRecursively;
  }

  public boolean getSkyframeNativeFileset() {
    return options.skyframeNativeFileset;
  }

  /**
   * Returns user-specified test environment variables and their values, as set by the --test_env
   * options.
   */
  @Deprecated
  @SkylarkCallable(
    name = "test_env",
    structField = true,
    doc =
        "A dictionary containing user-specified test environment variables and their values, "
            + "as set by the --test_env options. DO NOT USE! This is not the complete environment!"
  )
  public ImmutableMap<String, String> getTestEnv() {
    return testEnv.getFixedEnv();
  }

  /**
   * Returns user-specified test environment variables and their values, as set by the
   * {@code --test_env} options. It is incomplete in that it is not a superset of the
   * {@link #getActionEnvironment}, but both have to be applied, with this one being applied after
   * the other, such that {@code --test_env} settings can override {@code --action_env} settings.
   */
  // TODO(ulfjack): Just return the merged action and test action environment here?
  public ActionEnvironment getTestActionEnvironment() {
    return testEnv;
  }

  public int getMinParamFileSize() {
    return options.minParamFileSize;
  }

  @SkylarkCallable(name = "coverage_enabled", structField = true,
      doc = "A boolean that tells whether code coverage is enabled for this run. Note that this "
          + "does not compute whether a specific rule should be instrumented for code coverage "
          + "data collection. For that, see the <a href=\"ctx.html#coverage_instrumented\"><code>"
          + "ctx.coverage_instrumented</code></a> function.")
  public boolean isCodeCoverageEnabled() {
    return options.collectCodeCoverage;
  }

  public boolean isLLVMCoverageMapFormatEnabled() {
    return options.useLLVMCoverageMapFormat;
  }

  /** If false, AnalysisEnvironment doesn't register any actions created by the ConfiguredTarget. */
  public boolean isActionsEnabled() {
    return actionsEnabled;
  }

  public RunUnder getRunUnder() {
    return options.runUnder;
  }

  /**
   * Returns true if this is a host configuration.
   */
  public boolean isHostConfiguration() {
    return options.isHost;
  }

  public boolean checkVisibility() {
    return options.checkVisibility;
  }

  public boolean checkLicenses() {
    return options.checkLicenses;
  }

  public boolean enforceConstraints() {
    return options.enforceConstraints;
  }

  public List<Label> getActionListeners() {
    return isActionsEnabled() ? options.actionListeners : ImmutableList.<Label>of();
  }

  /**
   * Returns whether FileWriteAction may transparently compress its contents in the analysis phase
   * to save memory. Semantics are not affected.
   */
  public FileWriteAction.Compression transparentCompression() {
    return FileWriteAction.Compression.fromBoolean(options.transparentCompression);
  }

  /**
   * Returns whether we should trim configurations to only include the fragments needed to correctly
   * analyze a rule.
   */
  public boolean trimConfigurations() {
    return options.configsMode == Options.ConfigsMode.ON;
  }

  /**
   * Returns compilation mode.
   */
  public CompilationMode getCompilationMode() {
    return options.compilationMode;
  }

  /** Returns the cache key of the build options used to create this configuration. */
  public final String checksum() {
    return checksum;
  }

  /** Returns a copy of the build configuration options for this configuration. */
  public BuildOptions cloneOptions() {
    BuildOptions clone = buildOptions.clone();
    return clone;
  }

  /**
   * Returns the actual options reference used by this configuration.
   *
   * <p><b>Be very careful using this method.</b> Options classes are mutable - no caller
   * should ever call this method if there's any change the reference might be written to.
   * This method only exists because {@link #cloneOptions} can be expensive when applied to
   * every edge in a dependency graph.
   *
   * <p>Do not use this method without careful review with other Bazel developers.
   */
  public BuildOptions getOptions() {
    return buildOptions;
  }

  public String getCpu() {
    return options.cpu;
  }

  @VisibleForTesting
  public String getHostCpu() {
    return options.hostCpu;
  }

  public boolean runfilesEnabled() {
    switch (options.enableRunfiles) {
      case YES:
        return true;
      case NO:
        return false;
      default:
        return OS.getCurrent() != OS.WINDOWS;
    }
  }

  public boolean enableWindowsExeLauncher() {
    return options.windowsExeLauncher;
  }

  /**
   * Collects executables defined by fragments.
   */
  private PathFragment computeShellExecutable() {
    PathFragment result = null;

    for (Fragment fragment : fragments.values()) {
      if (fragment.getShellExecutable() != null) {
        Verify.verify(result == null);
        result = fragment.getShellExecutable();
      }
    }

    return result;
  }

  /**
   * Returns the transition that produces the "artifact owner" for this configuration, or null
   * if this configuration is its own owner.
   */
  @Nullable
  public PatchTransition getArtifactOwnerTransition() {
    PatchTransition ownerTransition = null;
    for (Fragment fragment : fragments.values()) {
      PatchTransition fragmentTransition = fragment.getArtifactOwnerTransition();
      if (fragmentTransition != null) {
        if (ownerTransition != null) {
          Verify.verify(ownerTransition == fragmentTransition,
              String.format(
                  "cannot determine owner transition: fragments returning both %s and %s",
                  ownerTransition.toString(), fragmentTransition.toString()));
        }
        ownerTransition = fragmentTransition;
      }
    }
    return ownerTransition;
  }

  /**
   * @return the list of default features used for all packages.
   */
  public List<String> getDefaultFeatures() {
    return options.defaultFeatures;
  }

  /**
   * Returns the "top-level" environment space, i.e. the set of environments all top-level
   * targets must be compatible with. An empty value implies no restrictions.
   */
  public List<Label> getTargetEnvironments() {
    return options.targetEnvironments;
  }

  /**
   * Returns the {@link Label} of the {@code environment_group} target that will be used to find the
   * target environment during auto-population.
   */
  public Label getAutoCpuEnvironmentGroup() {
    return options.autoCpuEnvironmentGroup;
  }

  public Class<? extends Fragment> getSkylarkFragmentByName(String name) {
    return skylarkVisibleFragments.get(name);
  }

  public ImmutableCollection<String> getSkylarkFragmentNames() {
    return skylarkVisibleFragments.keySet();
  }

  public ImmutableSet<String> getReservedActionMnemonics() {
    return reservedActionMnemonics;
  }

  /**
   * Returns an extra transition that should apply to top-level targets in this
   * configuration. Returns null if no transition is needed.
   */
  @Nullable
  public PatchTransition topLevelConfigurationHook(Target toTarget) {
    PatchTransition currentTransition = null;
    for (Fragment fragment : fragments.values()) {
      PatchTransition fragmentTransition = fragment.topLevelConfigurationHook(toTarget);
      if (fragmentTransition == null) {
        continue;
      } else if (currentTransition == null) {
        currentTransition = fragmentTransition;
      } else {
        currentTransition = new ComposingPatchTransition(currentTransition, fragmentTransition);
      }
    }
    return currentTransition;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.configurationId(checksum());
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    BuildEventStreamProtos.Configuration.Builder builder =
        BuildEventStreamProtos.Configuration.newBuilder()
            .setMnemonic(getMnemonic())
            .setPlatformName(getCpu())
            .putAllMakeVariable(getMakeEnvironment())
            .setCpu(getCpu());
    return GenericBuildEvent.protoChaining(this).setConfiguration(builder.build()).build();
  }
}
