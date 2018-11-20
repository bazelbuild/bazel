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
import com.google.common.base.Splitter;
import com.google.common.base.Suppliers;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.MutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.BuildConfigurationApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import java.util.ArrayList;
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
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Instances of BuildConfiguration represent a collection of context information which may affect a
 * build (for example: the target platform for compilation, or whether or not debug tables are
 * required). In fact, all "environmental" information (e.g. from the tool's command-line, as
 * opposed to the BUILD file) that can affect the output of any build tool should be explicitly
 * represented in the BuildConfiguration instance.
 *
 * <p>A single build may require building tools to run on a variety of platforms: when compiling a
 * server application for production, we must build the build tools (like compilers) to run on the
 * host platform, but cross-compile the application for the production environment.
 *
 * <p>There is always at least one BuildConfiguration instance in any build: the one representing
 * the host platform. Additional instances may be created, in a cross-compilation build, for
 * example.
 *
 * <p>Instances of BuildConfiguration are canonical:
 *
 * <pre>c1.equals(c2) <=> c1==c2.</pre>
 */
// TODO(janakr): If overhead of fragments class names is too high, add constructor that just takes
// fragments and gets names from them.
@AutoCodec
public class BuildConfiguration implements BuildConfigurationApi {
  /**
   * Sorts fragments by class name. This produces a stable order which, e.g., facilitates consistent
   * output from buildMnemonic.
   */
  @AutoCodec
  public static final Comparator<Class<? extends Fragment>> lexicalFragmentSorter =
      Comparator.comparing(Class::getName);

  private static final Interner<ImmutableSortedMap<Class<? extends Fragment>, Fragment>>
      fragmentsInterner = BlazeInterners.newWeakInterner();

  private static final Interner<ImmutableMap<String, String>>
      executionInfoInterner = BlazeInterners.newWeakInterner();

  /** Compute the default shell environment for actions from the command line options. */
  public interface ActionEnvironmentProvider {
    ActionEnvironment getActionEnvironment(BuildOptions options);
  }

  /**
   * An interface for language-specific configurations.
   *
   * <p>All implementations must be immutable and communicate this as clearly as possible (e.g.
   * declare {@link ImmutableList} signatures on their interfaces vs. {@link List}). This is because
   * fragment instances may be shared across configurations.
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
     * Returns a fragment of the output directory name for this configuration. The output
     * directory for the whole configuration contains all the short names by all fragments.
     */
    @Nullable
    public String getOutputDirectoryName() {
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
  }

  public static final Label convertOptionsLabel(String input) throws OptionsParsingException {
    try {
      // Check if the input starts with '/'. We don't check for "//" so that
      // we get a better error message if the user accidentally tries to use
      // an absolute path (starting with '/') for a label.
      if (!input.startsWith("/") && !input.startsWith("@")) {
        input = "//" + input;
      }
      return Label.parseAbsolute(input, ImmutableMap.of());
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
   * <p>(Note: any client that creates a view will also need to declare BuildView.Options, which
   * affect the <i>mechanism</i> of view construction, even if they don't affect the value of the
   * BuildConfiguration instances.)
   *
   * <p>IMPORTANT: when adding new options, be sure to consider whether those values should be
   * propagated to the host configuration or not.
   *
   * <p>ALSO IMPORTANT: all option types MUST define a toString method that gives identical results
   * for semantically identical option values. The simplest way to ensure that is to return the
   * input string.
   */
  public static class Options extends FragmentOptions implements Cloneable {
    public static final OptionDefinition CPU =
        OptionsParser.getOptionDefinitionByName(Options.class, "cpu");

    @Option(
      name = "experimental_separate_genfiles_directory",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Whether to have a separate genfiles directory or fold it into the bin directory"
    )
    public boolean separateGenfilesDirectory;

    @Option(
      name = "define",
      converter = Converters.AssignmentConverter.class,
      defaultValue = "",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Each --define option specifies an assignment for a build variable."
    )
    public List<Map.Entry<String, String>> commandLineBuildVariables;

    @Option(
      name = "cpu",
      defaultValue = "",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
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
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
      help =
          "If this option is enabled, filesets crossing package boundaries are reported "
              + "as errors. It does not work when check_fileset_dependencies_recursively is "
              + "disabled."
    )
    public boolean strictFilesets;

    @Option(
        name = "experimental_strict_fileset_output",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
        effectTags = {OptionEffectTag.EXECUTION},
        help =
            "If this option is enabled, filesets will treat all output artifacts as regular files. "
              + "They will not traverse directories or be sensitive to symlinks."
    )
    public boolean strictFilesetOutput;

    @Option(
      name = "stamp",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Stamp binaries with the date, username, hostname, workspace information, etc."
    )
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
              + "instrumented unless --instrument_test_targets is enabled."
    )
    public RegexFilter instrumentationFilter;

    @Option(
      name = "instrument_test_targets",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "When coverage is enabled, specifies whether to consider instrumenting test rules. "
              + "When set, test rules included by --instrumentation_filter are instrumented. "
              + "Otherwise, test rules are always excluded from coverage instrumentation."
    )
    public boolean instrumentTestTargets;

    @Option(
      name = "host_cpu",
      defaultValue = "",
      converter = AutoCpuConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "The host CPU."
    )
    public String hostCpu;

    @Option(
      name = "compilation_mode",
      abbrev = 'c',
      converter = CompilationMode.Converter.class,
      defaultValue = "fastbuild",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = { OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES },
      help = "Specify the mode the binary will be built in. Values: 'fastbuild', 'dbg', 'opt'."
    )
    public CompilationMode compilationMode;

    @Option(
        name = "host_compilation_mode",
        converter = CompilationMode.Converter.class,
        defaultValue = "opt",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = { OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES },
        help = "Specify the mode the tools used during the build will be built in. Values: "
            + "'fastbuild', 'dbg', 'opt'."
    )
    public CompilationMode hostCompilationMode;

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

    /**
     * This option is used by skylark transitions to add a disginguishing element to the output
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
      metadataTags = { OptionMetadataTag.INTERNAL }
    )
    public String transitionDirectoryNameFragment;

    @Option(
      name = "platform_suffix",
      defaultValue = "null",
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
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.TEST_RUNNER},
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
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
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
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If specified, Bazel will instrument code (using offline instrumentation where "
              + "possible) and will collect coverage information during tests. Only targets that "
              + " match --instrumentation_filter will be affected. Usually this option should "
              + " not be specified directly - 'bazel coverage' command should be used instead."
    )
    public boolean collectCodeCoverage;

    @Option(
      name = "experimental_java_coverage",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "If true Bazel will use a new way of computing code coverage for java targets."
    )
    public boolean experimentalJavaCoverage;

    @Option(
        name = "experimental_cc_coverage",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS
        },
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        help =
            "If specified, Bazel will use gcov to collect code coverage for C++ test targets. "
                + "This option only works for gcc compilation.")
    public boolean useGcovCoverage;

    @Option(
      name = "build_runfile_manifests",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "If true, write runfiles manifests for all targets.  " + "If false, omit them."
    )
    public boolean buildRunfilesManifests;

    @Option(
      name = "build_runfile_links",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, build runfiles symlink forests for all targets.  "
              + "If false, write only manifests when possible."
    )
    public boolean buildRunfiles;

    @Option(
      name = "legacy_external_runfiles",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, build runfiles symlink forests for external repositories under "
              + ".runfiles/wsname/external/repo (in addition to .runfiles/repo)."
    )
    public boolean legacyExternalRunfiles;

    @Option(
      name = "check_fileset_dependencies_recursively",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      deprecationWarning =
          "This flag is a no-op and fileset dependencies are always checked "
              + "to ensure correctness of builds.",
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
    )
    public boolean checkFilesetDependenciesRecursively;

    @Option(
      name = "experimental_skyframe_native_filesets",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      deprecationWarning = "This flag is a no-op and skyframe-native-filesets is always true."
    )
    public boolean skyframeNativeFileset;

    @Option(
      name = "run_under",
      defaultValue = "null",
      converter = RunUnderConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
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
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help = "If disabled, visibility errors are demoted to warnings."
    )
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
      converter = LabelListConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
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
        defaultValue = "500",
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
        defaultValue = "",
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
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "Declares this build's target environment. Must be a label reference to an "
              + "\"environment\" rule. If specified, all top-level targets must be "
              + "compatible with this environment."
    )
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
              + "target_environment values."
    )
    public Label autoCpuEnvironmentGroup;

    /** The source of make variables for this configuration. */
    public enum MakeVariableSource {
      CONFIGURATION,
      TOOLCHAIN
    }

    /** Converter for --make_variables_source. */
    public static class MakeVariableSourceConverter extends EnumConverter<MakeVariableSource> {
      public MakeVariableSourceConverter() {
        super(MakeVariableSource.class, "Make variable source");
      }
    }

    @Option(
      name = "make_variables_source",
      converter = MakeVariableSourceConverter.class,
      defaultValue = "configuration",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN}
    )
    public MakeVariableSource makeVariableSource;

    /** Values for --experimental_dynamic_configs. */
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
                + "support execution info, e.g. Genrule, CppCompile, Javac, SkylarkAction, "
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

    @Override
    public FragmentOptions getHost() {
      Options host = (Options) getDefault();

      host.outputDirectoryName = "host";
      host.compilationMode = hostCompilationMode;
      host.isHost = true;
      host.configsMode = configsMode;
      host.enableRunfiles = enableRunfiles;
      host.executionInfoModifier = executionInfoModifier;
      host.commandLineBuildVariables = commandLineBuildVariables;
      host.enforceConstraints = enforceConstraints;
      host.separateGenfilesDirectory = separateGenfilesDirectory;
      host.cpu = hostCpu;

      // === Runfiles ===
      host.buildRunfilesManifests = buildRunfilesManifests;
      host.buildRunfiles = buildRunfiles;

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

      return host;
    }


  }

  private final String checksum;

  private final ImmutableSortedMap<Class<? extends Fragment>, Fragment> fragments;
  private final FragmentClassSet fragmentClassSet;

  private final ImmutableMap<String, Class<? extends Fragment>> skylarkVisibleFragments;
  private final RepositoryName mainRepositoryName;
  private final ImmutableSet<String> reservedActionMnemonics;
  private CommandLineLimits commandLineLimits;

  /**
   * Directories in the output tree.
   *
   * <p>The computation of the output directory should be a non-injective mapping from
   * BuildConfiguration instances to strings. The result should identify the aspects of the
   * configuration that should be reflected in the output file names. Furthermore the returned
   * string must not contain shell metacharacters.
   *
   * <p>For configuration settings which are NOT part of the output directory name, rebuilding with
   * a different value of such a setting will build in the same output directory. This means that
   * any actions whose keys (see Action.getKey()) have changed will be rerun. That may result in a
   * lot of recompilation.
   *
   * <p>For configuration settings which ARE part of the output directory name, rebuilding with a
   * different value of such a setting will rebuild in a different output directory; this will
   * result in higher disk usage and more work the <i>first</i> time you rebuild with a different
   * setting, but will result in less work if you regularly switch back and forth between different
   * settings.
   *
   * <p>With one important exception, it's sound to choose any subset of the config's components for
   * this string, it just alters the dimensionality of the cache. In other words, it's a trade-off
   * on the "injectiveness" scale: at one extreme (output directory name contains all data in the
   * config, and is thus injective) you get extremely precise caching (no competition for the same
   * output-file locations) but you have to rebuild for even the slightest change in configuration.
   * At the other extreme (the output (directory name is a constant) you have very high competition
   * for output-file locations, but if a slight change in configuration doesn't affect a particular
   * build step, you're guaranteed not to have to rebuild it. The important exception has to do with
   * multiple configurations: every configuration in the build must have a different output
   * directory name so that their artifacts do not conflict.
   *
   * <p>The host configuration is special-cased: in order to guarantee that its output directory is
   * always separate from that of the target configuration, we simply pin it to "host". We do this
   * so that the build works even if the two configurations are too close (which is common) and so
   * that the path of artifacts in the host configuration is a bit more readable.
   */
  @AutoCodec.VisibleForSerialization
  public enum OutputDirectory {
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

    @AutoCodec.VisibleForSerialization
    public ArtifactRoot getRoot(
        String outputDirName, BlazeDirectories directories, RepositoryName mainRepositoryName) {
      // e.g., execroot/repo1
      Path execRoot = directories.getExecRoot(mainRepositoryName.strippedName());
      // e.g., execroot/repo1/bazel-out/config/bin
      Path outputDir = execRoot.getRelative(directories.getRelativeOutputPath())
          .getRelative(outputDirName);
      if (middleman) {
        return ArtifactRoot.middlemanRoot(execRoot, outputDir);
      }
      // e.g., [[execroot/repo1]/bazel-out/config/bin]
      return ArtifactRoot.asDerivedRoot(execRoot, outputDir.getRelative(nameFragment));
    }
  }

  private final BlazeDirectories directories;
  private final String outputDirName;

  // We precompute the roots for the main repository, since that's the common case.
  private final ArtifactRoot outputDirectoryForMainRepository;
  private final ArtifactRoot binDirectoryForMainRepository;
  private final ArtifactRoot includeDirectoryForMainRepository;
  private final ArtifactRoot genfilesDirectoryForMainRepository;
  private final ArtifactRoot coverageDirectoryForMainRepository;
  private final ArtifactRoot testlogsDirectoryForMainRepository;
  private final ArtifactRoot middlemanDirectoryForMainRepository;

  private final boolean separateGenfilesDirectory;

  /**
   * The global "make variables" such as "$(TARGET_CPU)"; these get applied to all rules analyzed in
   * this configuration.
   */
  private final ImmutableMap<String, String> globalMakeEnv;

  private final ActionEnvironment actionEnv;
  private final ActionEnvironment testEnv;

  private final BuildOptions buildOptions;
  private final BuildOptions.OptionsDiffForReconstruction buildOptionsDiff;
  private final Options options;

  private final String mnemonic;

  private final ImmutableMap<String, String> commandLineBuildVariables;

  private final int hashCode; // We can precompute the hash code as all its inputs are immutable.

  /** Data for introspecting the options used by this configuration. */
  private final TransitiveOptionDetails transitiveOptionDetails;

  private final Supplier<BuildConfigurationEvent> buildEventSupplier;

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
                && fragments.values().containsAll(other.fragments.values())
                && buildOptions.getNativeOptions().containsAll(other.buildOptions.getNativeOptions()));
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
    return fragments.values().equals(otherConfig.fragments.values())
        && buildOptions.equals(otherConfig.buildOptions);
  }

  private int computeHashCode() {
    return Objects.hash(fragments, buildOptions.getNativeOptions());
  }

  public void describe(StringBuilder sb) {
    for (Fragment fragment : fragments.values()) {
      sb.append(fragment.getClass().getName()).append('\n');
    }
    for (String s : buildOptions.toString().split(" ")) {
      sb.append(s).append('\n');
    }
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  /** Returns map of all the fragments for this configuration. */
  public ImmutableMap<Class<? extends Fragment>, Fragment> getFragmentsMap() {
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

  private static ImmutableSortedMap<Class<? extends Fragment>, Fragment> makeFragmentsMap(
      Map<Class<? extends Fragment>, Fragment> fragmentsMap) {
    return fragmentsInterner.intern(ImmutableSortedMap.copyOf(fragmentsMap, lexicalFragmentSorter));
  }

  /** Constructs a new BuildConfiguration instance. */
  public BuildConfiguration(
      BlazeDirectories directories,
      Map<Class<? extends Fragment>, Fragment> fragmentsMap,
      BuildOptions buildOptions,
      BuildOptions.OptionsDiffForReconstruction buildOptionsDiff,
      ImmutableSet<String> reservedActionMnemonics,
      ActionEnvironment actionEnvironment,
      String repositoryName) {
    this(
        directories,
        fragmentsMap,
        buildOptions,
        buildOptionsDiff,
        reservedActionMnemonics,
        actionEnvironment,
        RepositoryName.createFromValidStrippedName(repositoryName));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  BuildConfiguration(
      BlazeDirectories directories,
      Map<Class<? extends Fragment>, Fragment> fragmentsMap,
      BuildOptions buildOptions,
      BuildOptions.OptionsDiffForReconstruction buildOptionsDiff,
      ImmutableSet<String> reservedActionMnemonics,
      ActionEnvironment actionEnvironment,
      RepositoryName mainRepositoryName) {
    this.directories = directories;
    this.fragments = makeFragmentsMap(fragmentsMap);
    this.fragmentClassSet = FragmentClassSet.of(this.fragments.keySet());

    this.skylarkVisibleFragments = buildIndexOfSkylarkVisibleFragments();
    this.buildOptions = buildOptions.clone();
    this.buildOptionsDiff = buildOptionsDiff;
    this.options = buildOptions.get(Options.class);
    this.separateGenfilesDirectory = options.separateGenfilesDirectory;
    this.mainRepositoryName = mainRepositoryName;

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
        OutputDirectory.OUTPUT.getRoot(outputDirName, directories, mainRepositoryName);
    this.binDirectoryForMainRepository =
        OutputDirectory.BIN.getRoot(outputDirName, directories, mainRepositoryName);
    this.includeDirectoryForMainRepository =
        OutputDirectory.INCLUDE.getRoot(outputDirName, directories, mainRepositoryName);
    this.genfilesDirectoryForMainRepository =
        OutputDirectory.GENFILES.getRoot(outputDirName, directories, mainRepositoryName);
    this.coverageDirectoryForMainRepository =
        OutputDirectory.COVERAGE.getRoot(outputDirName, directories, mainRepositoryName);
    this.testlogsDirectoryForMainRepository =
        OutputDirectory.TESTLOGS.getRoot(outputDirName, directories, mainRepositoryName);
    this.middlemanDirectoryForMainRepository =
        OutputDirectory.MIDDLEMAN.getRoot(outputDirName, directories, mainRepositoryName);

    this.actionEnv = actionEnvironment;

    this.testEnv = setupTestEnvironment();

    this.transitiveOptionDetails =
        TransitiveOptionDetails.forOptionsWithDefaults(buildOptions.getNativeOptions());

    ImmutableMap.Builder<String, String> globalMakeEnvBuilder = ImmutableMap.builder();

    // TODO(configurability-team): Deprecate TARGET_CPU in favor of platforms.
    globalMakeEnvBuilder.put("TARGET_CPU", options.cpu);

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

    checksum = buildOptions.computeChecksum();
    hashCode = computeHashCode();

    this.reservedActionMnemonics = reservedActionMnemonics;
    this.buildEventSupplier = Suppliers.memoize(this::createBuildEvent);
    this.commandLineLimits = new CommandLineLimits(options.minParamFileSize);
  }

  /**
   * Returns a copy of this configuration only including the given fragments (which the current
   * configuration is assumed to have).
   */
  public BuildConfiguration clone(
      FragmentClassSet fragmentClasses,
      RuleClassProvider ruleClassProvider,
      BuildOptions defaultBuildOptions) {

    ClassToInstanceMap<Fragment> fragmentsMap = MutableClassToInstanceMap.create();
    for (Fragment fragment : fragments.values()) {
      if (fragmentClasses.fragmentClasses().contains(fragment.getClass())) {
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
            BuildOptions.diffForReconstruction(defaultBuildOptions, options),
            reservedActionMnemonics,
            actionEnv,
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
      SkylarkModule module = SkylarkInterfaceUtils.getSkylarkModule(fragmentClass);
      if (module != null) {
        builder.put(module.name(), fragmentClass);
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

  private String buildMnemonic() {
    // See explanation at declaration for outputRoots.
    String platformSuffix = (options.platformSuffix != null) ? options.platformSuffix : "";
    ArrayList<String> nameParts = new ArrayList<>();
    for (Fragment fragment : fragments.values()) {
      nameParts.add(fragment.getOutputDirectoryName());
    }
    nameParts.add(getCompilationMode() + platformSuffix);
    if (options.transitionDirectoryNameFragment != null) {
      nameParts.add(options.transitionDirectoryNameFragment);
    }
    return Joiner.on('-').skipNulls().join(nameParts);
  }

  /** Returns the output directory for this build configuration. */
  public ArtifactRoot getOutputDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? outputDirectoryForMainRepository
        : OutputDirectory.OUTPUT.getRoot(outputDirName, directories, mainRepositoryName);
  }

  @Override
  public ArtifactRoot getBinDir() {
    return getBinDirectory(RepositoryName.MAIN);
  }

  /** Returns the bin directory for this build configuration. */
  public ArtifactRoot getBinDirectory() {
    return getBinDirectory(RepositoryName.MAIN);
  }

  /**
   * TODO(kchodorow): This (and the other get*Directory functions) won't work with external
   * repositories without changes to how ArtifactFactory resolves derived roots. This is not an
   * issue right now because it only effects Blaze's include scanning (internal) and Bazel's
   * repositories (external) but will need to be fixed.
   */
  public ArtifactRoot getBinDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? binDirectoryForMainRepository
        : OutputDirectory.BIN.getRoot(outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns a relative path to the bin directory at execution time.
   */
  public PathFragment getBinFragment() {
    return getBinDirectory().getExecPath();
  }

  /** Returns the include directory for this build configuration. */
  public ArtifactRoot getIncludeDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? includeDirectoryForMainRepository
        : OutputDirectory.INCLUDE.getRoot(outputDirName, directories, mainRepositoryName);
  }

  @Override
  public ArtifactRoot getGenfilesDir() {
    return getGenfilesDirectory(RepositoryName.MAIN);
  }

  /** Returns the genfiles directory for this build configuration. */
  public ArtifactRoot getGenfilesDirectory() {
    return getGenfilesDirectory(RepositoryName.MAIN);
  }

  public ArtifactRoot getGenfilesDirectory(RepositoryName repositoryName) {
    if (!separateGenfilesDirectory) {
      return getBinDirectory(repositoryName);
    }

    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? genfilesDirectoryForMainRepository
        : OutputDirectory.GENFILES.getRoot(outputDirName, directories, mainRepositoryName);
  }

  /**
   * Returns the directory where coverage-related artifacts and metadata files should be stored.
   * This includes for example uninstrumented class files needed for Jacoco's coverage reporting
   * tools.
   */
  public ArtifactRoot getCoverageMetadataDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? coverageDirectoryForMainRepository
        : OutputDirectory.COVERAGE.getRoot(outputDirName, directories, mainRepositoryName);
  }

  /** Returns the testlogs directory for this build configuration. */
  public ArtifactRoot getTestLogsDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? testlogsDirectoryForMainRepository
        : OutputDirectory.TESTLOGS.getRoot(outputDirName, directories, mainRepositoryName);
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
  @Override
  public String getHostPathSeparator() {
    // TODO(bazel-team): Maybe do this in the constructor instead? This isn't serialization-safe.
    return OS.getCurrent() == OS.WINDOWS ? ";" : ":";
  }

  /** Returns the internal directory (used for middlemen) for this build configuration. */
  public ArtifactRoot getMiddlemanDirectory(RepositoryName repositoryName) {
    return repositoryName.isMain() || repositoryName.equals(mainRepositoryName)
        ? middlemanDirectoryForMainRepository
        : OutputDirectory.MIDDLEMAN.getRoot(outputDirName, directories, mainRepositoryName);
  }

  public boolean isStrictFilesets() {
    return options.strictFilesets;
  }

  public boolean isStrictFilesetOutput() {
    return options.strictFilesetOutput;
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
  @Override
  public ImmutableMap<String, String> getLocalShellEnvironment() {
    return actionEnv.getFixedEnv().toMap();
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
  public Iterable<String> getVariableShellEnvironment() {
    return actionEnv.getInheritedEnv();
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

  /** Which fragments does this configuration contain? */
  public FragmentClassSet fragmentClasses() {
    return fragmentClassSet;
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

  /**
   * Returns user-specified test environment variables and their values, as set by the --test_env
   * options.
   */
  @Override
  public ImmutableMap<String, String> getTestEnv() {
    return testEnv.getFixedEnv().toMap();
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

  public CommandLineLimits getCommandLineLimits() {
    return commandLineLimits;
  }

  @Override
  public boolean isCodeCoverageEnabled() {
    return options.collectCodeCoverage;
  }

  public boolean isExperimentalJavaCoverage() {
    return options.experimentalJavaCoverage;
  }

  public boolean useGcovCoverage() {
    return options.useGcovCoverage;
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

  public boolean allowAnalysisFailures() {
    return options.allowAnalysisFailures;
  }

  public boolean evaluatingForAnalysisTest() {
    return options.evaluatingForAnalysisTest;
  }

  public int analysisTestingDepsLimit() {
    return options.analysisTestingDepsLimit;
  }

  public List<Label> getActionListeners() {
    return options.actionListeners;
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
  public String checksum() {
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

  public BuildOptions.OptionsDiffForReconstruction getBuildOptionsDiff() {
    return buildOptionsDiff;
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

  /**
   * Returns a modified copy of {@code executionInfo} if any {@code executionInfoModifiers} apply to
   * the given {@code mnemonic}. Otherwise returns {@code executionInfo} unchanged.
   */
  public ImmutableMap<String, String> modifiedExecutionInfo(
      ImmutableMap<String, String> executionInfo, String mnemonic) {
    if (!options.executionInfoModifier.matches(mnemonic)) {
      return executionInfo;
    }
    LinkedHashMap<String, String> mutableCopy = new LinkedHashMap<>(executionInfo);
    modifyExecutionInfo(mutableCopy, mnemonic);
    return executionInfoInterner.intern(ImmutableMap.copyOf(mutableCopy));
  }

  /** Applies {@code executionInfoModifiers} to the given {@code executionInfo}. */
  public void modifyExecutionInfo(Map<String, String> executionInfo, String mnemonic) {
    options.executionInfoModifier.apply(mnemonic, executionInfo);
  }

  /** @return the list of default features used for all packages. */
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

  public BuildEventId getEventId() {
    return BuildEventId.configurationId(checksum());
  }

  public BuildConfigurationEvent toBuildEvent() {
    return buildEventSupplier.get();
  }

  private BuildConfigurationEvent createBuildEvent() {
    BuildEventId eventId = getEventId();
    BuildEventStreamProtos.BuildEvent.Builder builder =
        BuildEventStreamProtos.BuildEvent.newBuilder();
    builder
        .setId(eventId.asStreamProto())
        .setConfiguration(
            BuildEventStreamProtos.Configuration.newBuilder()
                .setMnemonic(getMnemonic())
                .setPlatformName(getCpu())
                .putAllMakeVariable(getMakeEnvironment())
                .setCpu(getCpu())
                .build());
    return new BuildConfigurationEvent(eventId, builder.build());
  }

  public ImmutableSet<String> getReservedActionMnemonics() {
    return reservedActionMnemonics;
  }

}
