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
package com.google.devtools.build.lib.buildtool;

import com.google.common.base.Optional;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.runtime.BlazeCommandEventHandler;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.regex.Pattern;

/**
 * A BuildRequest represents a single invocation of the build tool by a user.
 * A request specifies a list of targets to be built for a single
 * configuration, a pair of output/error streams, and additional options such
 * as --keep_going, --jobs, etc.
 */
public class BuildRequest implements OptionsClassProvider {
  /**
   * Options interface--can be used to parse command-line arguments.
   *
   * <p>See also ExecutionOptions; from the user's point of view, there's no
   * qualitative difference between these two sets of options.
   */
  public static class BuildRequestOptions extends OptionsBase {

    /* "Execution": options related to the execution of a build: */

    @Option(name = "jobs",
            abbrev = 'j',
            defaultValue = "200",
            category = "strategy",
            help = "The number of concurrent jobs to run. "
                + "0 means build sequentially. Values above " + MAX_JOBS
                + " are not allowed.")
    public int jobs;

    @Option(name = "progress_report_interval",
            defaultValue = "0",
            category = "verbosity",
            converter = ProgressReportIntervalConverter.class,
            help = "The number of seconds to wait between two reports on"
                + " still running jobs.  The default value 0 means to use"
                + " the default 10:30:60 incremental algorithm.")
    public int progressReportInterval;

    @Option(name = "explain",
            defaultValue = "null",
            category = "verbosity",
            converter = OptionsUtils.PathFragmentConverter.class,
            help = "Causes the build system to explain each executed step of the "
            + "build. The explanation is written to the specified log file.")
    public PathFragment explanationPath;

    @Option(name = "verbose_explanations",
            defaultValue = "false",
            category = "verbosity",
            help = "Increases the verbosity of the explanations issued if --explain is enabled. "
            + "Has no effect if --explain is not enabled.")
    public boolean verboseExplanations;

    @Option(name = "output_filter",
        converter = Converters.RegexPatternConverter.class,
        defaultValue = "null",
        category = "flags",
        help = "Only shows warnings for rules with a name matching the provided regular "
            + "expression.")
    public Pattern outputFilter;

    @Deprecated
    @Option(name = "dump_makefile",
            defaultValue = "false",
            category = "undocumented",
            help = "this flag has no effect.")
    public boolean dumpMakefile;

    @Deprecated
    @Option(name = "dump_action_graph",
        defaultValue = "false",
        category = "undocumented",
        help = "this flag has no effect.")

    public boolean dumpActionGraph;

    @Deprecated
    @Option(name = "dump_action_graph_for_package",
        allowMultiple = true,
        defaultValue = "",
        category = "undocumented",
        help = "this flag has no effect.")
    public List<String> dumpActionGraphForPackage = new ArrayList<>();

    @Deprecated
    @Option(name = "dump_action_graph_with_middlemen",
        defaultValue = "true",
        category = "undocumented",
        help = "this flag has no effect.")
    public boolean dumpActionGraphWithMiddlemen;

    @Deprecated
    @Option(name = "dump_providers",
        defaultValue = "false",
        category = "undocumented",
        help = "This is a no-op.")
    public boolean dumpProviders;

    @Deprecated
    @Option(name = "dump_targets",
            defaultValue = "null",
            category = "undocumented",
        help = "this flag has no effect.")
    public String dumpTargets;

    @Deprecated
    @Option(name = "dump_host_deps",
        defaultValue = "true",
        category = "undocumented",
        help = "Deprecated")
    public boolean dumpHostDeps;

    @Deprecated
    @Option(name = "dump_to_stdout",
        defaultValue = "false",
        category = "undocumented",
        help = "Deprecated")
    public boolean dumpToStdout;

    @Option(name = "analyze",
            defaultValue = "true",
            category = "undocumented",
            help = "Execute the analysis phase; this is the usual behaviour. "
                + "Specifying --noanalyze causes the build to stop before starting the "
                + "analysis phase, returning zero iff the package loading completed "
                + "successfully; this mode is useful for testing.")
    public boolean performAnalysisPhase;

    @Option(name = "build",
            defaultValue = "true",
            category = "what",
            help = "Execute the build; this is the usual behaviour. "
            + "Specifying --nobuild causes the build to stop before executing the "
            + "build actions, returning zero iff the package loading and analysis "
            + "phases completed successfully; this mode is useful for testing "
            + "those phases.")
    public boolean performExecutionPhase;

    @Option(name = "output_groups",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        allowMultiple = true,
        defaultValue = "",
        category = "undocumented",
        help = "Specifies which output groups of the top-level targets to build. "
            + "If omitted, a default set of output groups are built."
            + "When specified the default set is overridden."
            + "However you may use --output_groups=+<output_group> "
            + "or --output_groups=-<output_group> "
            + "to instead modify the set of output groups.")
    public List<String> outputGroups;

    @Option(name = "show_result",
            defaultValue = "1",
            category = "verbosity",
            help = "Show the results of the build.  For each "
            + "target, state whether or not it was brought up-to-date, and if "
            + "so, a list of output files that were built.  The printed files "
            + "are convenient strings for copy+pasting to the shell, to "
            + "execute them.\n"
            + "This option requires an integer argument, which "
            + "is the threshold number of targets above which result "
            + "information is not printed. "
            + "Thus zero causes suppression of the message and MAX_INT "
            + "causes printing of the result to occur always.  The default is one.")
    public int maxResultTargets;

    @Option(name = "experimental_show_artifacts",
        defaultValue = "false",
        category = "undocumented",
        help = "Output a list of all top level artifacts produced by this build."
            + "Use output format suitable for tool consumption. "
            + "This flag is temporary and intended to facilitate Android Studio integration. "
            + "This output format will likely change in the future or disappear completely."
    )
    public boolean showArtifacts;

    @Option(name = "announce",
            defaultValue = "false",
            category = "verbosity",
            help = "Deprecated. No-op.",
            deprecationWarning = "This option is now deprecated and is a no-op")
    public boolean announce;

    @Option(name = "symlink_prefix",
        defaultValue = "null",
        category = "misc",
        help = "The prefix that is prepended to any of the convenience symlinks that are created "
            + "after a build. If '/' is passed, then no symlinks are created and no warning is "
            + "emitted. If omitted, the default value is the name of the build tool."
        )
    public String symlinkPrefix;

    @Option(name = "experimental_multi_cpu",
            converter = Converters.CommaSeparatedOptionListConverter.class,
            allowMultiple = true,
            defaultValue = "",
            category = "semantics",
            help = "This flag allows specifying multiple target CPUs. If this is specified, "
                + "the --cpu option is ignored.")
    public List<String> multiCpus;

    @Option(name = "experimental_output_tree_tracking",
            defaultValue = "false",
            category = "undocumented",
            help = "If set, tell the output service (if any) to track when files in the output "
                + "tree have been modified externally (not by the build system). "
                + "This should improve incremental build speed when an appropriate output service "
                + "is enabled.")
    public boolean finalizeActions;

    @Option(
      name = "aspects",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      category = "undocumented", // for now
      help = "List of top-level aspects"
    )
    public List<String> aspects;

    public String getSymlinkPrefix(String productName) {
      return symlinkPrefix == null ? productName + "-" : symlinkPrefix;
    }
  }

  /**
   * Converter for progress_report_interval: [0, 3600].
   */
  public static class ProgressReportIntervalConverter extends RangeConverter {
    public ProgressReportIntervalConverter() {
      super(0, 3600);
    }
  }

  private static final int MAX_JOBS = 2000;
  private static final int JOBS_TOO_HIGH_WARNING = 1000;

  private final UUID id;
  private final LoadingCache<Class<? extends OptionsBase>, Optional<OptionsBase>> optionsCache;

  /** A human-readable description of all the non-default option settings. */
  private final String optionsDescription;

  /**
   * The name of the Blaze command that the user invoked.
   * Used for --announce.
   */
  private final String commandName;

  private final OutErr outErr;
  private final List<String> targets;

  private long startTimeMillis = 0; // milliseconds since UNIX epoch.

  private boolean runningInEmacs = false;
  private boolean runTests = false;

  private static final List<Class<? extends OptionsBase>> MANDATORY_OPTIONS = ImmutableList.of(
          BuildRequestOptions.class,
          PackageCacheOptions.class,
          LoadingOptions.class,
          BuildView.Options.class,
          ExecutionOptions.class);

  private BuildRequest(String commandName,
                       final OptionsProvider options,
                       final OptionsProvider startupOptions,
                       List<String> targets,
                       OutErr outErr,
                       UUID id,
                       long startTimeMillis) {
    this.commandName = commandName;
    this.optionsDescription = OptionsUtils.asShellEscapedString(options);
    this.outErr = outErr;
    this.targets = targets;
    this.id = id;
    this.startTimeMillis = startTimeMillis;
    this.optionsCache = CacheBuilder.newBuilder()
        .build(new CacheLoader<Class<? extends OptionsBase>, Optional<OptionsBase>>() {
          @Override
          public Optional<OptionsBase> load(Class<? extends OptionsBase> key) throws Exception {
            OptionsBase result = options.getOptions(key);
            if (result == null && startupOptions != null) {
              result = startupOptions.getOptions(key);
            }

            return Optional.fromNullable(result);
          }
        });

    for (Class<? extends OptionsBase> optionsClass : MANDATORY_OPTIONS) {
      Preconditions.checkNotNull(getOptions(optionsClass));
    }
  }

  /**
   * Returns a unique identifier that universally identifies this build.
   */
  public UUID getId() {
    return id;
  }

  /**
   * Returns the name of the Blaze command that the user invoked.
   */
  public String getCommandName() {
    return commandName;
  }

  /**
   * Set to true if this build request was initiated by Emacs.
   * (Certain output formatting may be necessary.)
   */
  public void setRunningInEmacs() {
    runningInEmacs = true;
  }

  boolean isRunningInEmacs() {
    return runningInEmacs;
  }

  /**
   * Enables test execution for this build request.
   */
  public void setRunTests() {
    runTests = true;
  }

  /**
   * Returns true if tests should be run by the build tool.
   */
  public boolean shouldRunTests() {
    return runTests;
  }

  /**
   * Returns the (immutable) list of targets to build in commandline
   * form.
   */
  public List<String> getTargets() {
    return targets;
  }

  /**
   * Returns the output/error streams to which errors and progress messages
   * should be sent during the fulfillment of this request.
   */
  public OutErr getOutErr() {
    return outErr;
  }

  @Override
  @SuppressWarnings("unchecked")
  public <T extends OptionsBase> T getOptions(Class<T> clazz) {
    try {
      return (T) optionsCache.get(clazz).orNull();
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the set of command-line options specified for this request.
   */
  public BuildRequestOptions getBuildOptions() {
    return getOptions(BuildRequestOptions.class);
  }

  /**
   * Returns the set of options related to the loading phase.
   */
  public PackageCacheOptions getPackageCacheOptions() {
    return getOptions(PackageCacheOptions.class);
  }

  /**
   * Returns the set of options related to the loading phase.
   */
  public LoadingOptions getLoadingOptions() {
    return getOptions(LoadingOptions.class);
  }

  /**
   * Returns the set of command-line options related to the view specified for
   * this request.
   */
  public BuildView.Options getViewOptions() {
    return getOptions(BuildView.Options.class);
  }

  /**
   * Returns the set of execution options specified for this request.
   */
  public ExecutionOptions getExecutionOptions() {
    return getOptions(ExecutionOptions.class);
  }

  /**
   * Returns the human-readable description of the non-default options
   * for this build request.
   */
  public String getOptionsDescription() {
    return optionsDescription;
  }

  /**
   * Return the time (according to System.currentTimeMillis()) at which the
   * service of this request was started.
   */
  public long getStartTime() {
    return startTimeMillis;
  }

  /**
   * Validates the options for this BuildRequest.
   *
   * <p>Issues warnings or throws {@code InvalidConfigurationException} for option settings that
   * conflict.
   *
   * @return list of warnings
   */
  public List<String> validateOptions() throws InvalidConfigurationException {
    List<String> warnings = new ArrayList<>();
    // Validate "jobs".
    int jobs = getBuildOptions().jobs;
    if (jobs < 0 || jobs > MAX_JOBS) {
      throw new InvalidConfigurationException(String.format(
          "Invalid parameter for --jobs: %d. Only values 0 <= jobs <= %d are allowed.", jobs,
          MAX_JOBS));
    }
    if (jobs > JOBS_TOO_HIGH_WARNING) {
      warnings.add(
          String.format("High value for --jobs: %d. You may run into memory issues", jobs));
    }

    int localTestJobs = getExecutionOptions().localTestJobs;
    if (localTestJobs < 0) {
      throw new InvalidConfigurationException(String.format(
          "Invalid parameter for --local_test_jobs: %d. Only values 0 or greater are "
              + "allowed.", localTestJobs));
    }
    if (localTestJobs > jobs) {
      warnings.add(
          String.format("High value for --local_test_jobs: %d. This exceeds the value for --jobs: "
              + "%d. Only up to %d local tests will run concurrently.", localTestJobs, jobs, jobs));
    }

    // Validate other BuildRequest options.
    if (getBuildOptions().verboseExplanations && getBuildOptions().explanationPath == null) {
      warnings.add("--verbose_explanations has no effect when --explain=<file> is not enabled");
    }

    return warnings;
  }

  /** Creates a new TopLevelArtifactContext from this build request. */
  public TopLevelArtifactContext getTopLevelArtifactContext() {
    return new TopLevelArtifactContext(
        getOptions(ExecutionOptions.class).testStrategy.equals("exclusive"),
        determineOutputGroups());
  }

  private ImmutableSortedSet<String> determineOutputGroups() {
    Set<String> current = Sets.newHashSet();

    boolean overridesDefaultOutputGroups = false;
    for (String outputGroup : getBuildOptions().outputGroups) {
      overridesDefaultOutputGroups |= !(outputGroup.startsWith("+") || outputGroup.startsWith("-"));
    }
    if (!overridesDefaultOutputGroups) {
      current.addAll(OutputGroupProvider.DEFAULT_GROUPS);
    }

    for (String outputGroup : getBuildOptions().outputGroups) {
      if (outputGroup.startsWith("+")) {
        current.add(outputGroup.substring(1));
      } else if (outputGroup.startsWith("-")) {
        current.remove(outputGroup.substring(1));
      } else {
        current.add(outputGroup);
      }
    }

    return ImmutableSortedSet.copyOf(current);
  }

  public ImmutableSortedSet<String> getMultiCpus() {
    return ImmutableSortedSet.copyOf(getBuildOptions().multiCpus);
  }

  public ImmutableList<String> getAspects() {
    return ImmutableList.copyOf(getBuildOptions().aspects);
  }

  public static BuildRequest create(String commandName, OptionsProvider options,
      OptionsProvider startupOptions,
      List<String> targets, OutErr outErr, UUID commandId, long commandStartTime) {

    BuildRequest request = new BuildRequest(commandName, options, startupOptions, targets, outErr,
        commandId, commandStartTime);

    // All this, just to pass a global boolean from the client to the server. :(
    if (options.getOptions(BlazeCommandEventHandler.Options.class).runningInEmacs) {
      request.setRunningInEmacs();
    }

    return request;
  }

}
