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
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.UiOptions;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.OptionsProvider;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

/**
 * A BuildRequest represents a single invocation of the build tool by a user.
 * A request specifies a list of targets to be built for a single
 * configuration, a pair of output/error streams, and additional options such
 * as --keep_going, --jobs, etc.
 */
public class BuildRequest implements OptionsProvider {
  private static final ImmutableList<Class<? extends OptionsBase>> MANDATORY_OPTIONS =
      ImmutableList.of(
          BuildRequestOptions.class,
          PackageOptions.class,
          BuildLanguageOptions.class,
          LoadingOptions.class,
          AnalysisOptions.class,
          ExecutionOptions.class,
          KeepGoingOption.class,
          LoadingPhaseThreadsOption.class);

  /** Returns a new Builder instance. */
  public static Builder builder() {
    return new Builder();
  }

  /** A Builder class to help create instances of BuildRequest. */
  public static final class Builder {
    private UUID id;
    private OptionsParsingResult options;
    private OptionsParsingResult startupOptions;
    private String commandName;
    private OutErr outErr;
    private List<String> targets;
    private long startTimeMillis; // milliseconds since UNIX epoch.
    private boolean needsInstrumentationFilter;
    private boolean runTests;
    private boolean checkForActionConflicts = true;

    private Builder() {}

    public Builder setId(UUID id) {
      this.id = id;
      return this;
    }

    public Builder setOptions(OptionsParsingResult options) {
      this.options = options;
      return this;
    }

    public Builder setStartupOptions(OptionsParsingResult startupOptions) {
      this.startupOptions = startupOptions;
      return this;
    }

    public Builder setCommandName(String commandName) {
      this.commandName = commandName;
      return this;
    }

    public Builder setOutErr(OutErr outErr) {
      this.outErr = outErr;
      return this;
    }

    public Builder setTargets(List<String> targets) {
      this.targets = targets;
      return this;
    }

    public Builder setStartTimeMillis(long startTimeMillis) {
      this.startTimeMillis = startTimeMillis;
      return this;
    }

    public Builder setNeedsInstrumentationFilter(boolean needsInstrumentationFilter) {
      this.needsInstrumentationFilter = needsInstrumentationFilter;
      return this;
    }

    public Builder setRunTests(boolean runTests) {
      this.runTests = runTests;
      return this;
    }

    public Builder setCheckforActionConflicts(boolean checkForActionConflicts) {
      this.checkForActionConflicts = checkForActionConflicts;
      return this;
    }

    public BuildRequest build() {
      return new BuildRequest(
          commandName,
          options,
          startupOptions,
          targets,
          outErr,
          id,
          startTimeMillis,
          needsInstrumentationFilter,
          runTests,
          checkForActionConflicts);
    }
  }

  private final UUID id;
  private final LoadingCache<Class<? extends OptionsBase>, Optional<OptionsBase>> optionsCache;
  private final Map<String, Object> starlarkOptions;

  /** A human-readable description of all the non-default option settings. */
  private final String optionsDescription;

  /**
   * The name of the Blaze command that the user invoked.
   * Used for --announce.
   */
  private final String commandName;

  private final OutErr outErr;
  private final List<String> targets;

  private final long startTimeMillis; // milliseconds since UNIX epoch.

  private final boolean needsInstrumentationFilter;
  private final boolean runningInEmacs;
  private final boolean runTests;
  private final boolean checkForActionConflicts;

  private BuildRequest(
      String commandName,
      final OptionsParsingResult options,
      final OptionsParsingResult startupOptions,
      List<String> targets,
      OutErr outErr,
      UUID id,
      long startTimeMillis,
      boolean needsInstrumentationFilter,
      boolean runTests,
      boolean checkForActionConflicts) {
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
    this.starlarkOptions = options.getStarlarkOptions();
    this.needsInstrumentationFilter = needsInstrumentationFilter;
    this.runTests = runTests;
    this.checkForActionConflicts = checkForActionConflicts;

    for (Class<? extends OptionsBase> optionsClass : MANDATORY_OPTIONS) {
      Preconditions.checkNotNull(getOptions(optionsClass));
    }

    // All this, just to pass a global boolean from the client to the server. :(
    this.runningInEmacs = options.getOptions(UiOptions.class).runningInEmacs;
  }

  /**
   * Since the OptionsProvider interface is used by many teams, this method is String-keyed even
   * though it should always contain labels for our purposes. Consumers of this method should
   * probably use the {@link BuildOptions#labelizeStarlarkOptions} method before doing meaningful
   * work with the results.
   */
  @Override
  public Map<String, Object> getStarlarkOptions() {
    return starlarkOptions;
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

  boolean isRunningInEmacs() {
    return runningInEmacs;
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

  /** Returns the set of options related to the loading phase. */
  public PackageOptions getPackageOptions() {
    return getOptions(PackageOptions.class);
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
  public AnalysisOptions getViewOptions() {
    return getOptions(AnalysisOptions.class);
  }

  /** Returns the value of the --keep_going option. */
  public boolean getKeepGoing() {
    return getOptions(KeepGoingOption.class).keepGoing;
  }

  /** Returns the value of the --loading_phase_threads option. */
  int getLoadingPhaseThreadCount() {
    return getOptions(LoadingPhaseThreadsOption.class).threads;
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

  public boolean needsInstrumentationFilter() {
    return needsInstrumentationFilter;
  }

  /**
   * Validates the options for this BuildRequest.
   *
   * <p>Issues warnings or throws {@code InvalidConfigurationException} for option settings that
   * conflict.
   *
   * @return list of warnings
   */
  public List<String> validateOptions() {
    List<String> warnings = new ArrayList<>();

    int localTestJobs = getExecutionOptions().localTestJobs;
    int jobs = getBuildOptions().jobs;
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
    BuildRequestOptions buildOptions = getBuildOptions();
    return new TopLevelArtifactContext(
        getOptions(ExecutionOptions.class).testStrategy.equals("exclusive"),
        getOptions(BuildEventProtocolOptions.class).expandFilesets,
        getOptions(BuildEventProtocolOptions.class).fullyResolveFilesetSymlinks,
        OutputGroupInfo.determineOutputGroups(
            buildOptions.outputGroups, buildOptions.runValidationActions));
  }

  public ImmutableSortedSet<String> getMultiCpus() {
    return ImmutableSortedSet.copyOf(getBuildOptions().multiCpus);
  }

  public ImmutableList<String> getAspects() {
    return ImmutableList.copyOf(getBuildOptions().aspects);
  }

  public boolean getCheckForActionConflicts() {
    return checkForActionConflicts;
  }
}
