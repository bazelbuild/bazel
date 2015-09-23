// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * Implements the loading phase; responsible for:
 * <ul>
 *   <li>target pattern evaluation
 *   <li>test suite expansion
 *   <li>loading the labels needed to construct the build configuration
 *   <li>loading the labels needed for the analysis with the build configuration
 *   <li>loading the transitive closure of the targets and the configuration labels
 * </ul>
 *
 * <p>In order to ensure correctness of incremental loading and of full cache hits, this class is
 * very restrictive about access to its internal state and to its collaborators. In particular, none
 * of the collaborators of this class may change in incompatible ways, such as changing the relative
 * working directory for the target pattern parser, without notifying this class.
 *
 * <p>For full caching, this class tracks the exact values of all inputs to the loading phase. To
 * maximize caching, it is vital that these change as rarely as possible.
 */
public class LoadingPhaseRunner {

  /**
   * Loading phase options.
   */
  public static class Options extends OptionsBase {

    @Option(name = "loading_phase_threads",
        defaultValue = "200",
        category = "undocumented",
        help = "Number of parallel threads to use for the loading phase.")
    public int loadingPhaseThreads;

    @Option(name = "build_tests_only",
        defaultValue = "false",
        category = "what",
        help = "If specified, only *_test and test_suite rules will be built "
          + "and other targets specified on the command line will be ignored. "
          + "By default everything that was requested will be built.")
    public boolean buildTestsOnly;

    @Option(name = "compile_one_dependency",
            defaultValue = "false",
            category = "what",
            help = "Compile a single dependency of the argument files.  "
            + "This is useful for syntax checking source files in IDEs, "
            + "for example, by rebuilding a single target that depends on "
            + "the source file to detect errors as early as possible in the "
            + "edit/build/test cycle.  This argument affects the way all "
            + "non-flag arguments are interpreted; instead of being targets "
            + "to build they are source filenames.  For each source filename "
            + "an arbitrary target that depends on it will be built.")
    public boolean compileOneDependency;

    @Option(name = "test_tag_filters",
        converter = CommaSeparatedOptionListConverter.class,
        defaultValue = "",
        category = "what",
        help = "Specifies a comma-separated list of test tags. Each tag can be optionally " +
               "preceded with '-' to specify excluded tags. Only those test targets will be " +
               "found that contain at least one included tag and do not contain any excluded " +
               "tags. This option affects --build_tests_only behavior and the test command."
        )
    public List<String> testTagFilterList;

    @Option(name = "test_size_filters",
        converter = TestSize.TestSizeFilterConverter.class,
        defaultValue = "",
        category = "what",
        help = "Specifies a comma-separated list of test sizes. Each size can be optionally " +
               "preceded with '-' to specify excluded sizes. Only those test targets will be " +
               "found that contain at least one included size and do not contain any excluded " +
               "sizes. This option affects --build_tests_only behavior and the test command."
        )
    public Set<TestSize> testSizeFilterSet;

    @Option(name = "test_timeout_filters",
        converter = TestTimeout.TestTimeoutFilterConverter.class,
        defaultValue = "",
        category = "what",
        help = "Specifies a comma-separated list of test timeouts. Each timeout can be " +
               "optionally preceded with '-' to specify excluded timeouts. Only those test " +
               "targets will be found that contain at least one included timeout and do not " +
               "contain any excluded timeouts. This option affects --build_tests_only behavior " +
               "and the test command."
        )
    public Set<TestTimeout> testTimeoutFilterSet;

    @Option(name = "test_lang_filters",
        converter = CommaSeparatedOptionListConverter.class,
        defaultValue = "",
        category = "what",
        help = "Specifies a comma-separated list of test languages. Each language can be " +
               "optionally preceded with '-' to specify excluded languages. Only those " +
               "test targets will be found that are written in the specified languages. " +
               "The name used for each language should be the same as the language prefix in the " +
               "*_test rule, e.g. one of 'cc', 'java', 'py', etc." +
               "This option affects --build_tests_only behavior and the test command."
        )
    public List<String> testLangFilterList;
  }

  /**
   * A callback interface to notify the caller about specific events.
   * TODO(bazel-team): maybe we should use the EventBus instead?
   */
  public interface Callback {
    /**
     * Called after the target patterns have been resolved to give the caller a chance to validate
     * the list before proceeding.
     */
    void notifyTargets(Collection<Target> targets) throws LoadingFailedException;

    /**
     * Called after loading has finished, to notify the caller about the visited packages.
     *
     * <p>The set of visited packages is the set of packages in the transitive closure of the
     * union of the top level targets.
     */
    void notifyVisitedPackages(Set<PackageIdentifier> visitedPackages);
  }

  /**
   * The result of the loading phase, i.e., whether there were errors, and which targets were
   * successfully loaded, plus some related metadata.
   */
  public static final class LoadingResult {
    private final boolean hasTargetPatternError;
    private final boolean hasLoadingError;
    private final ImmutableSet<Target> targetsToAnalyze;
    private final ImmutableSet<Target> testsToRun;
    private final ImmutableMap<PackageIdentifier, Path> packageRoots;

    public LoadingResult(boolean hasTargetPatternError, boolean hasLoadingError,
        Collection<Target> targetsToAnalyze, Collection<Target> testsToRun,
        ImmutableMap<PackageIdentifier, Path> packageRoots) {
      this.hasTargetPatternError = hasTargetPatternError;
      this.hasLoadingError = hasLoadingError;
      this.targetsToAnalyze =
          targetsToAnalyze == null ? null : ImmutableSet.copyOf(targetsToAnalyze);
      this.testsToRun = testsToRun == null ? null : ImmutableSet.copyOf(testsToRun);
      this.packageRoots = packageRoots;
    }

    /** Whether there were errors during target pattern evaluation. */
    public boolean hasTargetPatternError() {
      return hasTargetPatternError;
    }

    /** Whether there were errors during the loading phase. */
    public boolean hasLoadingError() {
      return hasLoadingError;
    }

    /** Successfully loaded targets that should be built. */
    public Collection<Target> getTargets() {
      return targetsToAnalyze;
    }

    /** Successfully loaded targets that should be run as tests. Must be a subset of the targets. */
    public Collection<Target> getTestsToRun() {
      return testsToRun;
    }

    /**
     * The map from package names to the package root where each package was found; this is used to
     * set up the symlink tree.
     */
    public ImmutableMap<PackageIdentifier, Path> getPackageRoots() {
      return packageRoots;
    }
  }

  private static final class ParseFailureListenerImpl extends DelegatingEventHandler
      implements ParseFailureListener {
    private final EventBus eventBus;

    private ParseFailureListenerImpl(EventHandler delegate, EventBus eventBus) {
      super(delegate);
      this.eventBus = eventBus;
    }

    @Override
    public void parsingError(String targetPattern, String message) {
      if (eventBus != null) {
        eventBus.post(new ParsingFailedEvent(targetPattern, message));
      }
    }
  }

  private static final Logger LOG = Logger.getLogger(LoadingPhaseRunner.class.getName());

  private final PackageManager packageManager;
  private final TargetPatternEvaluator targetPatternEvaluator;
  private final Set<String> ruleNames;
  private final TransitivePackageLoader pkgLoader;

  public LoadingPhaseRunner(PackageManager packageManager,
                            Set<String> ruleNames) {
    this.packageManager = packageManager;
    this.targetPatternEvaluator = packageManager.getTargetPatternEvaluator();
    this.ruleNames = ruleNames;
    this.pkgLoader = packageManager.newTransitiveLoader();
  }

  public TargetPatternEvaluator getTargetPatternEvaluator() {
    return targetPatternEvaluator;
  }

  public void updatePatternEvaluator(PathFragment relativeWorkingDirectory) {
    targetPatternEvaluator.updateOffset(relativeWorkingDirectory);
  }

  /**
   * This method only exists for the benefit of InfoCommand, which needs to construct
   * a {@code BuildConfigurationCollection} without running a full loading phase. Don't
   * add any more clients; instead, we should change info so that it doesn't need the configuration.
   */
  public boolean loadForConfigurations(EventHandler eventHandler,
      Set<Label> labelsToLoad, boolean keepGoing) throws InterruptedException {
    // Use a new Label Visitor here to avoid erasing the cache on the existing one.
    TransitivePackageLoader transitivePackageLoader = packageManager.newTransitiveLoader();
    boolean loadingSuccessful = transitivePackageLoader.sync(
        eventHandler, ImmutableSet.<Target>of(),
        labelsToLoad, keepGoing, /*parallelThreads=*/10,
        /*maxDepth=*/Integer.MAX_VALUE);
    return loadingSuccessful;
  }

  /**
   * Performs target pattern evaluation, test suite expansion (if requested), and loads the
   * transitive closure of the resulting targets as well as of the targets needed to use the
   * given build configuration provider.
   */
  public LoadingResult execute(EventHandler eventHandler, EventBus eventBus,
      List<String> targetPatterns, Options options,
      ListMultimap<String, Label> labelsToLoadUnconditionally, boolean keepGoing,
      boolean enableLoading, boolean determineTests, @Nullable Callback callback)
          throws TargetParsingException, LoadingFailedException, InterruptedException {
    LOG.info("Starting pattern evaluation");
    Stopwatch timer = Stopwatch.createStarted();
    if (options.buildTestsOnly && options.compileOneDependency) {
      throw new LoadingFailedException("--compile_one_dependency cannot be used together with "
          + "the --build_tests_only option or the 'bazel test' command ");
    }

    EventHandler parseFailureListener = new ParseFailureListenerImpl(eventHandler, eventBus);
    // Determine targets to build:
    ResolvedTargets<Target> targets = getTargetsToBuild(parseFailureListener,
        targetPatterns, options.compileOneDependency, keepGoing);

    ImmutableSet<Target> filteredTargets = targets.getFilteredTargets();

    boolean buildTestsOnly = options.buildTestsOnly;
    ImmutableSet<Target> testsToRun = null;
    ImmutableSet<Target> testFilteredTargets = ImmutableSet.of();

    // Now we have a list of targets to build. If the --build_tests_only option was specified or we
    // want to run tests, we need to determine the list of targets to test. For that, we remove
    // manual tests and apply the command line filters. Also, if --build_tests_only is specified,
    // then the list of filtered targets will be set as build list as well.
    if (determineTests || buildTestsOnly) {
      // Parse the targets to get the tests.
      ResolvedTargets<Target> testTargets = determineTests(parseFailureListener,
          targetPatterns, options, keepGoing);
      if (testTargets.getTargets().isEmpty() && !testTargets.getFilteredTargets().isEmpty()) {
        eventHandler.handle(Event.warn("All specified test targets were excluded by filters"));
      }

      if (buildTestsOnly) {
        // Replace original targets to build with test targets, so that only targets that are
        // actually going to be built are loaded in the loading phase. Note that this has a side
        // effect that any test_suite target requested to be built is replaced by the set of *_test
        // targets it represents; for example, this affects the status and the summary reports.
        Set<Target> allFilteredTargets = new HashSet<>();
        allFilteredTargets.addAll(targets.getTargets());
        allFilteredTargets.addAll(targets.getFilteredTargets());
        allFilteredTargets.removeAll(testTargets.getTargets());
        allFilteredTargets.addAll(testTargets.getFilteredTargets());
        testFilteredTargets = ImmutableSet.copyOf(allFilteredTargets);
        filteredTargets = ImmutableSet.of();

        targets = ResolvedTargets.<Target>builder()
            .merge(testTargets)
            .mergeError(targets.hasError())
            .build();
        if (determineTests) {
          testsToRun = testTargets.getTargets();
        }
      } else /*if (determineTests)*/ {
        testsToRun = testTargets.getTargets();
        targets = ResolvedTargets.<Target>builder()
            .merge(targets)
            // Avoid merge() here which would remove the filteredTargets from the targets.
            .addAll(testsToRun)
            .mergeError(testTargets.hasError())
            .build();
        // filteredTargets is correct in this case - it cannot contain tests that got back in
        // through test_suite expansion, because the test determination would also filter those out.
        // However, that's not obvious, and it might be better to explicitly recompute it.
      }
      if (testsToRun != null) {
        // Note that testsToRun can still be null here, if buildTestsOnly && !shouldRunTests.
        Preconditions.checkState(targets.getTargets().containsAll(testsToRun));
      }
    }

    eventBus.post(new TargetParsingCompleteEvent(targets.getTargets(),
        filteredTargets, testFilteredTargets,
        timer.stop().elapsed(TimeUnit.MILLISECONDS)));

    if (targets.hasError()) {
      eventHandler.handle(Event.warn("Target pattern parsing failed. Continuing anyway"));
    }
    if (callback != null) {
      callback.notifyTargets(targets.getTargets());
    }
    maybeReportDeprecation(eventHandler, targets.getTargets());
    BaseLoadingResult result;
    if (enableLoading) {
      result = doLoadingPhase(eventHandler, eventBus, targets.getTargets(),
          labelsToLoadUnconditionally, keepGoing, options.loadingPhaseThreads);
      LoadingResult loadingResult = new LoadingResult(targets.hasError(), !result.isSuccesful(),
          result.getTargets(), testsToRun,
          collectPackageRoots(pkgLoader.getErrorFreeVisitedPackages()));
      freeMemoryAfterLoading(callback, pkgLoader.getVisitedPackageNames());
      return loadingResult;
    } else {
      result = doSimpleLoadingPhase(eventHandler, eventBus, targets.getTargets(), keepGoing);
      return new LoadingResult(targets.hasError(), !result.isSuccesful(), result.getTargets(),
          testsToRun, collectPackageRoots(new ArrayList<Package>()));
    }
  }

  private void freeMemoryAfterLoading(Callback callback, Set<PackageIdentifier> visitedPackages) {
    if (callback != null) {
      callback.notifyVisitedPackages(visitedPackages);
    }
    // Clear some targets from the cache to free memory.
    packageManager.partiallyClear();
  }

  /** 
   * Simplified version of {@code doLoadingPhase} method. This method does not load targets.
   * It only does test_suite expansion and emits necessary events and logging messages for legacy
   * support.
   */
  private BaseLoadingResult doSimpleLoadingPhase(EventHandler eventHandler, EventBus eventBus,
      ImmutableSet<Target> targetsToLoad, boolean keepGoing)
          throws LoadingFailedException {
    Stopwatch timer = preLoadingLogging(eventHandler);

    BaseLoadingResult expandedResult;
    try {
      expandedResult = expandTestSuites(eventHandler, targetsToLoad, keepGoing);
    } catch (TargetParsingException e) {
      throw new LoadingFailedException("Loading failed; build aborted", e);
    }

    postLoadingLogging(eventBus, targetsToLoad, expandedResult.getTargets(), timer);
    return new BaseLoadingResult(expandedResult.getTargets(), expandedResult.isSuccesful());
  }

  /**
   * Visit the transitive closure of the targets, populating the package cache
   * and ensuring that all labels can be resolved and all rules were free from
   * errors.
   *
   * @param targetsToLoad the list of command-line target patterns specified by the user
   * @param labelsToLoadUnconditionally the labels to load unconditionally (presumably for the build
   *                                    configuration)
   * @param keepGoing if true, don't throw ViewCreationFailedException if some
   *                  targets could not be loaded, just skip thm.
   */
  private BaseLoadingResult doLoadingPhase(EventHandler eventHandler, EventBus eventBus,
      ImmutableSet<Target> targetsToLoad, ListMultimap<String, Label> labelsToLoadUnconditionally,
      boolean keepGoing, int loadingPhaseThreads)
          throws InterruptedException, LoadingFailedException {
    Stopwatch timer = preLoadingLogging(eventHandler);

    BaseLoadingResult baseResult = performLoadingOfTargets(eventHandler, eventBus, targetsToLoad,
        labelsToLoadUnconditionally, keepGoing, loadingPhaseThreads);
    BaseLoadingResult expandedResult;
    try {
      expandedResult = expandTestSuites(eventHandler, baseResult.getTargets(),
          keepGoing);
    } catch (TargetParsingException e) {
      // This shouldn't happen, because we've already loaded the targets successfully.
      throw (AssertionError) (new AssertionError("Unexpected target failure").initCause(e));
    }

    postLoadingLogging(eventBus, baseResult.getTargets(), expandedResult.getTargets(), timer);
    return new BaseLoadingResult(expandedResult.getTargets(),
        baseResult.isSuccesful() && expandedResult.isSuccesful());
  }

  private Stopwatch preLoadingLogging(EventHandler eventHandler) {
    eventHandler.handle(Event.progress("Loading..."));
    LOG.info("Starting loading phase");
    return Stopwatch.createStarted();
  }

  private void postLoadingLogging(EventBus eventBus, ImmutableSet<Target> originalTargetsToLoad,
      ImmutableSet<Target> expandedTargetsToLoad, Stopwatch timer) {
    Set<Target> testSuiteTargets = Sets.difference(originalTargetsToLoad, expandedTargetsToLoad);
    eventBus.post(new LoadingPhaseCompleteEvent(expandedTargetsToLoad, testSuiteTargets,
        packageManager.getStatistics(), timer.stop().elapsed(TimeUnit.MILLISECONDS)));
    LOG.info("Loading phase finished"); 
  }

  private BaseLoadingResult performLoadingOfTargets(EventHandler eventHandler, EventBus eventBus,
      ImmutableSet<Target> targetsToLoad, ListMultimap<String, Label> labelsToLoadUnconditionally,
      boolean keepGoing, int loadingPhaseThreads) throws InterruptedException,
      LoadingFailedException {
    Set<Label> labelsToLoad = ImmutableSet.copyOf(labelsToLoadUnconditionally.values());

    // For each label in {@code targetsToLoad}, ensure that the target to which
    // it refers exists, and also every target in its transitive closure of label
    // dependencies. Success guarantees that a call to
    // {@code getConfiguredTarget} for the same targets will not fail; the
    // configuration process is intolerant of missing packages/targets. Before
    // calling getConfiguredTarget(), clients must ensure that all necessary
    // packages/targets have been visited since the last sync/clear.
    boolean loadingSuccessful = pkgLoader.sync(eventHandler, targetsToLoad, labelsToLoad,
          keepGoing, loadingPhaseThreads, Integer.MAX_VALUE);

    ImmutableSet<Target> targetsToAnalyze;
    if (loadingSuccessful) {
      // Success: all loaded targets will be analyzed.
      targetsToAnalyze = targetsToLoad;
    } else if (keepGoing) {
      // Keep going: filter out the error-free targets and only continue with those.
      targetsToAnalyze = filterErrorFreeTargets(eventBus, targetsToLoad,
          labelsToLoadUnconditionally);
      reportAboutPartiallySuccesfulLoading(targetsToLoad, targetsToAnalyze, eventHandler);
    } else {
      throw new LoadingFailedException("Loading failed; build aborted");
    }
    return new BaseLoadingResult(targetsToAnalyze, loadingSuccessful);
  }

  private void reportAboutPartiallySuccesfulLoading(ImmutableSet<Target> requestedTargets,
      ImmutableSet<Target> loadedTargets, EventHandler eventHandler) {
    // Tell the user about the subset of successful targets.
    int requested = requestedTargets.size();
    int loaded = loadedTargets.size();
    if (0 < loaded) {
      String message = String.format("Loading succeeded for only %d of %d targets", loaded,
          requested);
      eventHandler.handle(Event.info(message));
      LOG.info(message);
    }
  }

  private BaseLoadingResult expandTestSuites(EventHandler eventHandler,
      ImmutableSet<Target> targets, boolean keepGoing)
      throws LoadingFailedException, TargetParsingException {
    // We use strict test_suite expansion here to match the analysis-time checks.
    ResolvedTargets<Target> expandedResult = TestTargetUtils.expandTestSuites(
        packageManager, eventHandler, targets, /*strict=*/true, /*keepGoing=*/true);
    if (expandedResult.hasError() && !keepGoing) {
      throw new LoadingFailedException("Could not expand test suite target");
    }
    return new BaseLoadingResult(expandedResult.getTargets(), !expandedResult.hasError());
  }

  private static class BaseLoadingResult {
    private final ImmutableSet<Target> targets;
    private final boolean succesful;

    BaseLoadingResult(ImmutableSet<Target> targets, boolean succesful) {
      this.targets = targets;
      this.succesful = succesful;
    }

    ImmutableSet<Target> getTargets() {
      return targets;
    }

    boolean isSuccesful() {
      return succesful;
    }
  }

  private Collection<Target> getTargetsForLabels(Collection<Label> labels) {
    Set<Target> result = new HashSet<>();

    for (Label label : labels) {
      try {
        result.add(packageManager.getLoadedTarget(label));
      } catch (NoSuchThingException e) {
        throw new IllegalStateException(e);  // The target should have been loaded
      }
    }

    return result;
  }

  private ImmutableSet<Target> filterErrorFreeTargets(
      EventBus eventBus, Collection<Target> targetsToLoad,
      ListMultimap<String, Label> labelsToLoadUnconditionally) throws LoadingFailedException {
    // Error out if any of the labels needed for the configuration could not be loaded.
    Collection<Label> labelsToLoad = new ArrayList<>(labelsToLoadUnconditionally.values());
    for (Target target : targetsToLoad) {
      labelsToLoad.add(target.getLabel());
    }
    Multimap<Label, Label> rootCauses = pkgLoader.getRootCauses(labelsToLoad);
    for (Map.Entry<String, Label> entry : labelsToLoadUnconditionally.entries()) {
      if (rootCauses.containsKey(entry.getValue())) {
        throw new LoadingFailedException("Failed to load required " + entry.getKey()
            + " target: '" + entry.getValue() + "'");
      }
    }

    // Post root causes for command-line targets that could not be loaded.
    for (Map.Entry<Label, Label> entry : rootCauses.entries()) {
      eventBus.post(new LoadingFailureEvent(entry.getKey(), entry.getValue()));
    }

    return ImmutableSet.copyOf(Sets.difference(ImmutableSet.copyOf(targetsToLoad),
        ImmutableSet.copyOf(getTargetsForLabels(rootCauses.keySet()))));
  }

  /**
   * Returns a map of collected package names to root paths.
   */
  public static ImmutableMap<PackageIdentifier, Path> collectPackageRoots(
      Collection<Package> packages) {
    // Make a map of the package names to their root paths.
    ImmutableMap.Builder<PackageIdentifier, Path> packageRoots = ImmutableMap.builder();
    for (Package pkg : packages) {
      packageRoots.put(pkg.getPackageIdentifier(), pkg.getSourceRoot());
    }
    return packageRoots.build();
  }

  /**
   * Interpret the command-line arguments.
   *
   * @param targetPatterns the list of command-line target patterns specified by the user
   * @param compileOneDependency if true, enables alternative interpretation of targetPatterns; see
   *     {@link Options#compileOneDependency}
   * @throws TargetParsingException if parsing failed and !keepGoing
   */
  private ResolvedTargets<Target> getTargetsToBuild(EventHandler eventHandler,
      List<String> targetPatterns, boolean compileOneDependency,
      boolean keepGoing) throws TargetParsingException, InterruptedException {
    ResolvedTargets<Target> result =
        targetPatternEvaluator.parseTargetPatternList(eventHandler, targetPatterns,
            FilteringPolicies.FILTER_MANUAL, keepGoing);
    if (compileOneDependency) {
      return new CompileOneDependencyTransformer(packageManager)
          .transformCompileOneDependency(eventHandler, result);
    }
    return result;
  }

  /**
   * Interpret test target labels from the command-line arguments and return the corresponding set
   * of targets, handling the filter flags, and expanding test suites.
   *
   * @param eventHandler the error event eventHandler
   * @param targetPatterns the list of command-line target patterns specified by the user
   * @param options the loading phase options
   * @param keepGoing value of the --keep_going flag
   */
  private ResolvedTargets<Target> determineTests(EventHandler eventHandler,
      List<String> targetPatterns, Options options, boolean keepGoing)
          throws TargetParsingException, InterruptedException {
    // Parse the targets to get the tests.
    ResolvedTargets<Target> testTargetsBuilder = targetPatternEvaluator.parseTargetPatternList(
        eventHandler, targetPatterns, FilteringPolicies.FILTER_TESTS, keepGoing);

    ResolvedTargets.Builder<Target> finalBuilder = ResolvedTargets.builder();
    finalBuilder.merge(testTargetsBuilder);
    finalBuilder.filter(TestFilter.forOptions(options, eventHandler, ruleNames));
    return finalBuilder.build();
  }

  /**
   * Emit a warning when a deprecated target is mentioned on the command line.
   *
   * <p>Note that this does not stop us from emitting "target X depends on deprecated target Y"
   * style warnings for the same target and it is a good thing; <i>depending</i> on a target and
   * <i>wanting</i> to build it are different things.
   */
  private void maybeReportDeprecation(EventHandler eventHandler, Collection<Target> targets) {
    for (Rule rule : Iterables.filter(targets, Rule.class)) {
      if (rule.isAttributeValueExplicitlySpecified("deprecation")) {
        eventHandler.handle(Event.warn(rule.getLocation(), String.format(
            "target '%s' is deprecated: %s", rule.getLabel(),
            NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING))));
      }
    }
  }
}
