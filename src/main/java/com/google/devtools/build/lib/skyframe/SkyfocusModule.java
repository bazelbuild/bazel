// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildPrecompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildToolFinalizingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.SkyfocusOptions;
import com.google.devtools.build.lib.runtime.SkyfocusOptions.SkyfocusDumpOption;
import com.google.devtools.build.lib.runtime.commands.info.UsedHeapSizeAfterGcInfoItem;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Skyfocus;
import com.google.devtools.build.lib.server.FailureDetails.Skyfocus.Code;
import com.google.devtools.build.lib.skyframe.SkyframeFocuser.FocusResult;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryGraphImpl;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.Keep;
import java.io.PrintStream;
import java.util.Set;
import java.util.function.LongFunction;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * SkyfocusModule implements the concept of using working sets to reduce the memory footprint for
 * incremental builds.
 *
 * <p>This is achieved with the `--experimental_working_set` build flag that takes in a
 * comma-separated list of files, which defines the active working set.
 *
 * <p>Then, the active working set will be used to apply the optimizing algorithm when {@link
 * BuildPrecompleteEvent} is fired, which is just before the build request stops. The core algorithm
 * is implemented in {@link SkyframeFocuser}.
 */
public class SkyfocusModule extends BlazeModule {

  // Leaf keys to be kept regardless of the working set.
  public static final ImmutableSet<SkyKey> INCLUDE_KEYS_IF_EXIST =
      ImmutableSet.of(
          // Necessary for build correctness of repos that are force-fetched between builds.
          // Only found in the Bazel graph, not Blaze's.
          //
          // TODO: b/312819241 - is there a better way to keep external repos in the graph?
          RepositoryDelegatorFunction.FORCE_FETCH.getKey(),
          RepositoryDelegatorFunction.FORCE_FETCH_CONFIGURE.getKey());

  private enum PendingSkyfocusState {
    // Blaze has to reset the evaluator state and restart analysis before running Skyfocus. This is
    // usually due to Skyfocus having dropped nodes in a prior invocation, and there's no way to
    // recover from it. This can be expensive.
    RERUN_ANALYSIS_THEN_RUN_FOCUS,

    // Trigger Skyfocus.
    RUN_FOCUS,

    DO_NOTHING
  }

  @Nullable private CommandEnvironment env;

  private PendingSkyfocusState pendingSkyfocusState;

  @Nullable private SkyfocusOptions skyfocusOptions;

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    // This should come before everything, as 'clean' would cause Blaze to drop its analysis
    // state, therefore focusing needs to be re-done no matter what.
    if (env.getCommandName().equals("clean")) {
      env.getSkyframeExecutor().setWorkingSet(ImmutableSet.of());
      return;
    }

    if (!env.commandActuallyBuilds()) {
      return;
    }
    // All commands that inherit from 'build' will have SkyfocusOptions.
    skyfocusOptions = env.getOptions().getOptions(SkyfocusOptions.class);
    Preconditions.checkNotNull(skyfocusOptions);

    if (!env.getSkyframeExecutor().getEvaluator().skyfocusSupported()) {
      env.getSkyframeExecutor().setWorkingSet(ImmutableSet.of());
      return;
    }

    env.getSkyframeExecutor().getEvaluator().setSkyfocusEnabled(skyfocusOptions.skyfocusEnabled);
    if (!skyfocusOptions.skyfocusEnabled) {
      env.getSkyframeExecutor().setWorkingSet(ImmutableSet.of());
      return;
    }

    // Allows this object to listen to build events.
    env.getEventBus().register(this);
    this.env = env;
    ImmutableSet<String> activeWorkingSet = env.getSkyframeExecutor().getWorkingSet();

    if (!activeWorkingSet.isEmpty()) {
      env.getReporter()
          .handle(
              Event.warn(
                  "You are using the experimental Skyfocus feature. Feel free to test it, "
                      + "but do not depend on it yet."));

      // TODO: b/323434582 - Implement verification sets.
      env.getReporter()
          .handle(
              Event.warn(
                  "Skyfocus: Changes not in the active working set will cause a build error."
                      + " Run '"
                      + env.getRuntime().getProductName()
                      + " info working_set' to print the set."));
    }

    ImmutableSet<String> newWorkingSet = ImmutableSet.copyOf(skyfocusOptions.workingSet);
    pendingSkyfocusState = getPendingSkyfocusState(activeWorkingSet, newWorkingSet);

    switch (pendingSkyfocusState) {
      case RERUN_ANALYSIS_THEN_RUN_FOCUS:
        env.getReporter()
            .handle(
                Event.warn(
                    "Working set changed to include new files, discarding analysis cache. This can"
                        + " be expensive, so choose your working set carefully."));
        env.getSkyframeExecutor().resetEvaluator();
        // fall through
      case RUN_FOCUS:
        env.getReporter()
            .handle(
                Event.info(
                    "Updated working set successfully. Skyfocus will run at the end of the"
                        + " build."));
        env.getSkyframeExecutor().setWorkingSet(newWorkingSet);
        env.getSkyframeExecutor().getEvaluator().setSkyfocusEnabled(true);
        break;
      case DO_NOTHING:
        // Do not replace the active working set.
        break;
    }
  }

  private boolean skyfocusEnabled() {
    return env.commandActuallyBuilds() && skyfocusOptions.skyfocusEnabled;
  }

  /**
   * Compute the next state of Skyfocus using the active and new working set definitions.
   *
   * <p>TODO: b/323434582 - this should incorporate checking other forms of potential build
   * incorrectness, like editing files outside of the working set.
   */
  private static PendingSkyfocusState getPendingSkyfocusState(
      Set<String> activeWorkingSet, Set<String> newWorkingSet) {

    // Skyfocus is not active.
    if (activeWorkingSet.isEmpty()) {
      if (newWorkingSet.isEmpty()) {
        // No new working set is defined. Do nothing.
        return PendingSkyfocusState.DO_NOTHING;
      } else {
        // New working set is defined. Run focus for the first time.
        return PendingSkyfocusState.RUN_FOCUS;
      }
    }

    // activeWorkingSet is not empty, so Skyfocus is active.
    if (newWorkingSet.isEmpty() || newWorkingSet.equals(activeWorkingSet)) {
      // Unchanged working set.
      return PendingSkyfocusState.DO_NOTHING;
    } else if (activeWorkingSet.containsAll(newWorkingSet)) {
      // New working set is a subset of the current working set. Refocus on the new working set and
      // minimize the memory footprint further.
      return PendingSkyfocusState.RUN_FOCUS;
    } else {
      // New working set contains new files. Unfortunately, this is a suboptimal path, and we
      // have to re-run full analysis.
      return PendingSkyfocusState.RERUN_ANALYSIS_THEN_RUN_FOCUS;
    }
  }

  /** Subscriber trigger for Skyfocus using information from {@link AnalysisPhaseCompleteEvent}. */
  @SuppressWarnings("unused")
  @Subscribe
  public void onAnalysisPhaseComplete(AnalysisPhaseCompleteEvent event) {
    if (!skyfocusEnabled()) {
      return;
    }

    // If there's an active working set and the analysis cache was dropped for any reason (e.g.
    // configuration change), we need to re-run Skyfocus.
    if (event.wasAnalysisCacheDropped() && !env.getSkyframeExecutor().getWorkingSet().isEmpty()) {
      pendingSkyfocusState = PendingSkyfocusState.RUN_FOCUS;
    }
  }

  /**
   * Subscriber trigger for Skyfocus using {@link BuildToolFinalizingEvent}.
   *
   * <p>This fires just before the build completes, which is the perfect time for applying Skyfocus.
   * Skyfocus events should be profiled as part of the build command, so it should happen before the
   * build completes or BuildTool request finishes.
   */
  @Keep
  @Subscribe
  public void onBuildToolFinalizingEvent(BuildToolFinalizingEvent event)
      throws InterruptedException, AbruptExitException {
    if (!skyfocusEnabled()) {
      return;
    }

    if (pendingSkyfocusState == PendingSkyfocusState.DO_NOTHING) {
      // Skyfocus doesn't need to run, nothing to do here.
      return;
    }

    if (!event.getDetailedExitCode().isSuccess()) {
      env.getReporter().handle(Event.warn("Skyfocus did not run due to an unsuccessful build."));
      return;
    }

    int beforeNodeCount = env.getSkyframeExecutor().getEvaluator().getValues().size();
    long beforeHeap = 0;
    long beforeActionCacheEntries = env.getBlazeWorkspace().getPersistentActionCache().size();
    if (skyfocusOptions.dumpPostGcStats) {
      beforeHeap = UsedHeapSizeAfterGcInfoItem.getHeapUsageAfterGc();
    }

    ImmutableMultiset<SkyFunctionName> skyFunctionCountBefore = ImmutableMultiset.of();
    InMemoryGraph graph = env.getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    SkyfocusDumpOption dumpKeysOption = skyfocusOptions.dumpKeys;
    if (skyfocusOptions.dumpKeys != SkyfocusDumpOption.NONE) {
      skyFunctionCountBefore = getSkyFunctionNameCount(graph);
    }

    // Run Skyfocus!
    FocusResult focusResult = focus();

    // Shouldn't result in an empty graph.
    Preconditions.checkState(!focusResult.getDeps().isEmpty());
    Preconditions.checkState(!focusResult.getRdeps().isEmpty());

    env.getSkyframeExecutor().setSkyfocusVerificationSet(focusResult.getVerificationSet());

    dumpKeys(dumpKeysOption, env.getReporter(), focusResult, graph, skyFunctionCountBefore);

    reportReductions(
        env.getReporter(),
        "Node count",
        beforeNodeCount,
        env.getSkyframeExecutor().getEvaluator().getValues().size(),
        Long::toString);

    reportReductions(
        env.getReporter(),
        "Action cache count",
        beforeActionCacheEntries,
        env.getBlazeWorkspace().getPersistentActionCache().size(),
        Long::toString);

    if (skyfocusOptions.dumpPostGcStats) {
      // Users may skip heap size reporting, which triggers slow manual GCs, in place of faster
      // focusing.
      reportReductions(
          env.getReporter(),
          "Heap",
          beforeHeap,
          UsedHeapSizeAfterGcInfoItem.getHeapUsageAfterGc(),
          StringUtilities::prettyPrintBytes);
    }

    env.getSkyframeExecutor().getEvaluator().cleanupLatestTopLevelEvaluations();
  }

  /** The main entry point of the Skyfocus optimizations agains the Skyframe graph. */
  private FocusResult focus() throws InterruptedException, AbruptExitException {
    // TODO: b/312819241 - add support for SerializationCheckingGraph for use in tests.
    InMemoryMemoizingEvaluator evaluator =
        (InMemoryMemoizingEvaluator) env.getSkyframeExecutor().getEvaluator();
    InMemoryGraphImpl graph = (InMemoryGraphImpl) evaluator.getInMemoryGraph();

    Reporter reporter = env.getReporter();

    // Compute the roots and leafs.
    Set<SkyKey> roots = evaluator.getLatestTopLevelEvaluations();
    // Skyfocus needs roots. If this fails, there's something wrong with the root-remembering
    // logic in the evaluator.
    Preconditions.checkState(roots != null && !roots.isEmpty());

    // TODO: b/312819241 - For simplicity's sake, use the first --package_path as the root.
    // This may be an issue with packages from a different package_path root.
    Root packageRoot = env.getPackageLocator().getPathEntries().get(0);
    ImmutableSet<RootedPath> workingSetRootedPaths =
        env.getSkyframeExecutor().getWorkingSet().stream()
            .map(f -> RootedPath.toRootedPath(packageRoot, PathFragment.create(f)))
            .collect(toImmutableSet());

    Set<SkyKey> leafs = Sets.newConcurrentHashSet();
    graph.parallelForEach(
        node -> {
          SkyKey k = node.getKey();
          if (k instanceof FileStateKey) {
            RootedPath rootedPath = ((FileStateKey) k).argument();
            if (workingSetRootedPaths.contains(rootedPath)) {
              leafs.add(k);
            }
          }
        });
    if (leafs.isEmpty()) {
      throw new AbruptExitException(
          createDetailedExitCode(
              "Failed to construct working set because none of the files in the working set are"
                  + " found in the transitive closure of the build.",
              Code.INVALID_WORKING_SET));
    }
    int missingCount = workingSetRootedPaths.size() - leafs.size();
    if (missingCount > 0) {
      reporter.handle(
          Event.warn(
              missingCount
                  + " files were not found in the transitive closure, and so they are not included"
                  + " in the working set. They are: "
                  + workingSetRootedPaths.stream()
                      .filter(Predicate.not(leafs::contains))
                      .map(r -> r.getRootRelativePath().toString())
                      .collect(joining(", "))));
    }

    // TODO: b/312819241 - this leaf is necessary for build correctness of volatile actions, like
    // stamping, but retains a lot of memory (100MB of retained heap for a 9+GB build).
    leafs.add(PrecomputedValue.BUILD_ID.getKey()); // needed to invalidate linkstamped targets.

    INCLUDE_KEYS_IF_EXIST.forEach(
        k -> {
          if (graph.getIfPresent(k) != null) {
            leafs.add(k);
          }
        });

    reporter.handle(
        Event.info(
            String.format(
                "Focusing on %d roots, %d leafs.. (use --dump_keys to show them)",
                roots.size(), leafs.size())));

    FocusResult focusResult;

    try (SilentCloseable c = Profiler.instance().profile("SkyframeFocuser")) {
      focusResult =
          SkyframeFocuser.focus(
              graph, env.getBlazeWorkspace().getPersistentActionCache(), reporter, roots, leafs);
    }

    return focusResult;
  }

  private static void reportReductions(
      Reporter reporter,
      String prefix,
      long before,
      long after,
      LongFunction<String> valueFormatter) {
    Preconditions.checkState(!prefix.isEmpty(), "A prefix must be specified.");

    String message =
        String.format(
            "%s: %s -> %s", prefix, valueFormatter.apply(before), valueFormatter.apply(after));
    if (before > 0) {
      message += String.format(" (-%.2f%%)", (double) (before - after) / before * 100);
    }

    reporter.handle(Event.info(message));
  }

  /**
   * Reports the computed set of SkyKeys that need to be kept in the Skyframe graph for incremental
   * correctness.
   *
   * @param reporter the event reporter
   * @param focusResult the result from SkyframeFocuser
   */
  private static void dumpKeys(
      SkyfocusDumpOption dumpKeysOption,
      Reporter reporter,
      SkyframeFocuser.FocusResult focusResult,
      InMemoryGraph graph,
      ImmutableMultiset<SkyFunctionName> skyFunctionNameCountsBefore) {
    if (dumpKeysOption == SkyfocusDumpOption.VERBOSE) {
      try (PrintStream pos = new PrintStream(reporter.getOutErr().getOutputStream())) {
        pos.println("Roots kept:\n");
        focusResult.getRoots().forEach(k -> pos.println(k.getCanonicalName()));

        pos.println("Leafs (including working set) kept:\n");
        focusResult.getLeafs().forEach(k -> pos.println("leaf: " + k.getCanonicalName()));

        pos.println("Rdeps kept:\n");
        focusResult.getRdeps().forEach(k -> pos.println(k.getCanonicalName()));

        pos.println("Deps kept:");
        focusResult.getDeps().forEach(k -> pos.println(k.getCanonicalName()));

        pos.println("Verification set:");
        focusResult.getVerificationSet().forEach(k -> pos.println(k.getCanonicalName()));
      }
    } else if (dumpKeysOption == SkyfocusDumpOption.COUNT) {
      reporter.handle(Event.info(String.format("Roots kept: %d", focusResult.getRoots().size())));
      reporter.handle(Event.info(String.format("Leafs kept: %d", focusResult.getLeafs().size())));
      reporter.handle(Event.info(String.format("Rdeps kept: %d", focusResult.getRdeps().size())));
      reporter.handle(Event.info(String.format("Deps kept: %d", focusResult.getDeps().size())));
      reporter.handle(
          Event.info(
              String.format("Verification set size: %d", focusResult.getVerificationSet().size())));
      ImmutableMultiset<SkyFunctionName> skyFunctionNameCountsAfter =
          getSkyFunctionNameCount(graph);
      skyFunctionNameCountsBefore.forEachEntry(
          (entry, beforeCount) ->
              reportReductions(
                  reporter,
                  entry.toString(),
                  beforeCount,
                  skyFunctionNameCountsAfter.count(entry),
                  Long::toString));
    }
  }

  private static ImmutableMultiset<SkyFunctionName> getSkyFunctionNameCount(InMemoryGraph graph) {
    Multiset<SkyFunctionName> counts = ConcurrentHashMultiset.create();
    graph.parallelForEach(entry -> counts.add(entry.getKey().functionName()));
    return Multisets.copyHighestCountFirst(counts);
  }

  private static DetailedExitCode createDetailedExitCode(String message, Skyfocus.Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setSkyfocus(Skyfocus.newBuilder().setCode(code).build())
            .build());
  }
}
