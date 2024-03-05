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

import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toCollection;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildtool.BuildPrecompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildToolFinalizingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.SkyfocusOptions;
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
import com.google.devtools.build.skyframe.InMemoryGraphImpl;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.Keep;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
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

    if (!commandActuallyBuilds(env.getCommand())) {
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
    return commandActuallyBuilds(env.getCommand()) && skyfocusOptions.skyfocusEnabled;
  }

  /**
   * Checks if the command builds.
   *
   * <p>Not all 'build = true' annotated commands actually run a build.
   */
  private static boolean commandActuallyBuilds(Command command) {
    if (!command.builds()) {
      return false;
    }
    // 'clean' and 'info' set 'build = true' to make build options accessible to users (and info
    // uses them), but does not run a build.
    if (command.name().equals("clean")) {
      return false;
    }
    if (command.name().equals("info")) {
      return false;
    }
    return true;
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
    if (skyfocusOptions.dumpPostGcStats) {
      beforeHeap = UsedHeapSizeAfterGcInfoItem.getHeapUsageAfterGc();
    }

    // Run Skyfocus!
    FocusResult focusResult = focus();

    // Shouldn't result in an empty graph.
    Preconditions.checkState(!focusResult.getDeps().isEmpty());
    Preconditions.checkState(!focusResult.getRdeps().isEmpty());

    env.getSkyframeExecutor().setSkyfocusVerificationSet(focusResult.getVerificationSet());

    if (skyfocusOptions.dumpKeys) {
      dumpKeys(env.getReporter(), focusResult);
    }

    reportNodeReduction(
        env.getReporter(),
        beforeNodeCount,
        env.getSkyframeExecutor().getEvaluator().getValues().size());

    if (skyfocusOptions.dumpPostGcStats) {
      // Users may skip heap size reporting, which triggers slow manual GCs, in place of faster
      // focusing.
      reportHeapReduction(
          env.getReporter(), beforeHeap, UsedHeapSizeAfterGcInfoItem.getHeapUsageAfterGc());
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
    HashSet<RootedPath> workingSetRootedPaths =
        env.getSkyframeExecutor().getWorkingSet().stream()
            .map(f -> RootedPath.toRootedPath(packageRoot, PathFragment.create(f)))
            .collect(toCollection(HashSet::new));

    Set<SkyKey> leafs = new LinkedHashSet<>();
    graph.parallelForEach(
        node -> {
          SkyKey k = node.getKey();
          if (k instanceof FileStateKey) {
            RootedPath rootedPath = ((FileStateKey) k).argument();
            if (workingSetRootedPaths.remove(rootedPath)) {
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
    if (!workingSetRootedPaths.isEmpty()) {
      reporter.handle(
          Event.warn(
              workingSetRootedPaths.size()
                  + " files were not found in the transitive closure, and "
                  + "so they are not included in the working set. They are: "
                  + workingSetRootedPaths.stream()
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
      focusResult = SkyframeFocuser.focus(graph, reporter, roots, leafs);
    }

    return focusResult;
  }

  private static void reportNodeReduction(
      Reporter reporter, int beforeNodeCount, int afterNodeCount) {
    reporter.handle(
        Event.info(
            String.format(
                "Node count: %s -> %s (%.2f%% reduction)",
                beforeNodeCount,
                afterNodeCount,
                (double) (beforeNodeCount - afterNodeCount) / beforeNodeCount * 100)));
  }

  private static void reportHeapReduction(Reporter reporter, long beforeHeap, long afterHeap) {
    reporter.handle(
        Event.info(
            String.format(
                "Heap: %s -> %s (%.2f%% reduction)",
                StringUtilities.prettyPrintBytes(beforeHeap),
                StringUtilities.prettyPrintBytes(afterHeap),
                (double) (beforeHeap - afterHeap) / beforeHeap * 100)));
  }

  /**
   * Reports the computed set of SkyKeys that need to be kept in the Skyframe graph for incremental
   * correctness.
   *
   * @param reporter the event reporter
   * @param focusResult the result from SkyframeFocuser
   */
  private static void dumpKeys(Reporter reporter, SkyframeFocuser.FocusResult focusResult) {
    try (PrintStream pos = new PrintStream(reporter.getOutErr().getOutputStream())) {
      focusResult
          .getRoots()
          .forEach(k -> reporter.handle(Event.info("root: " + k.getCanonicalName())));
      focusResult
          .getLeafs()
          .forEach(k -> reporter.handle(Event.info("leaf: " + k.getCanonicalName())));

      pos.println("Rdeps kept:\n");
      focusResult.getRdeps().forEach(k -> pos.println(k.getCanonicalName()));

      pos.println();

      pos.println("Deps kept:");
      focusResult.getDeps().forEach(k -> pos.println(k.getCanonicalName()));

      pos.println();

      pos.println("Verification set:");
      focusResult.getVerificationSet().forEach(k -> pos.println(k.getCanonicalName()));

      Map<SkyFunctionName, Long> skyKeyCount =
          Sets.union(focusResult.getRdeps(), focusResult.getDeps()).stream()
              .collect(Collectors.groupingBy(SkyKey::functionName, Collectors.counting()));

      pos.println();
      pos.println("Summary of kept keys:");
      skyKeyCount.forEach((k, v) -> pos.println(k + " " + v));
    }
  }

  private static DetailedExitCode createDetailedExitCode(String message, Skyfocus.Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setSkyfocus(Skyfocus.newBuilder().setCode(code).build())
            .build());
  }
}
