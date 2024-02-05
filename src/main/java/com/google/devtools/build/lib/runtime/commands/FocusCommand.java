// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.toCollection;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.info.UsedHeapSizeAfterGcInfoItem;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryGraphImpl;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeFocuser;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** The focus command. */
@Command(
    hidden = true, // experimental, don't show in the help command.
    options = {
      FocusCommand.FocusOptions.class,
      PackageOptions.class,
    },
    help =
        "Usage: %{product} focus <options>\n"
            + "Reduces the memory usage of the %{product} JVM by promising that the user will only "
            + "change a given set of files."
            + "\n%{options}",
    name = "focus",
    shortDescription = "EXPERIMENTAL. Reduce memory usage with working sets.")
public class FocusCommand implements BlazeCommand {

  /** The set of options for the focus command. */
  public static class FocusOptions extends OptionsBase {

    @Option(
        name = "experimental_working_set",
        defaultValue = "",
        effectTags = OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS,
        // Deliberately undocumented.
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        converter = Converters.CommaSeparatedOptionListConverter.class,
        help = "The working set. Specify as comma-separated workspace root-relative paths.")
    public List<String> workingSet;

    @Option(
        name = "dump_keys",
        defaultValue = "false",
        effectTags = OptionEffectTag.TERMINAL_OUTPUT,
        // Deliberately undocumented.
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        help = "Dump the focused SkyKeys.")
    public boolean dumpKeys;

    @Option(
        name = "dump_used_heap_size_after_gc",
        defaultValue = "false",
        effectTags = OptionEffectTag.TERMINAL_OUTPUT,
        // Deliberately undocumented.
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        help =
            "If enabled, trigger manual GC before/after focusing to report accurate heap sizes. "
                + "This will increase the focus command's latency.")
    public boolean dumpUsedHeapSizeAfterGc;
  }

  /**
   * Reports the computed set of SkyKeys that need to be kept in the Skyframe graph for incremental
   * correctness.
   *
   * @param reporter the event reporter
   * @param focusResult the result from SkyframeFocuser
   */
  private void dumpKeys(Reporter reporter, SkyframeFocuser.FocusResult focusResult) {
    try (PrintStream pos = new PrintStream(reporter.getOutErr().getOutputStream())) {
      pos.printf("Rdeps kept:\n");
      for (SkyKey key : focusResult.getRdeps()) {
        pos.printf("%s", key.getCanonicalName());
      }
      pos.println();
      pos.println("Deps kept:");
      for (SkyKey key : focusResult.getDeps()) {
        pos.printf("%s", key.getCanonicalName());
      }
      Map<SkyFunctionName, Long> skyKeyCount =
          Sets.union(focusResult.getRdeps(), focusResult.getDeps()).stream()
              .collect(Collectors.groupingBy(SkyKey::functionName, Collectors.counting()));

      pos.println();
      pos.println("Summary of kept keys:");
      skyKeyCount.forEach((k, v) -> pos.println(k + " " + v));
    }
  }

  private static void reportRequestStats(
      CommandEnvironment env, FocusOptions focusOptions, Set<SkyKey> roots, Set<SkyKey> leafs) {
    env.getReporter()
        .handle(
            Event.info(
                String.format(
                    "Focusing on %d roots, %d leafs.. (use --dump_keys to show them)",
                    roots.size(), leafs.size())));
    if (focusOptions.dumpKeys) {
      roots.forEach(k -> env.getReporter().handle(Event.info("root: " + k.getCanonicalName())));
      leafs.forEach(k -> env.getReporter().handle(Event.info("leaf: " + k.getCanonicalName())));
    }
  }

  private static void reportResults(
      CommandEnvironment env,
      boolean dumpUsedHeapSize,
      long beforeHeap,
      int beforeNodeCount,
      long afterHeap,
      int afterNodeCount) {
    StringBuilder results = new StringBuilder();
    if (dumpUsedHeapSize) {
      // Users may skip heap size reporting, which triggers slow manual GCs, in place of faster
      // focusing.
      results.append(
          String.format(
              "Heap: %s -> %s (%.2f%% reduction), ",
              StringUtilities.prettyPrintBytes(beforeHeap),
              StringUtilities.prettyPrintBytes(afterHeap),
              (double) (beforeHeap - afterHeap) / beforeHeap * 100));
    }
    results.append(
        String.format(
            "Node count: %s -> %s (%.2f%% reduction)",
            beforeNodeCount,
            afterNodeCount,
            (double) (beforeNodeCount - afterNodeCount) / beforeNodeCount * 100));
    env.getReporter().handle(Event.info(results.toString()));
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    env.getReporter()
        .handle(
            Event.warn(
                "The focus command is experimental. Feel free to test it, "
                    + "but do not depend on it yet."));

    FocusOptions focusOptions = options.getOptions(FocusOptions.class);

    SkyframeExecutor executor = env.getSkyframeExecutor();
    // TODO: b/312819241 - add support for SerializationCheckingGraph for use in tests.
    InMemoryMemoizingEvaluator evaluator = (InMemoryMemoizingEvaluator) executor.getEvaluator();
    InMemoryGraphImpl graph = (InMemoryGraphImpl) evaluator.getInMemoryGraph();

    // Compute the roots and leafs.
    // TODO: b/312819241 - find a less volatile way to specify roots.
    Set<SkyKey> roots = evaluator.getLatestTopLevelEvaluations();
    if (roots == null || roots.isEmpty()) {
      env.getReporter().handle(Event.error("Unable to focus without roots. Run a build first."));
      // TODO: b/312819241 - turn this into a FailureDetail and avoid crashing.
      throw new IllegalStateException("Unable to get root SkyKeys of the previous build.");
    }

    // TODO: b/312819241 - For simplicity's sake, use the first --package_path as the root.
    // This may be an issue with packages from a different package_path root.
    Root packageRoot = env.getPackageLocator().getPathEntries().get(0);
    HashSet<RootedPath> workingSetRootedPaths =
        focusOptions.workingSet.stream()
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
      // TODO: b/312819241 - turn this into a FailureDetail and avoid crashing.
      throw new IllegalStateException(
          "Failed to construct working set because files not found in the transitive closure: "
              + String.join(", ", focusOptions.workingSet));
    }
    if (!workingSetRootedPaths.isEmpty()) {
      env.getReporter()
          .handle(
              Event.warn(
                  workingSetRootedPaths.size()
                      + " files were not found in the transitive closure, and "
                      + "so they are not included in the working set."));
    }

    // TODO: b/312819241 - this leaf is necessary for build correctness of volatile actions, like
    // stamping, but retains a lot of memory (100MB of retained heap for a 9+GB build).
    leafs.add(PrecomputedValue.BUILD_ID.getKey()); // needed to invalidate linkstamped targets.

    reportRequestStats(env, focusOptions, roots, leafs);

    long beforeHeap = 0;
    long afterHeap = 0;

    int beforeNodeCount = graph.valuesSize();
    if (focusOptions.dumpUsedHeapSizeAfterGc) {
      beforeHeap = UsedHeapSizeAfterGcInfoItem.getHeapUsageAfterGc();
    }

    SkyframeFocuser.FocusResult focusResult;
    try (SilentCloseable c = Profiler.instance().profile("SkyframeFocuser")) {
      focusResult =
          SkyframeFocuser.focus(
              graph,
              env.getReporter(),
              roots,
              leafs,
              /* additionalDepsToKeep= */ (SkyKey k) -> {
                // ActionExecutionFunction#lookupInput allows getting a transitive dep without
                // adding a SkyframeDependency on it. In Blaze/Bazel's case, NestedSets are a major
                // user. To keep that working, it's not sufficient to only keep the direct deps
                // (e.g.
                // NestedSets), but also keep the nodes of the transitive artifacts
                // with this workaround.
                if (k instanceof ArtifactNestedSetKey) {
                  return ((ArtifactNestedSetKey) k)
                      .expandToArtifacts().stream().map(Artifact::key).collect(toImmutableSet());
                }
                return ImmutableSet.of();
              });
    } catch (InterruptedException e) {
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode("focus interrupted"));
    }

    int afterNodeCount = graph.valuesSize();
    if (focusOptions.dumpUsedHeapSizeAfterGc) {
      afterHeap = UsedHeapSizeAfterGcInfoItem.getHeapUsageAfterGc();
    }
    reportResults(
        env,
        focusOptions.dumpUsedHeapSizeAfterGc,
        beforeHeap,
        beforeNodeCount,
        afterHeap,
        afterNodeCount);

    if (focusOptions.dumpKeys) {
      dumpKeys(env.getReporter(), focusResult);
    }

    // Always succeeds (for now).
    return BlazeCommandResult.success();
  }
}
