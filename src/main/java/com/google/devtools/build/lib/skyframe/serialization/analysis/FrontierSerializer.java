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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.util.concurrent.Uninterruptibles.getUninterruptibly;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking.ACTIVE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking.FRONTIER_CANDIDATE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.LongVersionGetterTestInjection.getVersionGetterForTesting;
import static com.google.devtools.build.lib.util.TestType.isInTest;
import static java.lang.Math.min;
import static java.util.Objects.requireNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching.Code;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue.WithRichData;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredExecutionPlatformsValue;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsValue;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.IncrementalInMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Implements frontier serialization with pprof dumping using {@code
 * --experimental_remote_analysis_cache_mode=upload}.
 */
public final class FrontierSerializer {
  @VisibleForTesting static final int MAX_ERRORS_TO_REPORT = 5;

  private FrontierSerializer() {}

  /**
   * Serializes the frontier contained in the current Skyframe graph into a {@link ProfileCollector}
   * writing the resulting proto to {@code path}.
   *
   * @return empty if successful, otherwise a result containing the appropriate error
   */
  public static Optional<FailureDetail> serializeAndUploadFrontier(
      RemoteAnalysisCachingDependenciesProvider dependenciesProvider,
      InMemoryGraph graph,
      LongVersionGetter versionGetter,
      Reporter reporter,
      EventBus eventBus,
      boolean keepStateAfterBuild)
      throws InterruptedException {
    var stopwatch = Stopwatch.createStarted();

    ImmutableSet<SkyKey> selectedKeys;
    ImmutableMap<SkyKey, SelectionMarking> selection = null;
    if (dependenciesProvider.hasActiveDirectoriesMatcher()) {
      selection = computeSelection(graph, dependenciesProvider::withinActiveDirectories);
      selectedKeys = selection.keySet();
    } else {
      selectedKeys = computeFullSelection(graph);
    }

    reporter.handle(
        Event.info(
            String.format(
                "Found %d active or frontier keys in %s", selectedKeys.size(), stopwatch)));
    stopwatch.reset().start();

    if (dependenciesProvider.mode() == RemoteAnalysisCacheMode.DUMP_UPLOAD_MANIFEST_ONLY) {
      if (selection == null) {
        // `selection` is not computed when using full selection and doesn't have a meaningful
        // SelectionMarking assignment. Marks all keys as FRONTIER_CANDIDATES arbitrarily.
        selection =
            selectedKeys.stream()
                .collect(toImmutableMap(key -> key, unused -> SelectionMarking.FRONTIER_CANDIDATE));
      }
      reporter.handle(
          Event.warn("Dry run of upload, dumping selection to stdout (warning: can be large!)"));
      dumpUploadManifest(
          new PrintStream(
              new BufferedOutputStream(reporter.getOutErr().getOutputStream(), 1024 * 1024)),
          selection);
      return Optional.empty();
    }

    ObjectCodecs codecs = requireNonNull(dependenciesProvider.getObjectCodecs());
    FrontierNodeVersion frontierVersion;
    try {
      frontierVersion = dependenciesProvider.getSkyValueVersion();
    } catch (SerializationException e) {
      String message = "error computing frontier version " + e.getMessage();
      reporter.error(null, message);
      return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
    }

    String profilePath = dependenciesProvider.serializedFrontierProfile();
    var profileCollector = profilePath.isEmpty() ? null : new ProfileCollector();
    var serializationStats = new SelectedEntrySerializer.SerializationStats();

    if (versionGetter == null) {
      if (isInTest()) {
        versionGetter = getVersionGetterForTesting();
      } else {
        throw new NullPointerException("missing versionGetter");
      }
    }

    if (!keepStateAfterBuild) {
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // INCREMENTALITY PITFALLS WARNING
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //
      // The following code is not safe to run if the Skyframe graph needs to be
      // incrementally correct after this point.
      //
      // We only do this if --nokeep_state_after_build is set.
      try (var ignored = Profiler.instance().profile(null, "reclaimMemoryFromSkyframe")) {
        // TODO: More ideas, while keeping the constraint that we need the
        // structure of the selection and DTC to be able to compute the
        // FileOpNodes MTSV metadata (File, DirectoryListing) for invalidation.
        // - Delete SkyValues of nodes in the DTC, because the values are not needed by
        // SelectedEntrySerializer.
        // - Delete entire nodes not in selection and DTC, because they are never traversed in
        // SelectedEntrySerializer.
        stopwatch.reset().start();
        long rdepsDeleted = deleteAllRdeps(graph); // saves about 8% RAM b/418730298#comment26
        reporter.handle(
            Event.info(
                String.format(
                    "%s rdeps deleted to reclaim memory, took %s", rdepsDeleted, stopwatch)));
      }
    }

    stopwatch.reset().start();
    ListenableFuture<ImmutableList<Throwable>> writeStatus =
        SelectedEntrySerializer.uploadSelection(
            graph,
            versionGetter,
            codecs,
            frontierVersion,
            selectedKeys,
            dependenciesProvider.getFingerprintValueService(),
            eventBus,
            profileCollector,
            serializationStats);

    try {
      // Waits for the write to complete uninterruptibly. This avoids returning to the caller
      // while underlying worker threads are still processing.
      ImmutableList<Throwable> errors = getUninterruptibly(writeStatus);
      if (!errors.isEmpty()) {
        String message = getErrorMessage(errors);
        reporter.error(/* location= */ null, message, errors.get(0));
        return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
      }

      FingerprintValueStore.Stats stats =
          dependenciesProvider.getFingerprintValueService().getStats();

      reporter.handle(
          Event.info(
              String.format(
                  "Serialized %s/%s analysis/execution nodes into %s/%s key/value bytes and %s"
                      + " entries (%s batches) in %s",
                  serializationStats.analysisNodes(),
                  serializationStats.executionNodes(),
                  stats.keyBytesSent(),
                  stats.valueBytesSent(),
                  stats.entriesWritten(),
                  stats.setBatches(),
                  stopwatch)));
    } catch (ExecutionException e) {
      // The writeStatus future is not known to throw any ExecutionExceptions.
      Throwable cause = e.getCause();
      String message =
          "with unexpected exception type "
              + cause.getClass().getName()
              + ": "
              + cause.getMessage();
      reporter.error(/* location= */ null, message, cause);
      return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
    }

    if (profilePath.isEmpty()) {
      return Optional.empty();
    }

    try (var fileOutput = new FileOutputStream(profilePath);
        var bufferedOutput = new BufferedOutputStream(fileOutput)) {
      profileCollector.toProto().writeTo(bufferedOutput);
    } catch (IOException e) {
      String message = "Error writing serialization profile to file: " + e.getMessage();
      reporter.error(null, message, e);
      return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
    }
    return Optional.empty();
  }

  private static void dumpUploadManifest(PrintStream out, Map<SkyKey, SelectionMarking> selection) {
    var frontierCandidates = ImmutableList.builder();
    var activeSet = ImmutableList.builder();
    selection
        .entrySet()
        .forEach(
            entry -> {
              switch (entry.getValue()) {
                case ACTIVE -> activeSet.add(entry.getKey().getCanonicalName());
                case FRONTIER_CANDIDATE ->
                    frontierCandidates.add(entry.getKey().getCanonicalName());
              }
            });
    frontierCandidates.build().stream()
        .sorted()
        .forEach(k -> out.println("FRONTIER_CANDIDATE: " + k));
    activeSet.build().stream().sorted().forEach(k -> out.println("ACTIVE: " + k));
    out.flush();
  }

  private static ImmutableSet<SkyKey> computeFullSelection(InMemoryGraph graph) {
    Set<SkyKey> selection = ConcurrentHashMap.newKeySet();
    graph.parallelForEach(
        node -> {
          switch (node.getKey()) {
            case ActionLookupKey key -> {
              if (key.getLabel() != null) {
                selection.add(key);
              }
            }
            case ActionLookupData data -> {
              if (shouldUpload(data, node)) {
                selection.add(data);
              }
            }
            case Artifact artifact -> {
              SkyKey artifactKey = selectArtifactKey(artifact);
              if (artifactKey != null) {
                selection.add(artifactKey);
              }
            }
            default -> {}
          }
        });
    return ImmutableSet.copyOf(selection);
  }

  @VisibleForTesting
  enum SelectionMarking {
    /**
     * The entry is a frontier candidate.
     *
     * <p>If a node is still a frontier candidate at the end of the selection process, it is a
     * frontier node.
     */
    FRONTIER_CANDIDATE,
    /** The node is part of the active set. */
    ACTIVE
  }

  private static ImmutableMap<SkyKey, SelectionMarking> computeSelection(
      InMemoryGraph graph, Predicate<PackageIdentifier> matcher) {
    var selection = new ConcurrentHashMap<SkyKey, SelectionMarking>();
    graph.parallelForEach(
        node -> {
          switch (node.getKey()) {
            case ActionLookupKey key -> {
              Label label = key.getLabel();
              if (label != null && matcher.test(label.getPackageIdentifier())) {
                markActiveAndTraverseEdges(graph, key, selection);
              }
            }
            case ActionLookupData data -> {
              if (shouldUpload(data, node)) {
                // Notably, we don't check the `matcher` for execution values, because we want to
                // serialize all ActionLookupData even if they're below the frontier, because the
                // owning ActionLookupValue will be pruned.
                selection.putIfAbsent(data, FRONTIER_CANDIDATE);
              }
            }
            case Artifact artifact -> {
              SkyKey artifactKey = selectArtifactKey(artifact);
              if (artifactKey != null) {
                // TODO: b/441769854 - add test coverage
                selection.putIfAbsent(artifactKey, FRONTIER_CANDIDATE);
              }
            }
            // Some of the analysis nodes reachable from platforms/toolchains SkyFunctions will not
            // be reachable from the regular active analysis nodes traversal above. e.g. these are
            // only reachable from flags, like --extra_execution_platforms.

            // To further elaborate: the frontier contains the ActionLookupValue dependencies of
            // active nodes. Since platform/toolchain deps are not ActionLookupValues, the frontier
            // encountered with markActiveAndTraverseEdges does not contain their direct analysis
            // deps. However, those deps should logically be part of the frontier because
            // platform/toolchain nodes are not currently configured for serialization. To mitigate
            // this, we serialize their ActionLookupValue dependencies when they are encountered.
            //
            // The following cases will include them in the frontier for serialization if they're
            // not within the project's boundaries. They may also be induced as usual into the
            // active set before/after with markActiveAndTraverseEdges, so this ad-hoc traversal is
            // safe.
            //
            // Note that this unconditionally serializes all such deps whether they're reachable
            // from an active analysis node, which may be more work than necessary.
            //
            // TODO: b/397197410 - consider a deeper analysis on the tradeoffs between just
            // serializing the platform and toolchain SkyValues (and updating their respective
            // SkyFunctions to use SkyValueRetriever), instead of serializing their direct
            // dependencies here. Those SkyValues are entirely derived from the build configuration
            // fragments, and the values themselves look relatively straightforward to serialize.
            case RegisteredExecutionPlatformsValue.Key key ->
                markAnalysisDirectDepsAsFrontierCandidates(key, graph, selection);
            case RegisteredToolchainsValue.Key key ->
                markAnalysisDirectDepsAsFrontierCandidates(key, graph, selection);
            case ToolchainContextKey key ->
                markAnalysisDirectDepsAsFrontierCandidates(key, graph, selection);
            default -> {}
          }
        });
    return ImmutableMap.copyOf(selection);
  }

  private static boolean shouldUpload(ActionLookupData data, InMemoryNodeEntry node) {
    // `valueIsShareable` is used by a different system that does not serialize
    // RunfilesArtifactValue, but the FrontierSerializer should do so. A `WithRichData`
    // value type can be used to distinguish this case.
    return data.valueIsShareable() || node.getValue() instanceof WithRichData;
  }

  @Nullable
  private static SkyKey selectArtifactKey(Artifact artifact) {
    if (!artifact.valueIsShareable()) {
      // TODO: b/441769854 - add test coverage
      return null;
    }
    return switch (artifact) {
      case DerivedArtifact derived -> {
        // Artifact#key is the canonical function to produce the SkyKey that will build this
        // artifact. We want to avoid serializing ordinary DerivedArtifacts, which are never built
        // by Skyframe directly, and the function will return ActionLookupData as the canonical key
        // for those artifacts instead.
        SkyKey artifactKey = Artifact.key(derived);
        if (artifactKey instanceof ActionLookupData) {
          yield null; // Handled independently.
        }
        // Notably, we don't check the `matcher` for execution values, because we want to serialize
        // all ActionLookupData even if they're below the frontier, because the owning
        // ActionLookupValue will be pruned.
        yield artifactKey;
      }
      // Does not upload source artifacts because they are cheap to compute.
      case SourceArtifact ignored -> null;
    };
  }

  private static void markActiveAndTraverseEdges(
      InMemoryGraph graph,
      ActionLookupKey root,
      ConcurrentHashMap<SkyKey, SelectionMarking> selection) {
    if (root.getLabel() == null) {
      return;
    }

    InMemoryNodeEntry node = checkNotNull(graph.getIfPresent(root), root);
    // If this node is present in the graph but it's not done it means that it wasn't needed by any
    // node in the evaluation. This can only happen if we ran in upload mode with a warm Skyframe
    // which is not supported and should have thrown an error at the beginning of analysis.
    Preconditions.checkState(node.isDone());

    if (selection.put(root, ACTIVE) == ACTIVE) {
      return;
    }

    for (SkyKey dep : node.getDirectDeps()) {
      if (!(dep instanceof ActionLookupKey actionLookupKey)) {
        continue;
      }

      // Three cases where a child node is disqualified to be a frontier candidate:
      //
      // 1) It doesn't have a label (e.g. BuildInfoKey). These nodes are not deserialized by the
      // analysis functions we care about.
      // 2) It is _already_ marked as ACTIVE, which means it was visited as an rdep from an active
      // root. putIfAbsent will be a no-op.
      // 3) It _will_ be marked as ACTIVE when visited as a rdep from an active root later, and
      // overrides its FRONTIER_CANDIDATE state.
      //
      // In all cases, frontier candidates will never include nodes in the active directories. This
      // is enforced after selection completes.
      if (actionLookupKey.getLabel() != null) {
        selection.putIfAbsent(actionLookupKey, FRONTIER_CANDIDATE);
      }
    }
    for (SkyKey rdep : node.getReverseDepsForDoneEntry()) {
      if (!(rdep instanceof ActionLookupKey parent)) {
        continue;
      }
      // The active set can include nodes outside of the active directories iff they are in the UTC
      // of a root in the active directories.
      markActiveAndTraverseEdges(graph, parent, selection);
    }
  }

  /**
   * Iterates over the direct analysis deps of a node, and include them into the frontier if they've
   * not been seen before.
   */
  private static void markAnalysisDirectDepsAsFrontierCandidates(
      SkyKey key, InMemoryGraph graph, ConcurrentHashMap<SkyKey, SelectionMarking> selection) {
    graph
        .getIfPresent(key)
        .getDirectDeps()
        .forEach(
            depKey -> {
              if (depKey instanceof ActionLookupKey) {
                selection.putIfAbsent(depKey, FRONTIER_CANDIDATE);
              }
            });
  }

  public static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setRemoteAnalysisCaching(RemoteAnalysisCaching.newBuilder().setCode(detailedCode))
        .build();
  }

  /**
   * Deletes all rdeps from the graph.
   *
   * <p>This is not safe to call if the Skyframe graph needs to be incrementally correct after this
   * point.
   *
   * @return the number of rdeps deleted.
   */
  private static long deleteAllRdeps(InMemoryGraph graph) {
    AtomicLong deletedRdeps = new AtomicLong();
    graph.parallelForEach(
        node -> {
          IncrementalInMemoryNodeEntry incrementalInMemoryNodeEntry =
              (IncrementalInMemoryNodeEntry) node;
          for (SkyKey rdep : incrementalInMemoryNodeEntry.getReverseDepsForDoneEntry()) {
            incrementalInMemoryNodeEntry.removeReverseDep(rdep);
            deletedRdeps.incrementAndGet();
          }
          incrementalInMemoryNodeEntry.consolidateReverseDeps();
        });
    return deletedRdeps.get();
  }

  @VisibleForTesting
  static String getErrorMessage(ImmutableList<Throwable> errors) {
    var message = new StringBuilder();
    if (errors.size() > 1) {
      message.append("There were ").append(errors.size()).append(" write errors.");
      if (errors.size() > MAX_ERRORS_TO_REPORT) {
        message
            .append(" Only the first ")
            .append(MAX_ERRORS_TO_REPORT)
            .append(" will be reported.");
      }
      message.append('\n');
    }
    for (int i = 0; i < min(errors.size(), MAX_ERRORS_TO_REPORT); i++) {
      message.append(errors.get(i).getMessage()).append('\n');
    }
    return message.toString();
  }
}
