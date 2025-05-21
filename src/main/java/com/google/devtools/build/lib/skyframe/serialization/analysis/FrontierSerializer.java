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
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking.ACTIVE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking.FRONTIER_CANDIDATE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.LongVersionGetterTestInjection.getVersionGetterForTesting;
import static com.google.devtools.build.lib.util.TestType.isInTest;
import static java.util.Objects.requireNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching.Code;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue.WithRichData;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
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
import com.google.devtools.build.skyframe.SkyKey;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

/**
 * Implements frontier serialization with pprof dumping using {@code
 * --experimental_remote_analysis_cache_mode=upload}.
 */
public final class FrontierSerializer {
  private FrontierSerializer() {}

  /**
   * Serializes the frontier contained in the current Skyframe graph into a {@link ProfileCollector}
   * writing the resulting proto to {@code path}.
   *
   * @return empty if successful, otherwise a result containing the appropriate error
   */
  public static Optional<FailureDetail> serializeAndUploadFrontier(
      RemoteAnalysisCachingDependenciesProvider dependenciesProvider,
      SkyframeExecutor skyframeExecutor,
      LongVersionGetter versionGetter,
      Reporter reporter,
      EventBus eventBus)
      throws InterruptedException {
    var stopwatch = new ResettingStopwatch(Stopwatch.createStarted());
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();

    ImmutableMap<SkyKey, SelectionMarking> selection =
        computeSelection(graph, dependenciesProvider::withinActiveDirectories);

    reporter.handle(
        Event.info(
            String.format("Found %d active or frontier keys in %s", selection.size(), stopwatch)));

    if (dependenciesProvider.mode() == RemoteAnalysisCacheMode.DUMP_UPLOAD_MANIFEST_ONLY) {
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

    var profileCollector = new ProfileCollector();
    var serializedCount = new AtomicInteger();

    if (versionGetter == null) {
      if (isInTest()) {
        versionGetter = getVersionGetterForTesting();
      } else {
        throw new NullPointerException("missing versionGetter");
      }
    }

    ListenableFuture<Void> writeStatus =
        SelectedEntrySerializer.uploadSelection(
            graph,
            versionGetter,
            codecs,
            frontierVersion,
            selection,
            dependenciesProvider.getFingerprintValueService(),
            eventBus,
            profileCollector,
            serializedCount);

    try {
      var unusedNull = writeStatus.get();

      FingerprintValueStore.Stats stats =
          dependenciesProvider.getFingerprintValueService().getStats();

      reporter.handle(
          Event.info(
              String.format(
                  "Serialized %s frontier nodes into %s bytes and %s entries in %s",
                  serializedCount.get(),
                  stats.valueBytesSent(),
                  stats.entriesWritten(),
                  stopwatch)));
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      String message = cause.getMessage();
      if (!(cause instanceof SerializationException || cause instanceof IOException)) {
        message = "with unexpected exception type " + cause.getClass().getName() + ": " + message;
      }
      reporter.error(/* location= */ null, message, cause);
      return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
    }
    reporter.handle(
        Event.info(String.format("Waiting for write futures took an additional %s", stopwatch)));

    String profilePath = dependenciesProvider.serializedFrontierProfile();
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

  @VisibleForTesting
  static ImmutableMap<SkyKey, SelectionMarking> computeSelection(
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
              if (!data.valueIsShareable() && !(node.getValue() instanceof WithRichData)) {
                // `valueIsShareable` is used by a different system that does not serialize
                // RunfilesArtifactValue, but the FrontierSerializer should do so. A `WithRichData`
                // value type can be used to distinguish this case.
                return;
              }
              // Notably, we don't check the `matcher` for execution values, because we want to
              // serialize all ActionLookupData even if they're below the frontier, because the
              // owning ActionLookupValue will be pruned.
              selection.putIfAbsent(data, FRONTIER_CANDIDATE);
            }
            case Artifact artifact -> {
              if (!artifact.valueIsShareable()) {
                return;
              }
              switch (artifact) {
                case DerivedArtifact derived:
                  // Artifact#key is the canonical function to produce the SkyKey that will build
                  // this artifact. We want to avoid serializing ordinary DerivedArtifacts, which
                  // are never built by Skyframe directly, and the function will return
                  // ActionLookupData as the canonical key for those artifacts instead.
                  SkyKey artifactKey = Artifact.key(derived);
                  if (artifactKey instanceof ActionLookupData) {
                    return; // Already handled in the ActionLookupData switch case above.
                  }
                  // Like ActionLookupData, we want to serialize these even if they're below the
                  // frontier.
                  selection.putIfAbsent(artifactKey, FRONTIER_CANDIDATE);
                  break;
                case SourceArtifact ignored:
                  break; // Skips source artifacts because they are cheap to compute.
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

    // Marks ActionExecutionValues owned by active analysis nodes ACTIVE.
    return selection.entrySet().parallelStream()
        .collect(
            toImmutableMap(
                Map.Entry::getKey,
                entry ->
                    switch (entry.getKey()) {
                      case ActionLookupData lookupData ->
                          selection.get(lookupData.getActionLookupKey()) == ACTIVE
                              ? ACTIVE
                              : entry.getValue();
                      case DerivedArtifact artifact ->
                          selection.get(artifact.getArtifactOwner()) == ACTIVE
                              ? ACTIVE
                              : entry.getValue();
                      default -> entry.getValue();
                    }));
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

  /** Stopwatch that resets upon reporting the time via {@link #toString}. */
  private record ResettingStopwatch(Stopwatch stopwatch) {
    @Override
    public String toString() {
      String text = stopwatch.toString();
      stopwatch.reset().start();
      return text;
    }
  }

  public static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setRemoteAnalysisCaching(RemoteAnalysisCaching.newBuilder().setCode(detailedCode))
        .build();
  }
}
