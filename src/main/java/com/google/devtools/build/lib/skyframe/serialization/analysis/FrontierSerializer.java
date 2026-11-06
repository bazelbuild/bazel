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

import static com.google.common.util.concurrent.Uninterruptibles.getUninterruptibly;
import static com.google.devtools.build.lib.skyframe.serialization.ErrorMessageHelper.getErrorMessage;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.TimeUnit.MICROSECONDS;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupSummaryKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.RemoteAnalysisCacheStatistics.UploadPhaseTimings;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching.Code;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue.WithRichData;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingEventListener.UploadPhaseTimingsEvent;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.IncrementalInMemoryNodeEntry;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.time.Duration;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;

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
      SerializationDependenciesProvider serializationDependenciesProvider,
      MemoizingEvaluator evaluator,
      LongVersionGetter versionGetter,
      Reporter reporter,
      EventBus eventBus,
      boolean keepStateAfterBuild)
      throws InterruptedException {
    InMemoryGraph graph = evaluator.getInMemoryGraph();
    var stopwatch = Stopwatch.createStarted();

    ImmutableSet<SkyKey> selectedKeys =
        computeFullSelection(graph, serializationDependenciesProvider.getSkycacheAnalysisOnly());

    reporter.handle(
        Event.info(String.format("Found %d selected keys in %s", selectedKeys.size(), stopwatch)));
    stopwatch.reset().start();

    if (serializationDependenciesProvider.mode()
        == RemoteAnalysisCacheMode.DUMP_UPLOAD_MANIFEST_ONLY) {
      reporter.handle(
          Event.warn("Dry run of upload, dumping selection to stdout (warning: can be large!)"));
      dumpUploadManifest(
          new PrintStream(
              new BufferedOutputStream(reporter.getOutErr().getOutputStream(), 1024 * 1024)),
          selectedKeys);
      return Optional.empty();
    }

    CompressionService compressionService =
        serializationDependenciesProvider.getCompressionService();

    FingerprintValueService fingerprintValueService =
        serializationDependenciesProvider.getFingerprintValueService();
    if (fingerprintValueService == null) {
      return Optional.of(
          createFailureDetail(
              "Remote analysis cache initialization failed (FingerprintValueService is null).",
              Code.UPLOAD_FAILED));
    }
    KeyValueWriter fileInvalidationWriter =
        serializationDependenciesProvider.getFileInvalidationWriter();

    ObjectCodecs codecs = requireNonNull(serializationDependenciesProvider.getObjectCodecs());
    FrontierNodeVersion frontierVersion = serializationDependenciesProvider.getSkyValueVersion();
    String profilePath = serializationDependenciesProvider.getSerializedFrontierProfile();
    var profileCollector = profilePath.isEmpty() ? null : new ProfileCollector();
    var serializationStats = new SelectedEntrySerializer.SerializationStats();

    requireNonNull(versionGetter, "missing versionGetter");

    boolean shouldDiscardMemory = !keepStateAfterBuild;
    if (shouldDiscardMemory) {
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
        // saves about 8% RAM b/418730298#comment26
        var deletionStats =
            deleteNodesAndRdeps(graph, serializationDependenciesProvider.getSkycacheAnalysisOnly());
        reporter.handle(
            Event.info(
                String.format(
                    "%s nodes and %s rdeps deleted to reclaim memory, took %s",
                    deletionStats.nodes, deletionStats.rdeps, stopwatch)));
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
            compressionService,
            fingerprintValueService,
            fileInvalidationWriter,
            shouldDiscardMemory,
            eventBus,
            profileCollector,
            serializationStats,
            serializationDependenciesProvider.getEmitUploadedEvents(),
            requireNonNull(serializationDependenciesProvider.getFileOpNodes()));

    try {
      // Waits for the write to complete uninterruptibly. This avoids returning to the caller
      // while underlying worker threads are still processing.
      ImmutableList<Throwable> errors = getUninterruptibly(writeStatus);
      if (!errors.isEmpty()) {
        String message = getErrorMessage(errors);
        reporter.error(/* location= */ null, message, errors.get(0));
        return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
      }

      reporter.handle(
          Event.info(
              String.format(
                  "Skycache sync write: serialized %s/%s analysis/execution nodes in %s",
                  serializationStats.analysisNodes(),
                  serializationStats.executionNodes(),
                  stopwatch)));
      var phases = UploadPhases.of(serializationStats, stopwatch.elapsed());
      reporter.handle(Event.info(phases.format()));
      eventBus.post(new UploadPhaseTimingsEvent(phases.toProto()));
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

  /**
   * The sync write's elapsed time, broken into the synchronous phases of {@link
   * SelectedEntrySerializer#uploadSelection} and the remaining wait for writes to land.
   */
  private record UploadPhases(
      Duration materialize, Duration leafIds, Duration drive, Duration awaitWrites) {
    static UploadPhases of(SelectedEntrySerializer.SerializationStats stats, Duration total) {
      Duration materialize = stats.materializeNodeGraphTime();
      Duration leafIds = stats.computeLeafIdsAndUploadOrderTime();
      Duration drive = stats.driveUploadsTime();
      return new UploadPhases(
          materialize, leafIds, drive, total.minus(materialize).minus(leafIds).minus(drive));
    }

    UploadPhaseTimings toProto() {
      return UploadPhaseTimings.newBuilder()
          .setMaterializeNodeGraphMicros(MICROSECONDS.convert(materialize))
          .setComputeLeafIdsAndUploadOrderMicros(MICROSECONDS.convert(leafIds))
          .setDriveUploadsMicros(MICROSECONDS.convert(drive))
          .setAwaitWritesMicros(MICROSECONDS.convert(awaitWrites))
          .build();
    }

    String format() {
      return String.format(
          "Skycache sync write: materialize=%d ms, assign and order=%d ms, drive=%d ms,"
              + " await=%d ms",
          materialize.toMillis(), leafIds.toMillis(), drive.toMillis(), awaitWrites.toMillis());
    }
  }

  /**
   * Discards unneeded {@code PackageValue}s and {@code BzlLoadValue}s from the graph after
   * analysis.
   *
   * <p>This is a memory optimization to reduce peak heap before the execution phase. It retains all
   * packages associated with done {@code ConfiguredTargetValue}s (and top-level targets) and clears
   * the values of all other {@code PackageValue} nodes. The graph is mutilated after this and
   * incremental builds are not possible.
   */
  public static void computeSelectionAndMinimizeMemory(
      InMemoryGraph graph, Collection<Label> topLevelTargets) {
    Set<PackageIdentifier> packageIdentifierSet = Sets.newConcurrentHashSet();
    graph.parallelForEach(
        node -> {
          switch (node.getKey()) {
            case BzlLoadValue.Key unused -> ((IncrementalInMemoryNodeEntry) node).clearSkyValue();
            case ActionLookupKey key -> {
              if (!node.isDone() || key.getLabel() == null) {
                break;
              }
              // An ActionLookupKey can point to ActionLookupValues or
              // ConfiguredTargetValues. If it's a ConfiguredTargetValue, we record its
              // package to prevent clearing the corresponding PackageValue later.
              SkyValue value = node.getValue();
              if (value instanceof ConfiguredTargetValue configuredTargetValue) {
                Label label =
                    (configuredTargetValue.getConfiguredTarget()
                            instanceof AliasConfiguredTarget alias)
                        ? alias.getLabel()
                        : key.getLabel();
                packageIdentifierSet.add(label.getPackageIdentifier());
              }
            }
            default -> {}
          }
        });
    packageIdentifierSet.addAll(
        topLevelTargets.stream()
            .map(Label::getPackageIdentifier)
            .collect(ImmutableSet.toImmutableSet()));
    // We can clear the values of PackageIdentifier nodes whose package we have not seen in any
    // done ConfiguredTargetValues or top-level targets.
    graph.parallelForEach(
        node -> {
          if (!(node.getKey() instanceof PackageIdentifier key)) {
            return;
          }

          if (packageIdentifierSet.contains(key)) {
            return;
          }

          ((IncrementalInMemoryNodeEntry) node).clearSkyValue();
        });
  }

  private static ImmutableSet<SkyKey> computeFullSelection(
      InMemoryGraph graph, boolean skycacheAnalysisOnly) {
    Set<SkyKey> selection = Sets.newConcurrentHashSet();
    graph.parallelForEach(
        node -> {
          switch (node.getKey()) {
            case ActionLookupKey key -> {
              if (key.getLabel() != null) {
                selection.add(key);
              }
            }
            case ActionLookupSummaryKey summaryKey when skycacheAnalysisOnly -> {
              if (summaryKey.argument().getLabel() != null) {
                selection.add(summaryKey);
              }
            }
            case ActionLookupData data when !skycacheAnalysisOnly -> {
              if (data.getActionLookupKey().getLabel() != null && shouldUpload(data, node)) {
                selection.add(data);
              }
            }
            case Artifact artifact when !skycacheAnalysisOnly -> {
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

  private static void dumpUploadManifest(PrintStream out, Set<SkyKey> selection) {
    selection.stream()
        .map(SkyKey::getCanonicalName)
        .sorted()
        .forEach(k -> out.println("FRONTIER_CANDIDATE: " + k));
    out.flush();
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
      return null;
    }
    return switch (artifact) {
      case DerivedArtifact derived -> {
        ActionLookupKey owner = derived.getArtifactOwner();
        if (owner.getLabel() == null) {
          // These are various odd action owners, like that for the coverage report and the build
          // info artifacts. They aren't really useful to cache and they have some oddities (like
          // not depending on any source files at all) so they just cause trouble without anything
          // to show for it.
          yield null;
        }
        // Artifact#key is the canonical function to produce the SkyKey that will build this
        // artifact. We want to avoid serializing ordinary DerivedArtifacts, which are never built
        // by Skyframe directly, and the function will return ActionLookupData as the canonical key
        // for those artifacts instead.
        SkyKey artifactKey = Artifact.key(derived);
        if (artifactKey instanceof ActionLookupData) {
          yield null; // Handled independently.
        }
        yield artifactKey;
      }
      // Does not upload source artifacts because they are cheap to compute.
      case SourceArtifact ignored -> null;
    };
  }

  public static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setRemoteAnalysisCaching(RemoteAnalysisCaching.newBuilder().setCode(detailedCode))
        .build();
  }

  /**
   * Deletes select nodes and all rdeps from the graph.
   *
   * <p>This is not safe to call if the Skyframe graph needs to be incrementally correct after this
   * point.
   *
   * @return the number of nodes and rdeps deleted.
   */
  private static DeletionStats deleteNodesAndRdeps(
      InMemoryGraph graph, boolean skycacheAnalysisOnly) {
    AtomicLong deletedNodes = new AtomicLong();
    AtomicLong deletedRdeps = new AtomicLong();
    graph.parallelForEach(
        node -> {
          IncrementalInMemoryNodeEntry incrementalInMemoryNodeEntry =
              (IncrementalInMemoryNodeEntry) node;
          if (!skycacheAnalysisOnly && node.getKey() instanceof ActionLookupSummaryKey) {
            // skycacheAnalysisOnly mode requires ActionLookupSummaryKeys to remain in the graph.
            graph.remove(node.getKey());
            deletedNodes.incrementAndGet();
          } else {
            for (SkyKey rdep : incrementalInMemoryNodeEntry.getReverseDepsForDoneEntry()) {
              incrementalInMemoryNodeEntry.removeReverseDep(rdep);
              deletedRdeps.incrementAndGet();
            }
            incrementalInMemoryNodeEntry.consolidateReverseDeps();
          }
        });
    return new DeletionStats(deletedNodes.get(), deletedRdeps.get());
  }

  private record DeletionStats(long nodes, long rdeps) {}
}
