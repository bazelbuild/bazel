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
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking.ACTIVE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking.FRONTIER_CANDIDATE;
import static java.util.concurrent.ForkJoinPool.commonPool;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching.Code;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectBaseKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

/**
 * Implements frontier serialization with pprof dumping using {@code --serialized_frontier_profile}.
 */
public final class FrontierSerializer {

  private FrontierSerializer() {}

  /**
   * Serializes the frontier contained in the current Skyframe graph into a {@link ProfileCollector}
   * writing the resulting proto to {@code path}.
   *
   * @return empty if successful, otherwise a result containing the appropriate error
   */
  public static Optional<FailureDetail> dumpFrontierSerializationProfile(
      Supplier<ObjectCodecs> codecsSupplier,
      SkyframeExecutor skyframeExecutor,
      PathFragmentPrefixTrie matcher,
      FingerprintValueService fingerprintValueService,
      Reporter reporter,
      String path)
      throws InterruptedException {
    // Starts initializing ObjectCodecs in a background thread as it can take some time.
    var futureCodecs = new FutureTask<>(codecsSupplier::get);
    commonPool().execute(futureCodecs);

    var stopwatch = new ResettingStopwatch(Stopwatch.createStarted());
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();

    ConcurrentHashMap<ActionLookupKey, SelectionMarking> selection =
        computeSelection(graph, matcher);
    reporter.handle(
        Event.info(
            String.format("Found %d active or frontier keys in %s", selection.size(), stopwatch)));

    var profileCollector = new ProfileCollector();
    ObjectCodecs codecs;
    try {
      codecs = futureCodecs.get();
    } catch (ExecutionException e) {
      // No exceptions are expected here.
      throw new IllegalStateException("failed to initialize ObjectCodecs", e.getCause());
    }
    if (codecs == null) {
      String message = "serialization not supported";
      reporter.error(null, message);
      return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
    }

    reporter.handle(Event.info(String.format("Initializing codecs took %s\n", stopwatch)));

    var writeStatuses = Collections.synchronizedList(new ArrayList<ListenableFuture<Void>>());
    AtomicInteger frontierValueCount = new AtomicInteger();
    selection.forEach(
        /* parallelismThreshold= */ 0,
        (actionLookupKey, marking) -> {
          if (!marking.equals(FRONTIER_CANDIDATE)) {
            return;
          }
          try {
            SerializationResult<ByteString> keyBytes =
                codecs.serializeMemoizedAndBlocking(
                    fingerprintValueService, actionLookupKey, profileCollector);
            var keyWriteStatus = keyBytes.getFutureToBlockWritesOn();
            if (keyWriteStatus != null) {
              writeStatuses.add(keyWriteStatus);
            }

            InMemoryNodeEntry node = checkNotNull(graph.getIfPresent(actionLookupKey));
            SerializationResult<ByteString> valueBytes =
                codecs.serializeMemoizedAndBlocking(
                    fingerprintValueService, node.getValue(), profileCollector);
            var writeStatusFuture = valueBytes.getFutureToBlockWritesOn();
            if (writeStatusFuture != null) {
              writeStatuses.add(writeStatusFuture);
            }
            frontierValueCount.getAndIncrement();
          } catch (SerializationException e) {
            writeStatuses.add(immediateFailedFuture(e));
          }
        });

    reporter.handle(
        Event.info(
            String.format(
                "Serialized %s frontier entries in %s\n", frontierValueCount, stopwatch)));

    try {
      var unusedNull =
          Futures.whenAllSucceed(writeStatuses).call(() -> null, directExecutor()).get();
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
        Event.info(String.format("Waiting for write futures took an additional %s\n", stopwatch)));

    try (var fileOutput = new FileOutputStream(path);
        var bufferedOutput = new BufferedOutputStream(fileOutput)) {
      profileCollector.toProto().writeTo(bufferedOutput);
    } catch (IOException e) {
      String message = "Error writing serialization profile to file: " + e.getMessage();
      reporter.error(null, message, e);
      return Optional.of(createFailureDetail(message, Code.SERIALIZED_FRONTIER_PROFILE_FAILED));
    }
    return Optional.empty();
  }

  @VisibleForTesting
  enum SelectionMarking {
    /**
     * The entry is marked as a frontier candidate.
     *
     * <p>If a node is still a frontier candidate at the end of the selection process, it is a
     * frontier node and should be serialized.
     */
    FRONTIER_CANDIDATE,
    /** The node is in the active set and will not be serialized. */
    ACTIVE
  }

  @VisibleForTesting
  static ConcurrentHashMap<ActionLookupKey, SelectionMarking> computeSelection(
      InMemoryGraph graph, PathFragmentPrefixTrie matcher) {
    var selection = new ConcurrentHashMap<ActionLookupKey, SelectionMarking>();
    graph.parallelForEach(
        node -> {
          if (!(node.getKey() instanceof ActionLookupKey actionLookupKey)) {
            return;
          }
          Label label = actionLookupKey.getLabel();
          if (label == null) {
            return;
          }
          if (!matcher.includes(label.getPackageFragment())) {
            return;
          }
          markActiveAndTraverseEdges(graph, actionLookupKey, selection);
        });
    return selection;
  }

  private static void markActiveAndTraverseEdges(
      InMemoryGraph graph,
      ActionLookupKey root,
      ConcurrentHashMap<ActionLookupKey, SelectionMarking> selection) {
    Label label = root.getLabel();
    if (label == null) {
      return;
    }
    if (selection.put(root, ACTIVE) == ACTIVE) {
      return;
    }
    if (root instanceof AspectBaseKey aspectKey) {
      // Whenever an aspect is marked active, its base configured target must also be marked active.
      // This avoids a situation where an aspect inspects a deserialized configured target, which
      // may crash because the configured target doesn't have its actions.
      //
      // This is possible when an aspect is in the UTC of an active node via an attribute label.
      markActiveAndTraverseEdges(graph, aspectKey.getBaseConfiguredTargetKey(), selection);
    }
    InMemoryNodeEntry node = checkNotNull(graph.getIfPresent(root), root);
    for (SkyKey dep : node.getDirectDeps()) {
      if (!(dep instanceof ActionLookupKey child)) {
        continue;
      }
      selection.putIfAbsent(child, FRONTIER_CANDIDATE);
    }
    for (SkyKey rdep : node.getReverseDepsForDoneEntry()) {
      if (!(rdep instanceof ActionLookupKey parent)) {
        continue;
      }
      markActiveAndTraverseEdges(graph, parent, selection);
    }
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
        .setRemoteAnalysisCaching(
            FailureDetails.RemoteAnalysisCaching.newBuilder().setCode(detailedCode))
        .build();
  }
}
