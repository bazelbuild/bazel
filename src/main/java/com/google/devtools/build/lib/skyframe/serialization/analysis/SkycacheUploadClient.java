// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.serialization.ErrorMessageHelper.getErrorMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupSummaryKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.GroupedDeps;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Client for uploading to Skycache during computation. */
public final class SkycacheUploadClient {

  private final SelectedEntrySerializer selectedEntrySerializer;
  private final SelectedEntrySerializer.SerializationStatus writeStatuses;

  public SkycacheUploadClient(
      FingerprintValueService fingerprintValueService,
      ObjectCodecs codecs,
      FrontierNodeVersion frontierNodeVersion,
      InMemoryGraph graph,
      EventBus eventBus,
      LongVersionGetter versionGetter,
      boolean emitUploadedEvents,
      FileOpNodeMemoizingLookup fileOpNodes) {

    var fileDependencySerializer =
        new FileDependencySerializer(
            versionGetter,
            graph,
            fingerprintValueService,
            fingerprintValueService.getExecutor(),
            /* profileCollector= */ null);
    this.writeStatuses =
        new SelectedEntrySerializer.SerializationStatus(fileDependencySerializer.getCounters());
    var serializationStats = new SelectedEntrySerializer.SerializationStats();

    this.selectedEntrySerializer =
        new SelectedEntrySerializer(
            graph,
            codecs,
            frontierNodeVersion,
            fingerprintValueService,
            fileOpNodes,
            fileDependencySerializer,
            writeStatuses,
            /* shouldDiscardMemory= */ false,
            /* packageRefcounts= */ null,
            eventBus,
            /* profileCollector= */ null,
            serializationStats,
            emitUploadedEvents);
  }

  public void tryUpload(SkyKey key, SkyValue value, SkyFunction.Environment env)
      throws InterruptedException {
    if (getLabel(key) == null) {
      return;
    }
    writeStatuses.selectedEntryStartingCapped();
    try {
      if (key instanceof ActionLookupKey analysisKey) {
        // This is an analysis-phase entry, we need the direct deps of this entry, which is not
        // committed to Skyframe yet, so we pluck them out of the environment of the SkyFunction
        GroupedDeps temporaryDirectDeps = env.getTemporaryDirectDeps();
        Set<SkyKey> newlyRequestedDeps = env.getNewlyRequestedDeps();
        ImmutableList<SkyKey> deps =
            temporaryDirectDeps.isEmpty()
                ? ImmutableList.copyOf(newlyRequestedDeps)
                : ImmutableList.<SkyKey>builderWithExpectedSize(
                        temporaryDirectDeps.numElements() + newlyRequestedDeps.size())
                    .addAll(temporaryDirectDeps.getAllElementsAsIterable())
                    .addAll(newlyRequestedDeps)
                    .build();
        selectedEntrySerializer.uploadAnalysisEntry(analysisKey, value, deps);
      } else {
        // This is an execution-phase entry. We need the deps of its owner, which should be
        // available
        // in the graph because, well, this entry depends on that one. The value we want to upload
        // is still not available in the graph, so we pass it in.
        selectedEntrySerializer.uploadExecutionEntry(key, value);
      }
    } catch (MissingSkyframeEntryException e) {
      writeStatuses.selectedEntryFailed(e);
    } catch (Throwable t) {
      writeStatuses.selectedEntryFailed(t);
    }
  }

  @Nullable
  private static Label getLabel(SkyKey key) {
    return switch (key) {
      case ActionLookupKey alk -> alk.getLabel();
      case ActionLookupData ald -> ald.getLabel();
      case ActionLookupSummaryKey summaryKey -> summaryKey.argument().getLabel();
      case Artifact artifact -> artifact.getOwnerLabel();
      default -> null;
    };
  }

  public void waitForCompletion() throws InterruptedException, ExecutionException {
    writeStatuses.notifyAllStarted();
    ImmutableList<Throwable> errors = writeStatuses.get();
    if (errors.isEmpty()) {
      return;
    }
    String message = "Skycache upload failed: " + getErrorMessage(errors);
    FailureDetail detail =
        FailureDetail.newBuilder()
            .setMessage(message)
            .setRemoteAnalysisCaching(
                RemoteAnalysisCaching.newBuilder()
                    .setCode(RemoteAnalysisCaching.Code.UPLOAD_FAILED))
            .build();
    throw new ExecutionException(
        new AbruptExitException(DetailedExitCode.of(detail), errors.get(0)));
  }
}
