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

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.concurrent.ExecutionException;

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
      boolean emitUploadedEvents) {
    var fileOpNodes =
        new FileOpNodeMemoizingLookup(
            fingerprintValueService.getExecutor(),
            graph,
            /* selectedKeys= */ ImmutableSet.of(),
            /* shouldDiscardMemory= */ false,
            /* referencedPackages= */ null);

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
    writeStatuses.selectedEntryStartingCapped();
    try {
      if (key instanceof ActionLookupKey analysisKey) {
        // This is an analysis-phase entry, we need the direct deps of this entry, which is not
        // committed to Skyframe yet, so we pluck them out of the environment of the SkyFunction
        selectedEntrySerializer.uploadAnalysisEntry(
            analysisKey, value, env.getTemporaryDirectDeps().getAllElementsAsIterable());
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

  public void waitForCompletion() throws InterruptedException, ExecutionException {
    writeStatuses.notifyAllStarted();
    var unused = writeStatuses.get();
  }
}
