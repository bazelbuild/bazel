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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** A lease service that manages the lease of remote blobs. */
public class LeaseService {
  private final MemoizingEvaluator memoizingEvaluator;
  @Nullable private final ActionCache actionCache;
  private final AtomicBoolean leaseExtensionStarted = new AtomicBoolean(false);
  @Nullable LeaseExtension leaseExtension;

  public LeaseService(
      MemoizingEvaluator memoizingEvaluator,
      @Nullable ActionCache actionCache,
      @Nullable LeaseExtension leaseExtension) {
    this.memoizingEvaluator = memoizingEvaluator;
    this.actionCache = actionCache;
    this.leaseExtension = leaseExtension;
  }

  public void finalizeAction() {
    if (leaseExtensionStarted.compareAndSet(false, true)) {
      if (leaseExtension != null) {
        leaseExtension.start();
      }
    }
  }

  public void finalizeExecution(Set<ActionInput> missingActionInputs) {
    if (leaseExtension != null) {
      leaseExtension.stop();
    }

    if (!missingActionInputs.isEmpty()) {
      handleMissingInputs();
    }
  }

  /**
   * An interface whose implementations extend the leases of remote outputs referenced by skyframe.
   */
  public interface LeaseExtension {
    void start();

    void stop();
  }

  /** Clean up internal state when files are evicted from remote CAS. */
  private void handleMissingInputs() {
    // If any outputs are evicted, remove all remote metadata from skyframe and local action cache.
    //
    // With TTL based discarding and lease extension, remote cache eviction error won't happen if
    // remote cache can guarantee the TTL. However, if it happens, it usually means the remote cache
    // is under high load and it could possibly evict more blobs that Bazel wouldn't aware of.
    // Following builds could still fail for the same error (caused by different blobs).
    memoizingEvaluator.delete(
        key -> {
          if (key.functionName().equals(SkyFunctions.ACTION_EXECUTION)) {
            try {
              var value = memoizingEvaluator.getExistingValue(key);
              return value instanceof ActionExecutionValue
                  && isRemote((ActionExecutionValue) value);
            } catch (InterruptedException ignored) {
              return false;
            }
          }
          return false;
        });

    if (actionCache != null) {
      actionCache.removeIf(
          entry -> !entry.getOutputFiles().isEmpty() || !entry.getOutputTrees().isEmpty());
    }
  }

  private boolean isRemote(ActionExecutionValue value) {
    return value.getAllFileValues().values().stream().anyMatch(FileArtifactValue::isRemote)
        || value.getAllTreeArtifactValues().values().stream().anyMatch(this::isRemoteTree);
  }

  private boolean isRemoteTree(TreeArtifactValue treeArtifactValue) {
    return treeArtifactValue.getChildValues().values().stream()
            .anyMatch(FileArtifactValue::isRemote)
        || treeArtifactValue
            .getArchivedRepresentation()
            .map(ar -> ar.archivedFileValue().isRemote())
            .orElse(false);
  }
}
