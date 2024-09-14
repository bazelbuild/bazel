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

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** An {@link com.google.common.eventbus.EventBus} listener for remote analysis caching events. */
@ThreadSafety.ThreadSafe
public class RemoteAnalysisCachingEventListener {

  /**
   * An event for when a Skyframe node has been serialized, but its associated write futures (i.e.
   * RPC latency) may not be done yet.
   */
  public record SerializedNodeEvent(SkyKey key) {
    public SerializedNodeEvent {
      checkNotNull(key);
    }
  }

  private final Set<SkyKey> serializedKeys = ConcurrentHashMap.newKeySet();

  @Subscribe
  @AllowConcurrentEvents
  @SuppressWarnings("unused")
  public void onSerializationComplete(SerializedNodeEvent event) {
    serializedKeys.add(event.key());
  }

  /** Returns the counts of {@link SkyFunctionName} from serialized nodes of this invocation. */
  public Multiset<SkyFunctionName> getSkyfunctionCounts() {
    Multiset<SkyFunctionName> counts = HashMultiset.create();
    serializedKeys.forEach(key -> counts.add(key.functionName()));
    return counts;
  }

  /** Returns the count of serialized nodes of this invocation. */
  public int getSerializedKeysCount() {
    return serializedKeys.size();
  }
}
