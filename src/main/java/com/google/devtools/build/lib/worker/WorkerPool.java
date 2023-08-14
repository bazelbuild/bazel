// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import java.io.IOException;
import org.apache.commons.pool2.impl.EvictionPolicy;

/**
 * A worker pool that spawns multiple workers and delegates work to them. Allows separate
 * configuration for singleplex and multiplex workers. While the configuration is per mnemonic, the
 * actual pools need to be per WorkerKey, as different WorkerKeys may imply different process
 * startup options.
 *
 * <p>This is useful when the worker cannot handle multiple parallel requests on its own and we need
 * to pre-fork a couple of them instead. Multiplex workers <em>can</em> handle multiple parallel
 * requests, but do so through WorkerProxy instances.
 */
public interface WorkerPool {

  int getMaxTotalPerKey(WorkerKey key);

  int getNumActive(WorkerKey key);

  // TODO (b/242835648) filter throwed exceptions better
  void evictWithPolicy(EvictionPolicy<Worker> evictionPolicy) throws InterruptedException;

  Worker borrowObject(WorkerKey key) throws IOException, InterruptedException;

  void returnObject(WorkerKey key, Worker obj);

  void invalidateObject(WorkerKey key, Worker obj) throws InterruptedException;

  void setDoomedWorkers(ImmutableSet<Integer> workerIds);

  void clearDoomedWorkers();

  void setEventBus(EventBus eventBus);

  void close();
}
