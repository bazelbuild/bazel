// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.Path;

import org.apache.commons.pool2.impl.GenericKeyedObjectPool;
import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

import java.util.HashSet;
import java.util.Set;

import javax.annotation.concurrent.ThreadSafe;

/**
 * A worker pool that spawns multiple workers and delegates work to them.
 *
 * <p>This is useful when the worker cannot handle multiple parallel requests on its own and we need
 * to pre-fork a couple of them instead.
 */
@ThreadSafe
final class WorkerPool extends GenericKeyedObjectPool<WorkerKey, Worker> {
  final WorkerFactory workerFactory;
  final Set<Worker> workers = new HashSet<>();

  public WorkerPool(WorkerFactory factory, GenericKeyedObjectPoolConfig config) {
    super(factory, config);
    this.workerFactory = factory;
  }

  public void setLogDirectory(Path logDir) {
    this.workerFactory.setLogDirectory(logDir);
  }

  public void setReporter(Reporter reporter) {
    this.workerFactory.setReporter(reporter);
  }

  public void setVerbose(boolean verbose) {
    this.workerFactory.setVerbose(verbose);
  }
}
