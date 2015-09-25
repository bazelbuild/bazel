// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.Path;

import org.apache.commons.pool2.BaseKeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

/**
 * Factory used by the pool to create / destroy / validate worker processes.
 */
final class WorkerFactory extends BaseKeyedPooledObjectFactory<WorkerKey, Worker> {
  private Path logDir;
  private Reporter reporter;
  private boolean verbose;

  public void setLogDirectory(Path logDir) {
    this.logDir = logDir;
  }

  public void setReporter(Reporter reporter) {
    this.reporter = reporter;
  }

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  @Override
  public Worker create(WorkerKey key) throws Exception {
    return Worker.create(key, logDir, reporter, verbose);
  }

  /**
   * Use the DefaultPooledObject implementation.
   */
  @Override
  public PooledObject<Worker> wrap(Worker worker) {
    return new DefaultPooledObject<>(worker);
  }

  /**
   * When a worker process is discarded, destroy its process, too.
   */
  @Override
  public void destroyObject(WorkerKey key, PooledObject<Worker> p) throws Exception {
    if (verbose) {
      reporter.handle(
          Event.info(
              "Destroying "
                  + key.getMnemonic()
                  + " worker (id "
                  + p.getObject().getWorkerId()
                  + ")."));
    }
    p.getObject().destroy();
  }

  /**
   * The worker is considered to be valid when its process is still alive.
   */
  @Override
  public boolean validateObject(WorkerKey key, PooledObject<Worker> p) {
    return p.getObject().isAlive();
  }
}
