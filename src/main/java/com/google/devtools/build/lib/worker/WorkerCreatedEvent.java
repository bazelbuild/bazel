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

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/** An event fired during execution, when worker was destroyed. */
public final class WorkerCreatedEvent implements Postable {
  private final int workerPoolHash;
  private final String mnemonic;

  public WorkerCreatedEvent(int workerPoolHash, String mnemonic) {
    this.workerPoolHash = workerPoolHash;
    this.mnemonic = mnemonic;
  }

  public String getMnemonic() {
    return mnemonic;
  }

  public int getWorkerPoolHash() {
    return workerPoolHash;
  }
}
