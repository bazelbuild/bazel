// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.metrics;

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Collects metrics and is posted over the event bus after a build.
 */
@ThreadSafe
public class RemoteMetrics implements Postable {

  private AtomicLong totalBytesSent = new AtomicLong();
  private AtomicLong totalBytesReceived = new AtomicLong();

  public void bytesSent(long bytes) {
    totalBytesSent.addAndGet(bytes);
  }

  public void bytesReceived(long bytes) {
    totalBytesReceived.addAndGet(bytes);
  }

  public long totalBytesReceived() {
    return totalBytesReceived.get();
  }

  public long totalBytesSent() {
    return totalBytesSent.get();
  }
}
