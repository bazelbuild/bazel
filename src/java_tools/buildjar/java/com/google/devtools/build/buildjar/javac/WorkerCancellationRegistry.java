// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import java.util.concurrent.ConcurrentHashMap;

/** A helper class that keeps track of requests to be cancelled. */
public class WorkerCancellationRegistry {

  /** the map keeps track of requests that should be cancelled. */
  private final ConcurrentHashMap<Integer, Integer> cancelledRequests = new ConcurrentHashMap<>();

  public boolean checkIfRequestIsCancelled(int requestId) {
    return cancelledRequests.containsKey(requestId);
  }

  public void registerRequest(Integer requestId) {
    cancelledRequests.put(requestId, requestId);
  }

  public void unregisterRequest(Integer requestId) {
    cancelledRequests.remove(requestId, requestId);
  }
}
