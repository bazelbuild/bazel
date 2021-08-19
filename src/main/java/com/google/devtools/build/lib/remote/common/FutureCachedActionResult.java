// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import build.bazel.remote.execution.v2.ActionResult;

/**
 */
@Immutable
@ThreadSafe
public class FutureCachedActionResult {
  private ListenableFuture<ActionResult> actionResult;
  private String cacheName;

  private FutureCachedActionResult(ListenableFuture<ActionResult> actionResult, String cacheName) {
    this.actionResult = actionResult;
    this.cacheName = cacheName;
  }

  public static FutureCachedActionResult fromDisk(ListenableFuture<ActionResult> actionResult) {
    return new FutureCachedActionResult(actionResult, "disk");
  }
  public static FutureCachedActionResult fromRemote(ListenableFuture<ActionResult> actionResult) {
    return new FutureCachedActionResult(actionResult, "remote");
  }

  public ListenableFuture<ActionResult> getFutureAction() {
    return actionResult;
  }

  public String getCacheName() {
    return cacheName;
  }
}
