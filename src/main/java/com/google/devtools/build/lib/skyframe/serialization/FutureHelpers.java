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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;

/** Helpers for serialization futures. */
public final class FutureHelpers {

  /** Combines a list of {@code Void} futures into a single future. */
  static ListenableFuture<Void> aggregateStatusFutures(List<ListenableFuture<Void>> futures) {
    if (futures.size() == 1) {
      return futures.get(0);
    }
    return Futures.whenAllSucceed(futures).call(() -> null, directExecutor());
  }

  private FutureHelpers() {}
}
