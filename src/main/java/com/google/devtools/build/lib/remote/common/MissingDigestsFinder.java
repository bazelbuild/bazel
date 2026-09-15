// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;

/** Supports querying a remote cache whether it contains a list of blobs. */
public interface MissingDigestsFinder {

  /**
   * Returns a set of digests that the remote cache does not know about. The returned set is
   * guaranteed to be a subset of {@code digests}.
   *
   * @param digests The list of digests to look for.
   */
  ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests);
}
