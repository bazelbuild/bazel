// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Digest;

/**
 * A key in the remote action cache. The type wraps around a {@link Digest} of an {@link Action}.
 * Action keys are special in that they aren't content-addressable but refer to action results.
 *
 * <p>Terminology note: "action" is used here in the remote execution protocol sense, which is
 * equivalent to a Bazel "spawn" (a Bazel "action" being a higher-level concept).
 */
public record ActionKey(Digest digest) {
  public ActionKey {
    checkNotNull(digest, "digest");
  }
}
