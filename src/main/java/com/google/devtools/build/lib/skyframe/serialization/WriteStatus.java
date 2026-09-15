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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skybridge.SkybridgeInterface;

/**
 * Represents future success or failure of a write operation.
 *
 * <p>This can act like an ordinary future, but has special case, memory saving handling for
 * aggregation.
 *
 * <p>The {@link Boolean} result of this future indicates the "novelty" of the write. A {@code true}
 * result means new bytes were actually written to the storage backend; {@code false} means they
 * were already present. Novelty tracking is used for metrics and defaults to {@code true} if the
 * backend configuration doesn't support it.
 *
 * <p>OR semantics are used for aggregation: an aggregate is novel if any of its components are
 * novel.
 */
// The ImmediateWriteStatus class should be singleton, so it's cleaner to not derive it from
// AbstractFuture.
@SuppressWarnings("ShouldNotSubclass")
@SkybridgeInterface
public interface WriteStatus extends ListenableFuture<Boolean> {}
