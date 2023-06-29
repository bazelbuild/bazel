// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.util.TestType;
import java.util.Objects;

/**
 * A startup flag to decide whether some {@link com.google.devtools.build.lib.cmdline.Label}s will
 * use {@link com.google.devtools.build.lib.cmdline.LabelInterner} backed by {@link
 * com.google.devtools.build.lib.concurrent.PooledInterner} or just the regular bazel weak interner.
 *
 * <p>When this flag is true, {@code LabelInterner} will be applied for all {@code Label}s so that
 * they are able to switch between interning the instances between the regular bazel weak interner
 * and the static global pool.
 *
 * <p>Applying {@code LabelInterner} can reduce memory overhead of having duplicate {@code Label}
 * instances in both weak interner and {@link InMemoryGraphImpl}.
 */
// TODO(b/250641010): This flag is temporary to facilitate a controlled rollout. So it should be
//  removed after the new pooled interning for `Label` instances is fully released and stable.
public final class UsePooledLabelInterningFlag {

  private static final boolean USE_POOLED_LABEL_INTERNER =
      Objects.equals(System.getProperty("BAZEL_USE_POOLED_LABEL_INTERNER"), "1")
          || TestType.isInTest();

  public static boolean usePooledLabelInterningFlag() {
    return USE_POOLED_LABEL_INTERNER;
  }

  private UsePooledLabelInterningFlag() {}
}
