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

import java.util.Objects;

/**
 * A startup flag to decide whether some {@link SkyKey}s will use {@code SkyKeyInterner} or just the
 * regular bazel weak interner.
 *
 * <p>When this flag is true, {@code SkyKeyInterner} will be applied for some {@link SkyKey}s so
 * that they are able to switch between interning the instances between the regular bazel weak
 * interner and the static global pool.
 *
 * <p>Applying {@code SkyKeyInterner} can reduce memory overhead of having duplicate {@link SkyKey}
 * instances in both weak interner and some other storage.
 */
// TODO(b/250641010): This flag is temporary to facilitate a controlled rollout. So it should be
//  removed after the new pooled interning is fully released and stable.
public final class UsePooledSkyKeyInterningFlag {

  private static final boolean USE_POOLED_SKY_KEY_INTERNER =
      Objects.equals(System.getProperty("BAZEL_USE_POOLED_SKY_KEY_INTERNER"), "1");

  public static boolean usePooledSkyKeyInterningFlag() {
    return USE_POOLED_SKY_KEY_INTERNER;
  }

  private UsePooledSkyKeyInterningFlag() {}
}
