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
package com.google.devtools.build.skyframe;

import java.io.Serializable;

/**
 * A {@link SkyKey} is effectively a pair (type, name) that identifies a Skyframe value.
 *
 * <p>SkyKey implementations are heavily used as map keys. Thus, they should have fast {@link
 * #hashCode} implementations (cached if necessary). The same SkyKey may be created multiple times
 * by different {@code SkyFunction}s requesting it, and so it should have effective interning. There
 * will likely be more SkyKeys on the JVM heap than any other non-native type, so be mindful of
 * memory usage (in particular object wrapper size and memory alignment)! Typically the
 * implementation should have a fixed {@link #functionName} implementation and return itself as the
 * {@link #argument} in order to reduce the cost of wrapper objects.
 */
public interface SkyKey extends Serializable {
  SkyFunctionName functionName();

  default Object argument() {
    return this;
  }

  /**
   * Returns {@code true} if this key produces a {@link SkyValue} that can be reused across builds.
   *
   * <p>Values may be unshareable because they are just not serializable, or because they contain
   * data that cannot safely be reused as-is by another invocation, such as stamping information or
   * "flaky" values like test statuses.
   *
   * <p>Unshareable data should not be serialized, since it will never be reused. Attempts to fetch
   * a key's serialized data will call this method and only perform the fetch if it returns {@code
   * true}.
   *
   * <p>The result of this method only applies to non-error values. In case of an error, {@link
   * ErrorInfo#isTransitivelyTransient()} can be used to determine shareability.
   */
  default boolean valueIsShareable() {
    return true;
  }
}
