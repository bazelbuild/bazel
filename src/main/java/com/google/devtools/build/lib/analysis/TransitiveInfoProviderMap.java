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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import javax.annotation.concurrent.Immutable;

/**
 * Provides a mapping between an identifier for transitive information and its instance. (between
 * provider identifier and provider instance)
 *
 * <p>We have three kinds of provider identifiers:
 *
 * <ul>
 *   <li>Declared providers. They are exposed to Skylark and identified by {@link Provider.Key}.
 *       Provider instances are {@link Info}s.
 *   <li>Native providers. They are identified by their {@link Class} and their instances are
 *       instances of that class. They should implement {@link TransitiveInfoProvider} marker
 *       interface.
 *   <li>Legacy Skylark providers (deprecated). They are identified by simple strings, and their
 *       instances are more-less random objects.
 * </ul>
 */
@Immutable
public interface TransitiveInfoProviderMap extends ProviderCollection {
  /**
   * Returns a count of providers.
   *
   * Upper bound for {@code index} in {@link #getProviderKeyAt(int index)}
   * and {@link #getProviderInstanceAt(int index)} }.
   *
   * Low-level method, use with care.
   */
  int getProviderCount();

  /**
   * Return value is one of:
   *
   * <ul>
   *   <li>{@code Class<? extends TransitiveInfoProvider>}
   *   <li>String
   *   <li>{@link Provider.Key}
   * </ul>
   *
   * Low-level method, use with care.
   */
  Object getProviderKeyAt(int index);

  Object getProviderInstanceAt(int index);
}
