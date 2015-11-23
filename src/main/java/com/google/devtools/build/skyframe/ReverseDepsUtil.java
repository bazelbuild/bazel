// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;

import java.util.Collection;

/**
 * A utility interface for dealing with reverse deps in NodeEntry and BuildingState implementations.
 *
 * <p>The reason for this interface is to abstract out dealing with reverse deps, which is subject
 * to various non-trivial and fairly ugly memory/performance optimizations.
 *
 * <p>This interface is public only for the benefit of alternative graph implementations outside of
 * the package.
 */
public interface ReverseDepsUtil<T> {
  void addReverseDeps(T container, Collection<SkyKey> reverseDeps);
  /**
   * Checks that the reverse dependency is not already present. Implementations may choose not to
   * perform this check, or only to do it if reverseDeps is small, so that it does not impact
   * performance.
   */
  void maybeCheckReverseDepNotPresent(T container, SkyKey reverseDep);

  void checkReverseDep(T container, SkyKey reverseDep);

  void removeReverseDep(T container, SkyKey reverseDep);

  ImmutableSet<SkyKey> getReverseDeps(T container);

  String toString(T container);
}
