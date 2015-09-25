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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.collect.ImmutableCollection;

/**
 * An expander that converts a nested set into a flattened collection.
 *
 * <p>Expanders are initialized statically (there is one for each order), so they should
 * contain no state and all methods must be threadsafe.
 */
interface NestedSetExpander<E> {
  /**
   * Flattens the NestedSet into the builder.
   */
  void expandInto(NestedSet<E> nestedSet, Uniqueifier uniqueifier,
      ImmutableCollection.Builder<E> builder);
}
