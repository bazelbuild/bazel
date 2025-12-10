// Copyright 2025 The Bazel Authors. All rights reserved.
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

package net.starlark.java.types;

import com.google.common.collect.ImmutableList;
import java.util.List;

/**
 * Base class for all Starlark types.
 *
 * <p>Starlark typing is an experimental feature under development. See the tracking issue:
 * https://github.com/bazelbuild/bazel/issues/27370
 */
public abstract class StarlarkType {

  /**
   * Returns the list of supertypes of this type.
   *
   * <p>Preferred order is from the most specific to the least specific supertype. But if that is
   * not possible, the order can be arbitrary.
   */
  // TODO: #27370 - Add getSubtypes(), with the semantics that the actual subtype relation is the
  // union of these two methods.
  public List<StarlarkType> getSupertypes() {
    return ImmutableList.of();
  }
}
