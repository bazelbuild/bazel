// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.Attribute;
import javax.annotation.Nullable;

/**
 * A kind of dependency, used extensively in {@link DependencyResolver}.
 *
 * <p>Usually an attribute, but other special-cased kinds exist, for example, for visibility or
 * toolchains.
 */
public interface DependencyKind {

  /**
   * The attribute through which a dependency arises.
   *
   * <p>Returns {@code null} for visibility, the dependency pointing from an output file to its
   * generating rule and toolchain dependencies.
   */
  @Nullable
  Attribute getAttribute();

  /**
   * The aspect owning the attribute through which the dependency arises.
   *
   * <p>Should only be called for dependency kinds representing an attribute.
   */
  @Nullable
  AspectClass getOwningAspect();
}
