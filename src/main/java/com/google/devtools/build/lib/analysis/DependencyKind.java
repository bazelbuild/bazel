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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
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

  /** A dependency for visibility. */
  DependencyKind VISIBILITY_DEPENDENCY = new NonAttributeDependencyKind("VISIBILITY");

  /** The dependency on the rule that creates a given output file. */
  DependencyKind OUTPUT_FILE_RULE_DEPENDENCY = new NonAttributeDependencyKind("OUTPUT_FILE");

  /** A dependency on a resolved toolchain. */
  DependencyKind TOOLCHAIN_DEPENDENCY = new NonAttributeDependencyKind("TOOLCHAIN");

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

  /** A dependency caused by something that's not an attribute. Special cases enumerated below. */
  final class NonAttributeDependencyKind implements DependencyKind {
    private final String name;

    private NonAttributeDependencyKind(String name) {
      this.name = name;
    }

    @Override
    public Attribute getAttribute() {
      return null;
    }

    @Nullable
    @Override
    public AspectClass getOwningAspect() {
      throw new IllegalStateException();
    }

    @Override
    public String toString() {
      return String.format("%s(%s)", getClass().getSimpleName(), this.name);
    }
  }

  /** A dependency through an attribute, either that of an aspect or the rule itself. */
  @AutoValue
  abstract class AttributeDependencyKind implements DependencyKind {
    @Override
    public abstract Attribute getAttribute();

    @Override
    @Nullable
    public abstract AspectClass getOwningAspect();

    public static AttributeDependencyKind forRule(Attribute attribute) {
      return new AutoValue_DependencyKind_AttributeDependencyKind(attribute, null);
    }

    public static AttributeDependencyKind forAspect(Attribute attribute, AspectClass owningAspect) {
      return new AutoValue_DependencyKind_AttributeDependencyKind(
          attribute, Preconditions.checkNotNull(owningAspect));
    }
  }
}
