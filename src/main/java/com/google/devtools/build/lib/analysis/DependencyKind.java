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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ExecGroup;
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

    @Nullable
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

  /**
   * Represents a dependency on toolchain context whether it's the entity (target or aspect) owned
   * toolchain or the base target toolchain in case of aspects.
   */
  interface ToolchainDependencyKind extends DependencyKind {
    @Override
    public default Attribute getAttribute() {
      return null;
    }

    @Nullable
    @Override
    public default AspectClass getOwningAspect() {
      throw new IllegalStateException();
    }

    /** The name of the execution group represented by this dependency kind. */
    public String getExecGroupName();

    /** Returns true if this toolchain dependency is for the default exec group. */
    public boolean isDefaultExecGroup();
  }

  /**
   * A dependency of an entity (target or aspect) on a toolchain context, identified by the
   * execution group name.
   */
  @AutoValue
  abstract class ToolchainDependencyKindImpl implements ToolchainDependencyKind {}

  /**
   * A dependency for the aspect on its target's toolchain context, used for aspects propagating to
   * toolchains, identified by the execution group name and the toolchain type.
   */
  @AutoValue
  abstract class BaseTargetToolchainDependencyKind implements ToolchainDependencyKind {
    /** The toolchain type of the toolchain dependency. */
    public abstract Label getToolchainType();
  }

  /** Returns a {@link DependencyKind} for the given execution group. */
  static DependencyKind forExecGroup(String execGroupName) {
    if (ExecGroup.DEFAULT_EXEC_GROUP_NAME.equals(execGroupName)) {
      return defaultExecGroupToolchain();
    }
    return new AutoValue_DependencyKind_ToolchainDependencyKindImpl(execGroupName, false);
  }

  /** Returns a {@link DependencyKind} for the default execution group. */
  static DependencyKind defaultExecGroupToolchain() {
    return new AutoValue_DependencyKind_ToolchainDependencyKindImpl(
        ExecGroup.DEFAULT_EXEC_GROUP_NAME, true);
  }

  /** Returns a {@link DependencyKind} for the given execution group. */
  static DependencyKind forBaseTargetExecGroup(String execGroupName, Label toolchainType) {
    return new AutoValue_DependencyKind_BaseTargetToolchainDependencyKind(
        execGroupName, execGroupName.equals(ExecGroup.DEFAULT_EXEC_GROUP_NAME), toolchainType);
  }

  /** Predicate to check if a dependency represents an aspect's base target toolchain. */
  static boolean isBaseTargetToolchain(DependencyKind dependencyKind) {
    return dependencyKind instanceof BaseTargetToolchainDependencyKind;
  }

  /** Predicate to check if a dependency represents a toolchain. */
  static boolean isToolchain(DependencyKind dependencyKind) {
    return dependencyKind instanceof ToolchainDependencyKind;
  }

  /** Predicate to check if a dependency represents an attribute dependency. */
  static boolean isAttribute(DependencyKind dependencyKind) {
    return dependencyKind instanceof AttributeDependencyKind;
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
