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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.stream.Collectors;

/**
 * A pair of a string to be substituted and a string to substitute it with. For simplicity, these
 * are called key and value. All implementations must be immutable, and always return the identical
 * key. The returned values must be the same, though they need not be the same object.
 *
 * <p>It should be assumed that the {@link #getKey} invocation is cheap, and that the {@link
 * #getValue} invocation is expensive.
 */
@Immutable // if the keys and values in the passed in lists and maps are all immutable
public abstract class Substitution {
  private Substitution() {}

  public abstract String getKey();

  public abstract String getValue();

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class StringSubstitution extends Substitution {
    private final String key;
    private final String value;

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    StringSubstitution(String key, String value) {
      this.key = key;
      this.value = value;
    }

    @Override
    public String getKey() {
      return key;
    }

    @Override
    public String getValue() {
      return value;
    }
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class ListSubstitution extends Substitution {
    private final String key;
    private final ImmutableList<?> value;

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    ListSubstitution(String key, ImmutableList<?> value) {
      this.key = key;
      this.value = value;
    }

    @Override
    public String getKey() {
      return key;
    }

    @Override
    public String getValue() {
      return Joiner.on(" ").join(value);
    }
  }
  /** Returns an immutable Substitution instance for the given key and value. */
  public static Substitution of(final String key, final String value) {
    return new StringSubstitution(key, value);
  }

  /**
   * Returns an immutable Substitution instance for the key and list of values. The values will be
   * joined by spaces before substitution.
   */
  public static Substitution ofSpaceSeparatedList(final String key, final ImmutableList<?> value) {
    return new ListSubstitution(key, value);
  }

  /**
   * Returns an immutable Substitution instance for the key and list of values. The values will be
   * joined by the given string before substitution.
   */
  public static Substitution ofJoinedShortPaths(
      String key, ImmutableList<Artifact> artifacts, String joinStr) {
    return new JoinedArtifactListShortPathSubstitution(key, artifacts, joinStr);
  }

  /**
   * Returns an immutable Substitution instance for the key and list of values. The values will be
   * joined by the given string before substitution.
   */
  public static Substitution ofJoinedShortPaths(
      String key, NestedSet<Artifact> artifacts, String joinStr) {
    return new JoinedArtifactNestedSetShortPathSubstitution(key, artifacts, joinStr);
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (object instanceof Substitution) {
      Substitution substitution = (Substitution) object;
      return substitution.getKey().equals(this.getKey())
          && substitution.getValue().equals(this.getValue());
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(getKey(), getValue());
  }

  @Override
  public String toString() {
    return "Substitution(" + getKey() + " -> " + getValue() + ")";
  }

  /**
   * A substitution with a fixed key, and a computed value. The computed value must not change over
   * the lifetime of an instance, though the {@link #getValue} method may return different String
   * objects.
   *
   * <p>It should be assumed that the {@link #getKey} invocation is cheap, and that the {@link
   * #getValue} invocation is expensive.
   */
  public abstract static class ComputedSubstitution extends Substitution {
    private final String key;

    public ComputedSubstitution(String key) {
      this.key = key;
    }

    @Override
    public String getKey() {
      return key;
    }
  }

  /**
   * Expands a fragment value.
   *
   * <p>This is slightly more memory efficient since it defers the expansion of the path fragment's
   * string until requested. Often a template action is never executed, meaning the string is never
   * needed.
   */
  @AutoCodec
  public static final class PathFragmentSubstitution extends ComputedSubstitution {
    private final PathFragment pathFragment;

    public PathFragmentSubstitution(String key, PathFragment pathFragment) {
      super(key);
      this.pathFragment = pathFragment;
    }

    @Override
    public String getValue() {
      return pathFragment.getPathString();
    }
  }

  /**
   * Expands a label value to its canonical string value.
   *
   * <p>This is more memory efficient than directly using the {@Label#toString}, since that method
   * constructs a new string every time it's called.
   */
  @AutoCodec
  public static final class LabelSubstitution extends ComputedSubstitution {
    private final Label label;

    public LabelSubstitution(String key, Label label) {
      super(key);
      this.label = label;
    }

    @Override
    public String getValue() {
      return label.getCanonicalForm();
    }
  }

  /**
   * Expands a collection of artifacts to their short (root relative paths).
   *
   * <p>This is much more memory efficient than eagerly joining them into a string.
   */
  @AutoCodec
  public static final class JoinedArtifactListShortPathSubstitution extends ComputedSubstitution {
    private final ImmutableList<Artifact> artifacts;
    private final String joinStr;

    @AutoCodec.Instantiator
    public JoinedArtifactListShortPathSubstitution(
        String key, ImmutableList<Artifact> artifacts, String joinStr) {
      super(key);
      this.artifacts = artifacts;
      this.joinStr = joinStr;
    }

    @Override
    public String getValue() {
      return artifacts.stream()
          .map(artifact -> artifact.getRootRelativePath().getPathString())
          .collect(Collectors.joining(joinStr));
    }
  }

  /**
   * Expands a collection of artifacts to their short (root relative paths).
   *
   * <p>This is much more memory efficient than eagerly joining them into a string.
   */
  @AutoCodec
  public static final class JoinedArtifactNestedSetShortPathSubstitution
      extends ComputedSubstitution {
    private final NestedSet<Artifact> artifacts;
    private final String joinStr;

    @AutoCodec.Instantiator
    public JoinedArtifactNestedSetShortPathSubstitution(
        String key, NestedSet<Artifact> artifacts, String joinStr) {
      super(key);
      this.artifacts = artifacts;
      this.joinStr = joinStr;
    }

    @Override
    public String getValue() {
      return artifacts.toList().stream()
          .map(artifact -> artifact.getRootRelativePath().getPathString())
          .collect(Collectors.joining(joinStr));
    }
  }
}
