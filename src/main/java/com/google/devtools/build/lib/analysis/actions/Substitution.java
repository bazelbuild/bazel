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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nonnull;
import net.starlark.java.eval.EvalException;

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

  public abstract String getValue() throws EvalException;

  /* Not intended for use in production code */
  // TODO(hvd): migrate usages and delete
  @VisibleForTesting
  public final String getValueUnchecked() {
    try {
      return getValue();
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  private static final class StringSubstitution extends Substitution {
    private final String key;
    private final String value;

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

  private static final class ListSubstitution extends Substitution {
    private final String key;
    private final ImmutableList<?> value;

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
  public static Substitution of(@Nonnull final String key, @Nonnull final String value) {
    Preconditions.checkNotNull(key);
    Preconditions.checkNotNull(value);
    return new StringSubstitution(key, value);
  }

  /**
   * Returns an immutable Substitution instance for the key and list of values. The values will be
   * joined by spaces before substitution.
   */
  public static Substitution ofSpaceSeparatedList(
      @Nonnull final String key, @Nonnull final ImmutableList<?> value) {
    Preconditions.checkNotNull(key);
    Preconditions.checkNotNull(value);
    return new ListSubstitution(key, value);
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (object instanceof Substitution) {
      Substitution substitution = (Substitution) object;
      return substitution.getKey().equals(this.getKey())
          && substitution.getValueUnchecked().equals(this.getValueUnchecked());
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(getKey(), getValueUnchecked());
  }

  @Override
  public String toString() {
    return "Substitution(" + getKey() + " -> " + getValueUnchecked() + ")";
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

    public ComputedSubstitution(@Nonnull String key) {
      Preconditions.checkNotNull(key);
      this.key = key;
    }

    @Override
    public String getKey() {
      return key;
    }
  }
}
