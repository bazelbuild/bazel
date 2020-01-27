// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsCache;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.errorprone.annotations.Immutable;
import java.util.Objects;

/**
 * An abstract configuration transition that sets the Python version as per its {@link
 * #determineNewVersion} method, if transitioning is allowed.
 *
 * <p>See {@link PythonOptions#canTransitionPythonVersion} for information on when transitioning is
 * allowed.
 *
 * <p>Subclasses should override {@link #determineNewVersion}, as well as {@link #equals} and {@link
 * #hashCode}.
 */
@Immutable
public abstract class PythonVersionTransition implements PatchTransition {

  /** Returns a transition that sets the version to {@code newVersion}. */
  public static PythonVersionTransition toConstant(PythonVersion newVersion) {
    return new ToConstant(newVersion);
  }

  /**
   * Returns a transition that sets the version to {@link PythonOptions#getDefaultPythonVersion}.
   */
  public static PythonVersionTransition toDefault() {
    return ToDefault.INSTANCE;
  }

  /**
   * Returns the Python version to transition to, given the configuration.
   *
   * <p>Must return a target Python version ({@code PY2} or {@code PY3}).
   *
   * <p>Caution: This method must not modify {@code options}. See the class javadoc for {@link
   * PatchTransition}.
   */
  protected abstract PythonVersion determineNewVersion(BuildOptions options);

  @Override
  public abstract boolean equals(Object other);

  @Override
  public abstract int hashCode();

  // We added this cache after observing an O(100,000)-node build graph that applied multiple exec
  // transitions to Python 3 tools on every node. Before this cache, this produced O(100,000)
  // BuildOptions instances that consumed about a gigabyte of memory.
  private static final BuildOptionsCache<PythonVersion> cache = new BuildOptionsCache<>();

  @Override
  public BuildOptions patch(BuildOptions options) {
    PythonVersion newVersion = determineNewVersion(options);
    Preconditions.checkArgument(newVersion.isTargetValue());

    PythonOptions opts = options.get(PythonOptions.class);
    if (!opts.canTransitionPythonVersion(newVersion)) {
      return options;
    }
    return cache.applyTransition(
        options,
        newVersion,
        () -> {
          BuildOptions newOptions = options.clone();
          PythonOptions newOpts = newOptions.get(PythonOptions.class);
          newOpts.setPythonVersion(newVersion);
          return newOptions;
        });
  }

  /** A Python version transition that switches to the value specified in the constructor. */
  private static class ToConstant extends PythonVersionTransition {

    private final PythonVersion newVersion;

    public ToConstant(PythonVersion newVersion) {
      this.newVersion = newVersion;
    }

    @Override
    protected PythonVersion determineNewVersion(BuildOptions options) {
      return newVersion;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof ToConstant)) {
        return false;
      }
      return newVersion.equals(((ToConstant) other).newVersion);
    }

    @Override
    public int hashCode() {
      return Objects.hash(ToConstant.class, newVersion);
    }
  }

  /** A Python version transition that switches to the default given in the Python configuration. */
  private static class ToDefault extends PythonVersionTransition {

    private static final ToDefault INSTANCE = new ToDefault();

    // Singleton.
    private ToDefault() {}

    @Override
    protected PythonVersion determineNewVersion(BuildOptions options) {
      return options.get(PythonOptions.class).getDefaultPythonVersion();
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof ToDefault;
    }

    @Override
    public int hashCode() {
      // Avoid varargs array allocation by using hashCode() rather than hash().
      return Objects.hashCode(ToDefault.class);
    }
  }
}
