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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsCache;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.errorprone.annotations.Immutable;

/**
 * An abstract configuration transition that sets the Python version as per its {@link
 * #determineNewVersion} method, if transitioning is allowed.
 *
 * <p>See {@link PythonOptions#canTransitionPythonVersion} for information on when transitioning is
 * allowed.
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

  private PythonVersionTransition() {}

  /**
   * Returns the Python version to transition to, given the configuration.
   *
   * <p>Must return a target Python version ({@code PY2} or {@code PY3}).
   *
   * <p>Caution: This method must not modify {@code options}. See the class javadoc for {@link
   * PatchTransition}.
   */
  protected abstract PythonVersion determineNewVersion(BuildOptionsView options);

  // We added this cache after observing an O(100,000)-node build graph that applied multiple exec
  // transitions to Python 3 tools on every node. Before this cache, this produced O(100,000)
  // BuildOptions instances that consumed about a gigabyte of memory.
  private static final BuildOptionsCache<PythonVersion> cache =
      new BuildOptionsCache<>(PythonVersionTransition::transitionImpl);

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(PythonOptions.class);
  }

  @Override
  public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
    PythonVersion newVersion = determineNewVersion(options);
    checkArgument(newVersion.isTargetValue(), newVersion);

    // PythonOptions aren't present after NoConfigTransition. That implies rules that don't read
    // configuration and don't produce build actions. The only time those rules trigger this code
    // is in ExecutionTool.createConvenienceSymlinks.
    PythonOptions opts =
        options.underlying().hasNoConfig() ? null : options.get(PythonOptions.class);
    if (opts == null || !opts.canTransitionPythonVersion(newVersion)) {
      return options.underlying();
    }
    return cache.applyTransition(options, newVersion);
  }

  private static BuildOptions transitionImpl(BuildOptionsView options, PythonVersion newVersion) {
    BuildOptionsView newOptions = options.clone();
    PythonOptions newOpts = newOptions.get(PythonOptions.class);
    newOpts.setPythonVersion(newVersion);
    return newOptions.underlying();
  }

  /** A Python version transition that switches to the value specified in the constructor. */
  private static final class ToConstant extends PythonVersionTransition {

    private final PythonVersion newVersion;

    ToConstant(PythonVersion newVersion) {
      this.newVersion = checkNotNull(newVersion);
    }

    @Override
    protected PythonVersion determineNewVersion(BuildOptionsView options) {
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
      return 37 * ToConstant.class.hashCode() + newVersion.hashCode();
    }
  }

  /** A Python version transition that switches to the default given in the Python configuration. */
  private static final class ToDefault extends PythonVersionTransition {

    private static final ToDefault INSTANCE = new ToDefault();

    // Singleton.
    private ToDefault() {}

    @Override
    protected PythonVersion determineNewVersion(BuildOptionsView options) {
      return options.get(PythonOptions.class).getDefaultPythonVersion();
    }
  }
}
