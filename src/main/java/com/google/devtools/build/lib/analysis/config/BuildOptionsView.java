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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.base.Preconditions;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Wrapper for {@link BuildOptions} that only permits {@link BuildOptions#get} calls to {@link
 * FragmentOptions} from a pre-declared set.
 *
 * <p>This lets Blaze understand what fragments transitions require, which is helpful for
 * understanding what flags are and are not important to a given build.
 */
public class BuildOptionsView implements Cloneable {
  private final BuildOptions options;
  private final Set<Class<? extends FragmentOptions>> allowedFragments;

  /** Wraps a given {@link BuildOptions} with a "permitted" set of {@link FragmentOptions}. */
  public BuildOptionsView(
      BuildOptions options, Set<Class<? extends FragmentOptions>> allowedFragments) {
    this.options = options;
    this.allowedFragments = allowedFragments;
  }

  /**
   * Wrapper for {@link BuildOptions#get} that throws an {@link IllegalArgumentException} if the
   * given {@link FragmentOptions} isn't in the "permitted" set.
   */
  @Nullable
  public <T extends FragmentOptions> T get(Class<T> optionsClass) {
    return options.get(checkFragment(optionsClass));
  }

  /**
   * Wrapper for {@link BuildOptions#contains} that throws an {@link IllegalArgumentException} if
   * the given {@link FragmentOptions} isn't in the "permitted" set.
   */
  public boolean contains(Class<? extends FragmentOptions> optionsClass) {
    return options.contains(checkFragment(optionsClass));
  }

  /**
   * Returns a new {@link BuildOptionsView} instance bound to a clone of the original's {@link
   * BuildOptions}.
   */
  @Override
  public BuildOptionsView clone() {
    return new BuildOptionsView(options.clone(), allowedFragments);
  }

  /**
   * Returns the underlying {@link BuildOptions}.
   *
   * <p>Since this sheds all extra security from {@link BuildOptionsView}, this should only be used
   * when a transition is returning its final result.
   *
   * <p>!!! No transition should call any {@link BuildOptions} accessor after this! !!!
   */
  public BuildOptions underlying() {
    return options;
  }

  private <T extends FragmentOptions> Class<T> checkFragment(Class<T> optionsClass) {
    Preconditions.checkArgument(
        allowedFragments.contains(optionsClass),
        "Can't access %s in allowed fragments %s",
        optionsClass,
        allowedFragments);
    return optionsClass;
  }
}
