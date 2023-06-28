// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;

/**
 * Transitions to a stable, empty configuration for rules that don't rely on configuration.
 *
 * <p>This prevents unnecessary configured target forking, which prevents unnecessary build graph
 * bloat. That in turn reduces build time and memory use.
 *
 * <p>For example, imagine {@code cc_library //:foo} in config A depends on config-independent
 * target {@code //:noconfig} and {@code cc_library //:bar} in config B also depends on {@code
 * //:noconfig}. Without transitions, {@code //:noconfig} will be configured and analyzed twice: for
 * configs A and B. This is completely wasteful if {@code //:noconfig} does the same thing
 * regardless of configuration. Instead, apply this transition to {@code //:noconfig}.
 *
 * <p>This is safest for rules that don't produce actions and don't have dependencies. Remember that
 * even if a rule doesn't read configuration, if any of its transitive dependencies read
 * configuration or if the rule has a {@code select()}, its output may still be
 * configuration-dependent. So use with careful discretion.
 */
public class NoConfigTransition implements PatchTransition {

  @SerializationConstant public static final NoConfigTransition INSTANCE = new NoConfigTransition();

  public static final BuildOptions NO_CONFIG_OPTIONS = BuildOptions.builder().build();

  private NoConfigTransition() {}

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(CoreOptions.class);
  }

  @Override
  public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
    // Ideally the build options should be empty: no fragment options and no flags. But core Bazel
    // code assumes CoreOptions exists. For example CoreOptions.check_visibility is required for
    // basic configured target graph evaluation. So we provide CoreOptions with default values
    // (not inherited from parent configuration). This means flags like --check_visibility may not
    // be consistently applied. If this becomes a problem in practice we can carve out exceptions
    // to flags like that to propagate.
    // TODO(bazel-team): break out flags that configure Bazel's analysis phase into their own
    // FragmentOptions and propagate them to this configuration. Those flags should also be
    // ineligible outputs for other transitions because they're not meant for rule logic.  That
    // would guarantee consistency of flags like --check_visibility while still preventing forking.
    return BuildOptions.builder()
        .addFragmentOptions(options.get(CoreOptions.class).getDefault())
        .build();
  }

  /** Returns a {@link TransitionFactory} instance that generates the transition. */
  public static <T extends TransitionFactory.Data> TransitionFactory<T> createFactory() {
    return new AutoValue_NoConfigTransition_Factory<>();
  }

  /** A {@link TransitionFactory} implementation that generates the transition. */
  @AutoValue
  abstract static class Factory<T extends TransitionFactory.Data> implements TransitionFactory<T> {
    @Override
    public PatchTransition create(T unused) {
      return INSTANCE;
    }

    @Override
    public boolean isTool() {
      return true;
    }
  }
}
