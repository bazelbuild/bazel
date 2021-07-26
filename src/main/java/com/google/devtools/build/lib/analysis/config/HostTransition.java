// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** Dynamic transition to the host configuration. */
public final class HostTransition implements PatchTransition {

  @AutoCodec public static final HostTransition INSTANCE = new HostTransition();

  private HostTransition() {}

  @Override
  public boolean isHostTransition() {
    return true;
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(CoreOptions.class);
  }

  @Override
  public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
    if (options.get(CoreOptions.class).isHost) {
      // If the input already comes from the host configuration, just return the existing values.
      //
      // We don't do this just for convenience: if an
      // {@link
      // com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy}
      // overrides option defaults, {@link FragmentOptions#getHost} won't honor that policy. That's
      // because it uses its own options parser that's not aware of the policy. This can create
      // problems for, e.g., {@link JavaOptions#getHost}, which promotes --host_foo flags to
      // --foo flags. That works the first time you do it (since both of the original values
      // were policy-processed). But not subsequent times.
      //
      // There are various ways to solve this problem (pass the policy to host options computation,
      // manually set host.hostFoo = original.hostFoo). But those raise larger questions about the
      // nature of host/target relationships, so for the time being this is a straightforward
      // and practical fix.
      return options.clone().underlying();
    } else {
      return options.underlying().createHostOptions();
    }
  }

  /** Returns a {@link TransitionFactory} instance that generates the host transition. */
  public static <T extends TransitionFactory.Data> TransitionFactory<T> createFactory() {
    return new AutoValue_HostTransition_Factory<>();
  }

  /**
   * Returns {@code true} if the given {@link TransitionFactory} is an instance of the host
   * transition.
   */
  public static <T extends TransitionFactory.Data> boolean isInstance(
      TransitionFactory<T> instance) {
    return instance instanceof Factory;
  }

  /** A {@link TransitionFactory} implementation that generates the host transition. */
  @AutoValue
  abstract static class Factory<T extends TransitionFactory.Data> implements TransitionFactory<T> {
    @Override
    public PatchTransition create(T unused) {
      return INSTANCE;
    }

    @Override
    public boolean isHost() {
      return true;
    }

    @Override
    public boolean isTool() {
      return true;
    }
  }
}
