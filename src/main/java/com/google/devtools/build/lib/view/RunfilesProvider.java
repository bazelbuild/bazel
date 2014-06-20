// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.view.RunfilesCollector.State;

import java.util.HashMap;
import java.util.Map;

/**
 * A rule that provides runfiles.
 *
 * <p>Artifacts specified as runfiles are available during execution of the rule's actions.
 *
 * <p>Note that {@link RuleConfiguredTarget} already declares implementation of RunfilesProvider,
 * so if your rule inherits from {@code RuleConfiguredTarget}, you will not need an explicit
 * {@code implements} declaration.
 */
public interface RunfilesProvider extends TransitiveInfoProvider {
  /**
   * Returns transitive runfiles for this target given a particular runfiles collector state.
   *
   * @param state the state of the collector
   */
  Runfiles getTransitiveRunfiles(RunfilesCollector.State state);

  /**
   * A builder class for RunfilesProvider. This builder is used to avoid having references
   * to RuleContext objects after the target is created thus enforcing isolation and
   * reducing memory consumption.
   */
  public static final class Builder {
    
    private final HashMap<State, Runfiles> runfileMap = new HashMap<>();

    public Builder add(Runfiles runfiles, State state) {
      runfileMap.put(state, runfiles);
      return this;
    }

    public RunfilesProvider build() {
      for (State state : State.values()) {
        if (!runfileMap.containsKey(state)) {
          runfileMap.put(state, Runfiles.EMPTY);
        }
      }
      return new RunfilesProviderImpl(runfileMap);
    }

    /**
     * Builds the RunfilesProvider and fills the unspecified runfiles states
     * with the remainingRunfiles.
     */
    public Builder addRemaining(Runfiles remainingRunfiles) {
      for (State state : State.values()) {
        if (!runfileMap.containsKey(state)) {
          runfileMap.put(state, remainingRunfiles);
        }
      }
      return this;
    }
  }

  // TODO(bazel-team): there's another RunfilesProviderImpl in GenericRuleConfiguredTargetBuilder,
  // merge these two or better remove one. On the long run RunfilesProviderBuilder should be used
  // everywhere and the RunfilesCollector should be removed.
  /**
   * An implementation class for the RunfilesProvider.
   */
  public static final class RunfilesProviderImpl implements RunfilesProvider {

    private final ImmutableMap<State, Runfiles> runfileMap;

    public RunfilesProviderImpl(Map<State, Runfiles> runfileMap) {
      this.runfileMap = ImmutableMap.copyOf(runfileMap);
    }

    @Override
    public Runfiles getTransitiveRunfiles(State state) {
      return runfileMap.get(state);
    }

    /**
     * Creates an implementation class for RunfilesProvider.
     */
    public static RunfilesProvider dataSpecificRunfilesProvider(
        Runfiles defaultRunfiles, Runfiles dataRunfiles) {
      return new RunfilesProvider.Builder()
          .add(dataRunfiles, RunfilesCollector.State.DATA)
          .addRemaining(defaultRunfiles)
          .build();
    }
  }

  /**
   * An empty RunfilesProvider.
   */
  public static final RunfilesProvider EMPTY = new RunfilesProvider() {

    @Override
    public Runfiles getTransitiveRunfiles(State state) {
      return Runfiles.EMPTY;
    }
  };
}
