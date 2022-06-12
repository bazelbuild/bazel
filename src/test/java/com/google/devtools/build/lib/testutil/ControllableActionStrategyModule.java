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
package com.google.devtools.build.lib.testutil;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;

/**
 * A {@link BlazeModule} that uses {@link SpawnController} to inject custom behavior.
 *
 * <p>The identifiers of strategies to make controllable are passed to the constructor. These
 * strategies are expected to already exist in the {@link SpawnStrategyRegistry.Builder} when {@link
 * #registerSpawnStrategies} is called, so the modules responsible for registering them should be
 * added to the runtime builder <em>before</em> the {@code ControllableActionStrategyModule}. Each
 * strategy corresponding to a specified identifier is replaced in the {@link
 * SpawnStrategyRegistry.Builder} with a {@linkplain SpawnController#wrap controllable wrapper}.
 */
public final class ControllableActionStrategyModule extends BlazeModule {

  private final SpawnController spawnController;
  private final ImmutableList<String> identifiers;

  public ControllableActionStrategyModule(SpawnController spawnController, String... identifiers) {
    checkArgument(identifiers.length > 0, "No identifiers given");
    this.spawnController = checkNotNull(spawnController);
    this.identifiers = ImmutableList.copyOf(identifiers);
  }

  @Override
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env)
      throws AbruptExitException {
    for (String identifier : identifiers) {
      SpawnStrategy delegate = registryBuilder.toStrategy(identifier, getClass().getSimpleName());
      registryBuilder.registerStrategy(spawnController.wrap(delegate), identifier);
    }
  }
}
