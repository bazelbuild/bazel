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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Executor.ActionContext;

import java.util.Map;

/**
 * An object describing that actions require a particular implementation of an
 * {@link ActionContext}.
 *
 * <p>This is expected to be implemented by modules that also implement actions which need these
 * contexts. Other modules will provide implementations for various action contexts by implementing
 * {@link ActionContextProvider}.
 *
 * <p>Example: a module requires {@code SpawnActionContext} to do its job, and it creates
 * actions with the mnemonic <code>C++</code>. Then the {@link #getSpawnActionContexts} method of
 * this module would return a map with the key <code>"C++"</code> in it.
 *
 * <p>The module can either decide for itself which implementation is needed and make the value
 * associated with this key a constant or defer that decision to the user, for example, by
 * providing a command line option and setting the value in the map based on that.
 *
 * <p>Other modules are free to provide different implementations of {@code SpawnActionContext}.
 * This can be used, for example, to implement sandboxed or distributed execution of
 * {@code SpawnAction}s in different ways, while giving the user control over how exactly they
 * are executed.
 */
public interface ActionContextConsumer {
  /**
   * Returns a map from spawn action mnemonics created by this module to the name of the
   * implementation of {@code SpawnActionContext} that the module wants to use for executing
   * it.
   *
   * <p>If a spawn action is executed whose mnemonic maps to the empty string or is not
   * present in the map at all, the choice of the implementation is left to Blaze.
   *
   * <p>Matching on mnemonics is done case-insensitively so it is recommended that any
   * implementation of this method makes sure that no two keys that refer to the same mnemonic are
   * present in the returned map. The easiest way to assure this is to use a map created using
   * {@code new TreeMap<>(String.CASE_INSENSITIVE_ORDER)}.
   */
  Map<String, String> getSpawnActionContexts();

  /**
   * Returns a map from action context class to the implementation required by the module.
   *
   * <p>If the implementation name is the empty string, the choice is left to Blaze.
   */
  Multimap<Class<? extends ActionContext>, String> getActionContexts();
}
