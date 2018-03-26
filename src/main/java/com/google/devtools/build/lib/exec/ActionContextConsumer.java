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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.exec.SpawnActionContextMaps.Builder;

/**
 * An object describing that actions require a particular implementation of an {@link
 * ActionContext}.
 *
 * <p>This is expected to be implemented by modules that also implement actions which need these
 * contexts. Other modules will provide implementations for various action contexts by implementing
 * {@link ActionContextProvider}.
 *
 * <p>Example: a module requires {@code SpawnActionContext} to do its job, and it creates actions
 * with the mnemonic <code>C++</code>. Then the {@link #populate(Builder)} method of this module
 * would put <code>("C++", strategy)</code> in the map returned by {@link
 * Builder#strategyByMnemonicMap()}.
 *
 * <p>The module can either decide for itself which implementation is needed and make the value
 * associated with this key a constant or defer that decision to the user, for example, by providing
 * a command line option and setting the value in the map based on that.
 *
 * <p>Other modules are free to provide different implementations of {@code SpawnActionContext}.
 * This can be used, for example, to implement sandboxed or distributed execution of {@code
 * SpawnAction}s in different ways, while giving the user control over how exactly they are
 * executed.
 *
 * <p>Example: if a module requires {@code MyCustomActionContext} to be available, but doesn't
 * associate it with any strategy, its {@link #populate(Builder)} should add {@code
 * (MyCustomActionContext.class, "")} to the builder's {@link Builder#strategyByContextMap}.
 *
 * <p>Example: if a module requires {@code MyLocalCustomActionContext} to be available, and wants it
 * to always use the "local" strategy, its {@link #populate(Builder)} should add {@code
 * (MyCustomActionContext.class, "local")} to the builder's {@link Builder#strategyByContextMap}. .
 */
public interface ActionContextConsumer {
  /**
   * Provides a {@link SpawnActionContextMaps.Builder} instance which modules may use to configure
   * the {@link ActionContext} instances the module requires for particular actions.
   */
  void populate(SpawnActionContextMaps.Builder builder);
}
