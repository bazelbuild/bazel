// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.util.SerializedAbruptExitException;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.util.Collections;

/**
 * Provides a piece of functionality in the Service Component (SC).
 *
 * <p>Each service must be structured as an interface type (which extends this interface) and an
 * implementation type (which implements the interface type). The interface type provides a stable
 * API through which the Logical Component (LC) can access the service, whose implementation is
 * provided by the SC.
 *
 * <p>The set of services is passed into {@link BlazeRuntime#main} and fixed for the lifetime of the
 * server. A service can be obtained by calling {@link BlazeRuntime#getBlazeService} with the
 * interface type as the argument.
 */
public interface BlazeService extends OptionsSupplier {

  @Override
  default Iterable<Class<? extends OptionsBase>> getStartupOptions() {
    return Collections.emptyList();
  }

  @Override
  default Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return Collections.emptyList();
  }

  @Override
  default Iterable<Class<? extends OptionsBase>> getCommandOptions(String commandName) {
    return Collections.emptyList();
  }

  /**
   * Called at the beginning of Bazel startup, right before {@link BlazeModule#globalInit}.
   *
   * @param startupOptions the server's startup options
   * @param blazeServices the available services, including this service itself
   * @throws SerializedAbruptExitException to shut down the server immediately
   */
  default void globalInit(OptionsProvider startupOptions, Iterable<BlazeService> blazeServices)
      throws SerializedAbruptExitException {}
}
