// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.network;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.network.ConnectivityStatus.Status;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.common.options.Converters.DurationConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;

/** Stores network status for Bazel-adjacent services, usually remote build and BES. */
public class ConnectivityModule extends BlazeModule implements ConnectivityStatusProvider {

  /** Options that define the behavior of the Connectivity Modules. */
  public static class ConnectivityOptions extends OptionsBase {
    @Option(
        name = "connectivity_check_frequency",
        defaultValue = "5s",
        documentationCategory = OptionDocumentationCategory.REMOTE,
        effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
        converter = DurationConverter.class,
        help =
            "How often to perform a connectivity and reachability check for remote services "
                + "(eg. 5s). 0 disables the check and assumes connectivity.")
    public Duration cacheLifetime;
  }

  /**
   * Attempts to retrieve and return the specified service's status from the cache, calculating it
   * if it's not present in the cache.
   *
   * @param service the name of the service we want to determine connectivity status for
   * @param cache a cache that stores the current connectivity status for each service
   * @return the connectivity status for the specified service
   */
  protected ConnectivityStatus determineConnectivity(String service, Cache<String, Status> cache) {
    return new ConnectivityStatus(Status.OK, /* serviceInfo= */ "");
  }

  private Duration cacheLifetime;
  private Cache<String, Status> cache;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(ConnectivityOptions.class)
        : ImmutableList.of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    ConnectivityOptions options = env.getOptions().getOptions(ConnectivityOptions.class);
    if (options == null) {
      return;
    }
    Duration newCacheLifetime = options.cacheLifetime;
    // Initialize the cache if we haven't yet, or if the options have changed.
    // TODO(steinman): Make this a LoadingCache where load() calls determineConnectivity().
    if (cache == null || !newCacheLifetime.equals(cacheLifetime)) {
      cache = CacheBuilder.newBuilder().expireAfterWrite(newCacheLifetime).build();
      cacheLifetime = newCacheLifetime;
    }
  }

  @Override
  public ConnectivityStatus getStatus(String service) {
    if (cacheLifetime.isZero()) {
      return new ConnectivityStatus(Status.OK, /* serviceInfo= */ "");
    }
    return determineConnectivity(service, cache);
  }
}
