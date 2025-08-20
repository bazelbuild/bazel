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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A registry of {@link BlazeService} implementations.
 *
 * <p>A {@link BlazeService} is identified by the canonical name of the service class. At most one
 * implementation is supported for any given service class.
 *
 * <p>Usages:
 *
 * <pre>{@code
 * BlazeServiceRegistry registry =
 *     BlazeServiceRegistry.builder()
 *         .register(FooService.class, new FooServiceImpl())
 *         .register(BarService.class, new BarServiceImpl())
 *         .build();
 *
 * // ...
 *
 * FooService fooService = registry.get(FooService.class);
 * BarService barService = registry.get(BarService.class);
 * }</pre>
 */
public final class BlazeServiceRegistry {
  private final ImmutableMap<String, BlazeService> services;

  private BlazeServiceRegistry(ImmutableMap<String, BlazeService> services) {
    this.services = services;
  }

  public static Builder builder() {
    return new Builder();
  }

  @SuppressWarnings("unchecked")
  @Nullable
  public <T extends BlazeService> T get(Class<T> service) {
    return (T) services.get(getServiceKey(service));
  }

  private static String getServiceKey(Class<? extends BlazeService> service) {
    // Use the canonical name of the service interface class so we get a stable identifier that
    // won't change across different versions of LC and SC.
    return service.getCanonicalName();
  }

  /** Builder for {@link BlazeServiceRegistry}. */
  public static class Builder {
    private final Map<String, BlazeService> services = Maps.newHashMap();

    /**
     * Registers a service implementation for the given service class.
     *
     * <p>A service is identified by the canonical name of the service class. It is an error to
     * register the service with the same identifier more than once.
     */
    @CanIgnoreReturnValue
    public Builder register(Class<? extends BlazeService> service, BlazeService impl) {
      var key = getServiceKey(service);
      checkState(
          !services.containsKey(key),
          "At most one %s is supported, but found two: %s and %s",
          key,
          services.get(key),
          impl);
      services.put(key, impl);
      return this;
    }

    public BlazeServiceRegistry build() {
      return new BlazeServiceRegistry(ImmutableMap.copyOf(services));
    }
  }
}
