// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A provider that wraps a transitive info provider map.
 *
 * <p>Useful for wrapping providers in aspects in a general way where there would otherwise be a
 * conflict.
 */
public interface WrappingProvider extends TransitiveInfoProvider {
  TransitiveInfoProviderMap getTransitiveInfoProviderMap();

  /**
   * Helper methods for {@link WrappingProvider}.
   *
   * <p>This class can be removed once Java 8 is supported and we can put static methods on
   * interfaces.
   */
  class Helper {

    /** Gets the actual providers from a list of wrappers. */
    public static <T extends TransitiveInfoProvider> List<T> unwrapProviders(
        Iterable<? extends WrappingProvider> wrappingProviders, Class<T> providerClass) {
      List<T> result = new ArrayList<>();
      for (WrappingProvider wrappingProvider : wrappingProviders) {
        T provider = wrappingProvider.getTransitiveInfoProviderMap().getProvider(providerClass);
        if (provider != null) {
          result.add(provider);
        }
      }
      return result;
    }

    /** Reads a provider via a wrapping provider. Returns null if not wrapped. */
    @Nullable
    public static <T extends TransitiveInfoProvider, U extends WrappingProvider>
        T getWrappedProvider(
            TransitiveInfoCollection collection,
            Class<U> wrappingProviderClass,
            Class<T> providerClass) {
      WrappingProvider wrappingProvider = collection.getProvider(wrappingProviderClass);
      if (wrappingProvider != null) {
        return wrappingProvider.getTransitiveInfoProviderMap().getProvider(providerClass);
      }
      return null;
    }
  }
}
