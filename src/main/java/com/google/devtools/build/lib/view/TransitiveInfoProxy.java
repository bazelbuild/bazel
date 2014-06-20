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

import com.google.devtools.build.lib.syntax.Label;

/**
 * A utility class to construct wrappers for TransitiveInfoProviders.
 *
 * <p>To enable the proxy, set {@link Options#extendedSanityChecks} to true.
 */
public class TransitiveInfoProxy {

  /**
   * Returns a proxied TransitiveInfoCollection which is produced by serializing the
   * given {@code infoCollection} and then deserializing the result.
   */
  public static TransitiveInfoCollection createCollectionProxy(
      final EnumerableTransitiveInfoCollection collection) {
    return new TransitiveInfoCollection() {
      @Override
      public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerType) {
        return collection.getProvider(providerType);
      }

      @Override
      public Label getLabel() {
        return collection.getLabel();
      }
    };
  }
}
