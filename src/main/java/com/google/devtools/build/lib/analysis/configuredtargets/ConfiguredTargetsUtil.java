// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.configuredtargets;

import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;

/** Utility methods for configured gargets. */
public final class ConfiguredTargetsUtil {

  private ConfiguredTargetsUtil() {}

  /**
   * Returns a Dict of provider names to their values for a configured target.
   *
   * <p>This map is intended to be used from Starlark query output methods, so all values must be
   * accessible in Starlark. If the value of a provider is not convertible to a Starlark value, that
   * name/value pair is left out of the map.
   */
  public static Dict<String, Object> getProvidersDict(
      AbstractConfiguredTarget target, TransitiveInfoProviderMap providers) {
    Dict.Builder<String, Object> res = Dict.builder();
    for (int i = 0; i < providers.getProviderCount(); i++) {
      // The key may be of many types, but we need a string for the intended use.
      Object key = providers.getProviderKeyAt(i);
      Object v = providers.getProviderInstanceAt(i);
      String keyAsString;
      if (key instanceof String) {
        keyAsString = key.toString();
      } else if (key instanceof Provider.Key) {
        if (key instanceof StarlarkProvider.Key) {
          StarlarkProvider.Key k = (StarlarkProvider.Key) key;
          keyAsString = k.getExtensionLabel().toString() + "%" + k.getExportedName();
        } else {
          keyAsString = key.toString();
        }
      } else if (key instanceof Class) {
        keyAsString = ((Class) key).getSimpleName();
      } else {
        // ???
        continue;
      }
      try {
        res.put(keyAsString, Starlark.fromJava(v, null));
      } catch (IllegalArgumentException ex) {
        // This is OK. If this is not a valid StarlarkValue, we just leave it out of the map.
      }
    }
    return res.buildImmutable();
  }
}
