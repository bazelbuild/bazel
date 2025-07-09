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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructProvider;
import java.util.HashMap;
import net.starlark.java.eval.EvalException;

/** Helper static methods for handling ExtraLinkTimeLibraries. */
public final class StarlarkDefinedLinkTimeLibrary {

  /** The Builder interface builds an ExtraLinkTimeLibrary. */
  /** Merge the ExtraLinkTimeLibrary based on the inputs. */
  public static StarlarkInfo merge(ImmutableList<StarlarkInfo> libraries) {
    HashMap<String, ImmutableList.Builder<Depset>> depsetMapBuilder = new HashMap<>();
    HashMap<String, Object> constantsMap = new HashMap<>();

    for (StarlarkInfo library : libraries) {
      for (String key : library.getFieldNames()) {
        Object value = library.getValue(key);
        if (value instanceof Depset depset) {
          depsetMapBuilder.computeIfAbsent(key, k -> ImmutableList.builder()).add(depset);
        } else {
          constantsMap.put(key, value);
        }
      }
    }

    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    for (String key : depsetMapBuilder.keySet()) {
      try {
        builder.put(
            key,
            Depset.fromDirectAndTransitive(
                Order.LINK_ORDER,
                ImmutableList.of(),
                depsetMapBuilder.get(key).build(),
                /* strict= */ true));
      } catch (EvalException e) {
        // should never happen; exception comes from bad order argument.
        throw new IllegalStateException(e);
      }
    }
    builder.putAll(constantsMap);
    // Note that we're returning Struct instead of the right provider. This situation will be
    // rectified once this code is rewritten to Starlark.
    return StructProvider.STRUCT.create(builder.buildOrThrow(), "");
  }

  private StarlarkDefinedLinkTimeLibrary() {}
}
