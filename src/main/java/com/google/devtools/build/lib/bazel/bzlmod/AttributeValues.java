// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.Maps;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;

/** Wraps a dictionary of attribute names and values. Always uses a dict to represent them */
@AutoValue
@GenerateTypeAdapter
public abstract class AttributeValues {

  public static AttributeValues create(Dict<String, Object> attribs) {
    return new AutoValue_AttributeValues(attribs);
  }

  public static AttributeValues create(Map<String, Object> attribs) {
    return new AutoValue_AttributeValues(
        Dict.immutableCopyOf(Maps.transformValues(attribs, AttributeValues::valueToStarlark)));
  }

  public abstract Dict<String, Object> attributes();

  // TODO(salmasamy) this is a copy of Attribute::valueToStarlark, Maybe think of a better place?
  private static Object valueToStarlark(Object x) {
    // Is x a non-empty string_list_dict?
    if (x instanceof Map<?, ?> map) {
      if (!map.isEmpty() && map.values().iterator().next() instanceof List) {
        Dict.Builder<Object, Object> dict = Dict.builder();
        for (Map.Entry<?, ?> e : map.entrySet()) {
          dict.put(e.getKey(), Starlark.fromJava(e.getValue(), null));
        }
        return dict.buildImmutable();
      }
    }
    // For all other attribute values, shallow conversion is safe.
    return Starlark.fromJava(x, null);
  }
}
