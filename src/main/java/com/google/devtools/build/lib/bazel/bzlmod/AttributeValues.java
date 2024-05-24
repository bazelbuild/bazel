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

import static java.util.Collections.singletonList;

import com.google.auto.value.AutoValue;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
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

  public static void validateAttrs(AttributeValues attributes, String what) throws EvalException {
    for (var entry : attributes.attributes().entrySet()) {
      validateSingleAttr(entry.getKey(), entry.getValue(), what);
    }
  }

  public static void validateSingleAttr(String attrName, Object attrValue, String what)
      throws EvalException {
    var maybeNonVisibleLabel = getFirstNonVisibleLabel(attrValue);
    if (maybeNonVisibleLabel.isEmpty()) {
      return;
    }
    Label label = maybeNonVisibleLabel.get();
    String repoName = label.getRepository().getName();
    throw Starlark.errorf(
        "no repository visible as '@%s' to the %s, but referenced by label '@%s//%s:%s' in"
            + " attribute '%s' of %s. Is the %s missing a bazel_dep or use_repo(..., \"%s\")?",
        repoName,
        label.getRepository().getOwnerRepoDisplayString(),
        repoName,
        label.getPackageName(),
        label.getName(),
        attrName,
        what,
        label.getRepository().getOwnerModuleDisplayString(),
        repoName);
  }

  private static Optional<Label> getFirstNonVisibleLabel(Object nativeAttrValue) {
    Collection<?> toValidate =
        switch (nativeAttrValue) {
          case List<?> list -> list;
          case Map<?, ?> map -> map.keySet();
          case null, default -> singletonList(nativeAttrValue);
        };
    for (var item : toValidate) {
      if (item instanceof Label label && !label.getRepository().isVisible()) {
        return Optional.of(label);
      }
    }
    return Optional.empty();
  }

  // TODO(salmasamy) this is a copy of Attribute::valueToStarlark, Maybe think of a better place?
  private static Object valueToStarlark(Object x) {
    // Is x a non-empty string_list_dict?
    if (x instanceof Map) {
      Map<?, ?> map = (Map<?, ?>) x;
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
