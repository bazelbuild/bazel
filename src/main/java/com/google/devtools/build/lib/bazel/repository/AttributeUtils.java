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
//
package com.google.devtools.build.lib.bazel.repository;

import static java.util.Collections.singletonList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;
import net.starlark.java.spelling.SpellChecker;

/** Utilities related to processing attributes in external deps contexts. */
public class AttributeUtils {
  private AttributeUtils() {}

  /**
   * Type-checks the given attribute values against a defined attribute schema, potentially
   * converting the values wherever necessary.
   *
   * @param attrs With {@code attrIndices}, defines the attribute schema.
   * @param kwargs The supplied attribute values (keyed by the attribute names).
   * @param where A context string used in error messages to denote where this typechecking is
   *     happening.
   * @param repoMappingWhere A context string used in error messages about invalid apparent repo
   *     names, to denote where this repo mapping is anchored.
   * @return The type-checked and converted values, in the same order as {@code attrs}.
   */
  public static ImmutableList<Object> typeCheckAttrValues(
      ImmutableList<Attribute> attrs,
      ImmutableMap<String, Integer> attrIndices,
      Map<String, Object> kwargs,
      LabelConverter labelConverter,
      Code errorCode,
      ImmutableList<CallStackEntry> callStack,
      String where,
      String repoMappingWhere)
      throws ExternalDepsException {
    var attrValues = new Object[attrs.size()];
    for (Entry<String, Object> attrValue : kwargs.entrySet()) {
      if (attrValue.getValue().equals(Starlark.NONE)) {
        continue;
      }
      Integer attrIndex = attrIndices.get(attrValue.getKey());
      if (attrIndex == null) {
        throw ExternalDepsException.withCallStackAndMessage(
            errorCode,
            callStack,
            "in %s, unknown attribute '%s' provided%s",
            where,
            attrValue.getKey(),
            SpellChecker.didYouMean(attrValue.getKey(), attrIndices.keySet()));
      }
      Attribute attr = attrs.get(attrIndex);
      Object nativeValue;
      try {
        nativeValue =
            attr.getType()
                .convert(
                    attrValue.getValue(),
                    "attribute '%s'".formatted(attr.getPublicName()),
                    labelConverter);
      } catch (ConversionException e) {
        throw ExternalDepsException.withCallStackAndMessage(
            errorCode, callStack, "in %s, %s", where, e.getMessage());
      }

      // Check that the value is actually allowed.
      if (attr.checkAllowedValues() && !attr.getAllowedValues().apply(nativeValue)) {
        throw ExternalDepsException.withCallStackAndMessage(
            errorCode,
            callStack,
            "in %s, the value for attribute '%s' %s",
            where,
            attr.getPublicName(),
            attr.getAllowedValues().getErrorReason(nativeValue));
      }

      attrValues[attrIndex] = Attribute.valueToStarlark(nativeValue);
    }

    // Check that all mandatory attributes have been specified, and fill in default values.
    // Along the way, verify that labels in the attribute values refer to visible repos only.
    for (int i = 0; i < attrValues.length; i++) {
      Attribute attr = attrs.get(i);
      if (attr.isMandatory() && attrValues[i] == null) {
        throw ExternalDepsException.withCallStackAndMessage(
            errorCode,
            callStack,
            "in %s, mandatory attribute '%s' isn't being specified",
            where,
            attr.getPublicName());
      }
      if (attrValues[i] == null) {
        attrValues[i] = Attribute.valueToStarlark(attr.getDefaultValueUnchecked());
      }
      Label firstNonVisibleLabel = getFirstNonVisibleLabel(attrValues[i]);
      if (firstNonVisibleLabel != null) {
        throw ExternalDepsException.withCallStackAndMessage(
            errorCode,
            callStack,
            "in %s, no repository visible as '@%s' %s, but referenced by label '@%s//%s:%s'"
                + " in attribute '%s'",
            where,
            firstNonVisibleLabel.getRepository().getName(),
            repoMappingWhere,
            firstNonVisibleLabel.getRepository().getName(),
            firstNonVisibleLabel.getPackageFragment(),
            firstNonVisibleLabel.getName(),
            attr.getPublicName());
      }
    }
    return ImmutableList.copyOf(attrValues);
  }

  @Nullable
  private static Label getFirstNonVisibleLabel(Object nativeAttrValue) {
    Collection<?> toValidate =
        switch (nativeAttrValue) {
          case List<?> list -> list;
          case Map<?, ?> map -> map.keySet();
          case null, default -> singletonList(nativeAttrValue);
        };
    for (var item : toValidate) {
      if (item instanceof Label label && !label.getRepository().isVisible()) {
        return label;
      }
    }
    return null;
  }
}
