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

package com.google.devtools.common.options;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Collection;
import java.util.Map;
import javax.annotation.concurrent.Immutable;

/**
 * This extends IsolatedOptionsData with information that can only be determined once all the {@link
 * OptionsBase} subclasses for a parser are known. In particular, this includes expansion
 * information.
 */
@Immutable
final class OptionsData extends IsolatedOptionsData {

  /**
   * Mapping from each Option-annotated field with a {@code String[]} expansion to that expansion.
   */
  // TODO(brandjon): This is technically not necessarily immutable due to String[], and should use
  // ImmutableList. Either fix this or remove @Immutable.
  private final ImmutableMap<Field, String[]> evaluatedExpansions;

  /** Construct {@link OptionsData} by extending an {@link IsolatedOptionsData} with new info. */
  private OptionsData(IsolatedOptionsData base, Map<Field, String[]> evaluatedExpansions) {
    super(base);
    this.evaluatedExpansions = ImmutableMap.copyOf(evaluatedExpansions);
  }

  private static final String[] EMPTY_EXPANSION = new String[] {};

  /**
   * Returns the expansion of an options field, regardless of whether it was defined using {@link
   * Option#expansion} or {@link Option#expansionFunction}. If the field is not an expansion option,
   * returns an empty array.
   */
  public String[] getEvaluatedExpansion(Field field) {
    String[] result = evaluatedExpansions.get(field);
    return result != null ? result : EMPTY_EXPANSION;
  }

  /**
   * Constructs an {@link OptionsData} object for a parser that knows about the given {@link
   * OptionsBase} classes. In addition to the work done to construct the {@link
   * IsolatedOptionsData}, this also computes expansion information.
   */
  public static OptionsData from(Collection<Class<? extends OptionsBase>> classes) {
    IsolatedOptionsData isolatedData = IsolatedOptionsData.from(classes);

    // All that's left is to compute expansions.
    Map<Field, String[]> evaluatedExpansionsBuilder = Maps.newHashMap();
    for (Map.Entry<String, Field> entry : isolatedData.getAllNamedFields()) {
      Field field = entry.getValue();
      Option annotation = field.getAnnotation(Option.class);
      // Determine either the hard-coded expansion, or the ExpansionFunction class.
      String[] constExpansion = annotation.expansion();
      Class<? extends ExpansionFunction> expansionFunctionClass = annotation.expansionFunction();
      if (constExpansion.length > 0 && usesExpansionFunction(annotation)) {
        throw new AssertionError(
            "Cannot set both expansion and expansionFunction for option --" + annotation.name());
      } else if (constExpansion.length > 0) {
        evaluatedExpansionsBuilder.put(field, constExpansion);
      } else if (usesExpansionFunction(annotation)) {
        if (Modifier.isAbstract(expansionFunctionClass.getModifiers())) {
          throw new AssertionError(
              "The expansionFunction type " + expansionFunctionClass + " must be a concrete type");
        }
        // Evaluate the ExpansionFunction.
        ExpansionFunction instance;
        try {
          Constructor<?> constructor = expansionFunctionClass.getConstructor();
          instance = (ExpansionFunction) constructor.newInstance();
        } catch (Exception e) {
          // This indicates an error in the ExpansionFunction, and should be discovered the first
          // time it is used.
          throw new AssertionError(e);
        }
        String[] expansion = instance.getExpansion(isolatedData);
        evaluatedExpansionsBuilder.put(field, expansion);
      }
    }

    return new OptionsData(isolatedData, evaluatedExpansionsBuilder);
  }
}
