// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Converter that translates a string of the form "value1,value2,-value3,value4"
 * into a corresponding set of allowed Enum values.
 *
 * <p>Values preceded by '-' are excluded from this set. So "value1,-value2,value3"
 * translates to the set [EnumType.value1, EnumType.value3].
 *
 * <p>If *all* values are exclusions (e.g. "-value1,-value2,-value3"), the returned
 * set contains all values for the Enum type *except* those specified.
 */
class EnumFilterConverter<E extends Enum<E>> implements Converter<Set<E>> {

  private final Set<String> allowedValues = new LinkedHashSet<>();
  private final Class<E> typeClass;
  private final String prettyEnumName;

  /**
   * Constructor.
   *
   * @param typeClass this should be E.class (Java generics can't infer that directly)
   * @param userFriendlyName a user-friendly description of this enum type
   */
  EnumFilterConverter(Class<E> typeClass, String userFriendlyName) {
    this.typeClass = typeClass;
    this.prettyEnumName = userFriendlyName;
    for (E value : EnumSet.allOf(typeClass)) {
      allowedValues.add(value.name());
    }
  }

  /**
   * Returns the set of allowed values for the option.
   *
   * Implements {@link #convert(String)}.
   */
  @Override
  public Set<E> convert(String input) throws OptionsParsingException {
    if (input.isEmpty()) {
      return Collections.emptySet();
    }
    EnumSet<E> includedSet = EnumSet.noneOf(typeClass);
    EnumSet<E> excludedSet = EnumSet.noneOf(typeClass);
    for (String value : input.split(",", -1)) {
      boolean excludeFlag = value.startsWith("-");
      String s = (excludeFlag ? value.substring(1) : value).toUpperCase();
      if (!allowedValues.contains(s)) {
        throw new OptionsParsingException("Invalid " + prettyEnumName + " filter '" + value +
            "' in the input '" + input + "'");
      }
      (excludeFlag ? excludedSet : includedSet).add(Enum.valueOf(typeClass, s));
    }
    if (includedSet.isEmpty()) {
      includedSet = EnumSet.complementOf(excludedSet);
    } else {
      includedSet.removeAll(excludedSet);
    }
    if (includedSet.isEmpty()) {
      throw new OptionsParsingException(
          Character.toUpperCase(prettyEnumName.charAt(0)) + prettyEnumName.substring(1) +
          " filter '" + input + "' definition cannot match any tests");
    }
    return includedSet;
  }

  /**
   * Implements {@link #getTypeDescription()}.
   */
  @Override
  public final String getTypeDescription() {
    return "comma-separated list of values: "
        + StringUtil.joinEnglishList(allowedValues).toLowerCase();
  }
}
