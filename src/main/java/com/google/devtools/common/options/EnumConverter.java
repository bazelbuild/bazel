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

package com.google.devtools.common.options;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Ascii;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Arrays;
import java.util.HashMap;

/**
 * A converter superclass for converters that parse enums.
 *
 * <p>Just subclass this class, creating a zero argument constructor that calls {@link
 * #EnumConverter(Class, String)}.
 *
 * <p>This class compares the input string to the string returned by the toString() method of each
 * enum member in a case-insensitive way. Usually, this is the name of the symbol, but beware if you
 * override toString()!
 */
public abstract class EnumConverter<T extends Enum<T>> extends Converter.Contextless<T> {

  private final Class<T> enumType;
  protected final String typeName;

  /**
   * Creates a new enum converter. You *must* implement a zero-argument constructor that delegates
   * to this constructor, passing in the appropriate parameters.
   *
   * @param enumType The type of your enumeration; usually a class literal like MyEnum.class. All
   *     enum constants of the given type must have unique case-insensitive toString() values.
   * @param typeName The intuitive name of your enumeration, for example, the type name for
   *     CompilationMode might be "compilation mode".
   */
  protected EnumConverter(Class<T> enumType, String typeName) {
    this.enumType = checkUniqueCaseInsensitiveStringRepresentation(enumType);
    this.typeName = typeName;
  }

  @CanIgnoreReturnValue
  private static <T extends Enum<T>> Class<T> checkUniqueCaseInsensitiveStringRepresentation(
      Class<T> enumType) {
    HashMap<String, Enum<?>> enumConstants = new HashMap<>();
    for (Enum<?> value : enumType.getEnumConstants()) {
      String key = Ascii.toLowerCase(value.toString());
      if (enumConstants.containsKey(key)) {
        throw new IllegalArgumentException(
            String.format(
                "Enum type %s values %s and %s collide in their case-insensitive string"
                    + " representation '%s'",
                enumType.getName(), enumConstants.get(key).name(), value.name(), key));
      }
      enumConstants.put(key, value);
    }
    return enumType;
  }

  /** Implements {@link Converter#convert(String, Object)}. */
  @Override
  public T convert(String input) throws OptionsParsingException {
    for (T value : enumType.getEnumConstants()) {
      if (Ascii.equalsIgnoreCase(value.toString(), input)) {
        return value;
      }
    }
    throw new OptionsParsingException(
        "Not a valid %s: '%s' (should be %s)".formatted(typeName, input, getTypeDescription()));
  }

  /** Implements {@link #getTypeDescription()}. */
  @Override
  public String getTypeDescription() {
    return Ascii.toLowerCase(
        Converters.joinEnglishList(Arrays.asList(enumType.getEnumConstants())));
  }

  @Override
  public boolean starlarkConvertible() {
    return true;
  }

  @Override
  public String reverseForStarlark(Object converted) {
    checkArgument(enumType.isInstance(converted));
    return Ascii.toLowerCase(converted.toString());
  }

  public final Class<T> getEnumType() {
    return enumType;
  }
}
