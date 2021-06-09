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

import com.google.devtools.common.options.Converters.BooleanConverter;

/**
 * Converter that can convert both the standard set of boolean string values and enumerations. If
 * there is an overlap in values, those from the underlying enumeration will be taken.
 *
 * <p>Note that for the flag to take one of its enum values on the command line, it must be of the
 * form "--flag=value". That is, "--flag value" and "-f value" (if the flag has a short-form of "f")
 * will result in "value" being left as residue on the command line. This maintains compatibility
 * with boolean flags where "--flag true" and "-f true" also leave "true" as residue on the command
 * line.
 */
public abstract class BoolOrEnumConverter<T extends Enum<T>> extends EnumConverter<T> {
  private final T falseValue;
  private final T trueValue;

  /**
   * You *must* implement a zero-argument constructor that delegates
   * to this constructor, passing in the appropriate parameters. This
   * comes from the base {@link EnumConverter} class.
   *
   * @param enumType The type of your enumeration; usually a class literal
   *                 like MyEnum.class
   * @param typeName The intuitive name of your enumeration, for example, the
   *                 type name for CompilationMode might be "compilation mode".
   * @param trueValue The enumeration value to associate with {@code true}.
   * @param falseValue The enumeration value to associate with {@code false}.
   */
  protected BoolOrEnumConverter(Class<T> enumType, String typeName, T trueValue, T falseValue) {
    super(enumType, typeName);
    this.trueValue = trueValue;
    this.falseValue = falseValue;
  }

  @Override
  public T convert(String input) throws OptionsParsingException {
    try {
      return super.convert(input);
    } catch (OptionsParsingException eEnum) {
      try {
        BooleanConverter booleanConverter = new BooleanConverter();
        boolean value = booleanConverter.convert(input);
        return value ? trueValue : falseValue;
      } catch (OptionsParsingException eBoolean) {
        // TODO(b/111883901): Rethrowing the exception from the enum converter does not report the
        // allowable boolean values.
        throw eEnum;
      }
    }
  }
}
