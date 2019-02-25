// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Printer.BasePrinter;
import javax.annotation.Nullable;

/** Stores information about a function parameter definition. */
public class FunctionParamInfo {

  private final String name;
  private final String docString;
  @Nullable private final Object defaultValue;
  private final boolean mandatory;

  private FunctionParamInfo(
      String name, String docString, @Nullable Object defaultValue, boolean mandatory) {
    this.name = name;
    this.docString = docString;
    this.defaultValue = defaultValue;
    this.mandatory = mandatory;
  }

  /** Constructor to be used for normal parameters. */
  public static FunctionParamInfo forParam(
      String name, String docString, @Nullable Object defaultValue) {
    return new FunctionParamInfo(name, docString, defaultValue, defaultValue == null);
  }

  /** Constructor to be used for *args or **kwargs. */
  public static FunctionParamInfo forSpecialParam(String name, String docString) {
    return new FunctionParamInfo(name, docString, null, false);
  }

  /**
   * Return the name of this parameter (for example, in 'def foo(bar):', the only parameter is
   * named 'bar'.
   */
  public String getName() {
    return name;
  }

  /**
   * Return the documented description of this parameter (if specified in the function's docstring).
   */
  public String getDocString() {
    return docString;
  }

  /**
   * Returns true if this function has a default value and the default value can be displayed
   * as a string.
   */
  public boolean hasDefaultValueString() {
    return defaultValue != null && !getDefaultString().isEmpty();
  }

  /**
   * Returns a string representing the default value this function parameter.
   *
   * @throws IllegalStateException if there is no default value of this function parameter;
   *     invoke {@link #hasDefaultValueString()} first to check whether there is a default
   *     parameter
   */
  public String getDefaultString() {
    if (defaultValue == null) {
      return "";
    }
    BasePrinter printer = Printer.getSimplifiedPrinter();
    printer.repr(defaultValue);
    return printer.toString();
  }

  /**
   * Returns 'required' if this parameter is mandatory, otherwise returns 'optional'.
   */
  public String getMandatoryString() {
    return mandatory ? "required" : "optional";
  }
}
