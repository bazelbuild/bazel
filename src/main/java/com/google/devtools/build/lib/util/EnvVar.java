// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.common.options.OptionsParsingException;

/** A request to set or unset a particular environment variable. */
public sealed interface EnvVar {
  /** The name of the environment variable. */
  String name();

  /** Set the environment variable to the given value. */
  @AutoCodec
  record Set(String name, String value) implements EnvVar {}

  /** Inherit the value of the environment variable from the client environment. */
  @AutoCodec
  record Inherit(String name) implements EnvVar {}

  /**
   * Unset the environment variable, i.e., remove any previous assignment or even explicitly unset
   * it if implicitly inheriting the client environment.
   */
  @AutoCodec
  record Unset(String name) implements EnvVar {}

  /**
   * A converter for variable assignments from the parameter list of a blaze command invocation.
   * Assignments are expected to have the form "name[=value]", where names and values are defined to
   * be as permissive as possible and value part can be optional (in which case it is considered to
   * be inherited). The special syntax "=name" is also supported and interpreted as a request to
   * unset the variable with the given name.
   */
  public static class Converter
      extends com.google.devtools.common.options.Converter.Contextless<EnvVar> {

    @Override
    public EnvVar convert(String input) throws OptionsParsingException {
      int pos = input.indexOf('=');
      if (input.isEmpty() || input.equals("=")) {
        throw new OptionsParsingException(
            "Variable definitions must be in the form of a 'name=value', 'name', or '=name'"
                + " assignment");
      } else if (pos == 0) {
        return new EnvVar.Unset(input.substring(1));
      } else if (pos < 0) {
        return new EnvVar.Inherit(input);
      }
      String name = input.substring(0, pos);
      String value = input.substring(pos + 1);
      return new EnvVar.Set(name, value);
    }

    @Override
    public boolean starlarkConvertible() {
      return true;
    }

    @Override
    public String reverseForStarlark(Object converted) {
      if (converted instanceof EnvVar.Set set) {
        return set.name() + "=" + set.value();
      } else if (converted instanceof EnvVar.Inherit inherit) {
        return inherit.name();
      } else if (converted instanceof EnvVar.Unset unset) {
        return "=" + unset.name();
      } else {
        throw new IllegalArgumentException(
            "EnvVar.Converter can only reverse EnvVar types, got: " + converted);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a 'name[=value]' assignment with an optional value part or the special syntax '=name'"
          + " to unset a variable";
    }
  }
}
