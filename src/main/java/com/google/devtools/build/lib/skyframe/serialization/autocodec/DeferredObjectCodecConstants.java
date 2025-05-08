// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import static com.google.devtools.build.lib.util.StringUtilities.capitalize;

import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import java.lang.invoke.MethodHandles;
import javax.lang.model.element.Name;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeKind;

/** Common {@link DeferredObjectCodec} building constants and helper methods. */
final class DeferredObjectCodecConstants {
  /** Name of constructor scoped variable holding the {@link MethodHandles.Lookup} instance. */
  static final String CONSTRUCTOR_LOOKUP_NAME = "lookup";

  /** Class name for the codec's {@link DeferredObjectCodec.DeferredValue} implementation. */
  static final String BUILDER_TYPE_NAME = "Builder";

  /** Name of an instance variable in the deserialize method storing the builder instance. */
  static final String BUILDER_NAME = "builder";

  static String makeGetterName(VariableElement parameter) {
    String prefix = parameter.asType().getKind() == TypeKind.BOOLEAN ? "is" : "get";
    String suffix = capitalize(parameter.getSimpleName().toString());
    return prefix + suffix;
  }

  static String makeSetterName(Name name) {
    return "set" + capitalize(name.toString());
  }

  private DeferredObjectCodecConstants() {}
}
