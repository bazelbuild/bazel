// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.FunctionSignature.WithValues;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A constructor for {@link SkylarkClassObject}.
 */
public final class SkylarkClassObjectConstructor extends BaseFunction {
  /**
   * "struct" function.
   */
  public static final SkylarkClassObjectConstructor STRUCT =
      new SkylarkClassObjectConstructor("struct");


  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      WithValues.create(FunctionSignature.KWARGS);

  public SkylarkClassObjectConstructor(String name, Location location) {
    super(name, SIGNATURE, location);
  }

  public SkylarkClassObjectConstructor(String name) {
    this(name, Location.BUILTIN);
  }

  @Override
  protected Object call(Object[] args, @Nullable FuncallExpression ast, @Nullable Environment env)
      throws EvalException, ConversionException, InterruptedException {
    @SuppressWarnings("unchecked")
    Map<String, Object> kwargs = (Map<String, Object>) args[0];
    return new SkylarkClassObject(this, kwargs, ast != null ? ast.getLocation() : Location.BUILTIN);
  }

  /**
   * Creates a built-in class object (i.e. without creation loc). The errorMessage has to have
   * exactly one '%s' parameter to substitute the field name.
   */
  public SkylarkClassObject create(Map<String, Object> values, String message) {
    return new SkylarkClassObject(this, values, message);
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }

  @Override
  public boolean equals(@Nullable Object other) {
    return other == this;
  }
}
