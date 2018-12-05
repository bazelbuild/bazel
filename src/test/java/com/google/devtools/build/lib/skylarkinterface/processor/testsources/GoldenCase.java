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

package com.google.devtools.build.lib.skylarkinterface.processor.testsources;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;

/**
 * Test source file verifying various proper uses of SkylarkCallable.
 */
public class GoldenCase {

  @SkylarkCallable(
    name = "struct_field_method",
    documented = false,
    structField = true)
  public String structFieldMethod() {
    return "foo";
  }

  @SkylarkCallable(
    name = "struct_field_method_with_info",
    documented = false,
    structField = true,
    useSkylarkSemantics = true
  )
  public String structFieldMethodWithInfo(SkylarkSemantics semantics) {
    return "foo";
  }

  @SkylarkCallable(
    name = "zero_arg_method",
    documented = false)
  public Integer zeroArgMethod() {
    return 0;
  }

  @SkylarkCallable(name = "zero_arg_method_with_environment", documented = false,
      useEnvironment = true)
  public Integer zeroArgMethod(Environment environment) {
    return 0;
  }

  @SkylarkCallable(
    name = "zero_arg_method_with_skylark_info",
    documented = false,
    useAst = true,
    useLocation = true,
    useEnvironment = true,
    useSkylarkSemantics = true
  )
  public Integer zeroArgMethod(
      Location location,
      FuncallExpression ast,
      Environment environment,
      SkylarkSemantics semantics) {
    return 0;
  }

  @SkylarkCallable(name = "three_arg_method_with_ast",
      documented = false,
      parameters = {
          @Param(name = "one", type = String.class, named = true),
          @Param(name = "two", type = Integer.class, named = true),
          @Param(name = "three", type = String.class, named = true,
              defaultValue = "None", noneable = true),
      },
      useAst = true)
  public String threeArgMethod(String one, Integer two, String three, FuncallExpression ast) {
    return "bar";
  }

  @SkylarkCallable(
    name = "three_arg_method_with_params",
    documented = false,
    parameters = {
      @Param(name = "one", type = String.class, named = true),
      @Param(name = "two", type = Integer.class, named = true),
      @Param(name = "three",
          allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Integer.class),
          },
          named = true, defaultValue = "None", noneable = true),
    })
  public String threeArgMethod(String one, Integer two, Object three) {
    return "baz";
  }

  @SkylarkCallable(
    name = "three_arg_method_with_params_and_info",
    documented = false,
    parameters = {
      @Param(name = "one", type = String.class, named = true),
      @Param(name = "two", type = Integer.class, named = true),
      @Param(name = "three", type = String.class, named = true),
    },
    useAst = true,
    useLocation = true,
    useEnvironment = true,
    useSkylarkSemantics = true
  )
  public String threeArgMethodWithParams(
      String one,
      Integer two,
      String three,
      Location location,
      FuncallExpression ast,
      Environment environment,
      SkylarkSemantics skylarkSemantics) {
    return "baz";
  }

  @SkylarkCallable(
      name = "many_arg_method_mixing_positional_and_named",
      documented = false,
      parameters = {
          @Param(name = "one", type = String.class, positional = true, named = false),
          @Param(name = "two", type = String.class, positional = true, named = true),
          @Param(name = "three", type = String.class, positional = true, named = true,
              defaultValue = "three"),
          @Param(name = "four", type = String.class, positional = false, named = true),
          @Param(name = "five", type = String.class, positional = false, named = true,
              defaultValue = "five"),
          @Param(name = "six", type = String.class, positional = false, named = true),
      },
      useLocation = true
  )
  public String manyArgMethodMixingPositoinalAndNamed(
      String one,
      String two,
      String three,
      String four,
      String five,
      String six,
      Location location) {
    return "baz";
  }

  @SkylarkCallable(
      name = "two_arg_method_with_params_and_info_and_kwargs",
      documented = false,
      parameters = {
        @Param(name = "one", type = String.class, named = true),
        @Param(name = "two", type = Integer.class, named = true),
      },
      extraKeywords = @Param(name = "kwargs"),
      useAst = true,
      useLocation = true,
      useEnvironment = true,
      useSkylarkSemantics = true,
      useContext = true)
  public String twoArgMethodWithParamsAndInfoAndKwargs(
      String one,
      Integer two,
      SkylarkDict<?, ?> kwargs,
      Location location,
      FuncallExpression ast,
      Environment environment,
      SkylarkSemantics skylarkSemantics,
      StarlarkContext context) {
    return "blep";
  }

  @SkylarkCallable(
    name = "two_arg_method_with_env_and_args_and_kwargs",
    documented = false,
    parameters = {
      @Param(name = "one", type = String.class, named = true),
      @Param(name = "two", type = Integer.class, named = true),
    },
    extraPositionals = @Param(name = "args"),
    extraKeywords = @Param(name = "kwargs"),
    useEnvironment = true
  )
  public String twoArgMethodWithParamsAndInfoAndKwargs(
      String one,
      Integer two,
      SkylarkList<?> args,
      SkylarkDict<?, ?> kwargs,
      Environment environment) {
    return "yar";
  }

  @SkylarkCallable(
    name = "selfCallMethod",
    selfCall = true,
    parameters = {
        @Param(name = "one", type = String.class, named = true),
        @Param(name = "two", type = Integer.class, named = true),
    },
    documented = false
  )
  public Integer selfCallMethod(String one, Integer two) {
    return 0;
  }

  @SkylarkCallable(
      name = "struct_field_method_with_extra_args",
      documented = false,
      structField = true,
      useLocation = true,
      useEnvironment = true,
      useSkylarkSemantics = true
  )
  public String structFieldMethodWithInfo(Location location,
      Environment environment,
      SkylarkSemantics skylarkSemantics) {
    return "dragon";
  }
}
