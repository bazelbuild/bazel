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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.spelling.SpellChecker;

/**
 * A BuiltinCallable is a callable Starlark value that reflectively invokes a
 * StarlarkMethod-annotated method of a Java object.
 */
// TODO(adonovan): make this private. Most users would be content with StarlarkCallable; the rest
// need only a means of querying the function's parameters.
public final class BuiltinCallable implements StarlarkCallable {

  private final Object obj;
  private final String methodName;
  @Nullable private final MethodDescriptor desc;

  /**
   * Constructs a BuiltinCallable for a StarlarkCallable-annotated method of the given name (as seen
   * by Starlark, not Java).
   */
  // TODO(adonovan): eliminate calls to this constructor from tests; use getattr instead.
  BuiltinCallable(Object obj, String methodName) {
    this(obj, methodName, /*desc=*/ null);
  }

  /**
   * Constructs a BuiltinCallable for a StarlarkCallable-annotated method of the given name (as seen
   * by Starlark, not Java).
   *
   * <p>This constructor should be used only for ephemeral BuiltinCallable values created
   * transiently during a call such as {@code x.f()}, when the caller has already looked up the
   * MethodDescriptor using the same semantics as the thread that will be used in the call. Use the
   * other (slower) constructor if there is any possibility that the semantics of the {@code x.f}
   * operation differ from those of the thread used in the call.
   */
  BuiltinCallable(Object obj, String methodName, MethodDescriptor desc) {
    this.obj = obj;
    this.methodName = methodName;
    this.desc = desc;
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException, InterruptedException {
    MethodDescriptor desc =
        this.desc != null ? this.desc : getMethodDescriptor(thread.getSemantics());
    Preconditions.checkArgument(
        !desc.isStructField(),
        "struct field methods should be handled by DotExpression separately");
    Object[] vector = getArgumentVector(thread, desc, positional, named);
    return desc.call(
        obj instanceof String ? StringModule.INSTANCE : obj, vector, thread.mutability());
  }

  private MethodDescriptor getMethodDescriptor(StarlarkSemantics semantics) {
    return CallUtils.getAnnotatedMethod(semantics, obj.getClass(), methodName);
  }

  /**
   * Returns the StarlarkMethod annotation of this Starlark-callable Java method.
   *
   * @deprecated This method is intended only for docgen, and uses the default semantics.
   */
  @Deprecated
  public StarlarkMethod getAnnotation() {
    return getMethodDescriptor(StarlarkSemantics.DEFAULT).getAnnotation();
  }

  @Override
  public String getName() {
    return methodName;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<built-in function " + methodName + ">");
  }

  @Override
  public String toString() {
    return methodName;
  }

  /**
   * Converts the arguments of a Starlark call into the argument vector for a reflective call to a
   * StarlarkMethod-annotated Java method.
   *
   * @param thread the Starlark thread for the call
   * @param loc the location of the call expression, or BUILTIN for calls from Java
   * @param desc descriptor for the StarlarkCallable-annotated method
   * @param positional a list of positional arguments
   * @param named a list of named arguments, as alternating Strings/Objects. May contain dups.
   * @return the array of arguments which may be passed to {@link MethodDescriptor#call}
   * @throws EvalException if the given set of arguments are invalid for the given method. For
   *     example, if any arguments are of unexpected type, or not all mandatory parameters are
   *     specified by the user
   */
  private Object[] getArgumentVector(
      StarlarkThread thread,
      MethodDescriptor desc, // intentionally shadows this.desc
      Object[] positional,
      Object[] named)
      throws EvalException {

    // Overview of steps:
    // - allocate vector of actual arguments of correct size.
    // - process positional arguments, accumulating surplus ones into *args.
    // - process named arguments, accumulating surplus ones into **kwargs.
    // - set default values for missing optionals, and report missing mandatory parameters.
    // - set special parameters.
    // The static checks ensure that positional parameters appear before named,
    // and mandatory positionals appear before optional.
    // No additional memory allocation occurs in the common (success) case.
    // Flag-disabled parameters are skipped during argument matching, as if they do not exist. They
    // are instead assigned their flag-disabled values.

    ParamDescriptor[] parameters = desc.getParameters();

    // Allocate argument vector.
    int n = parameters.length;
    if (desc.acceptsExtraArgs()) {
      n++;
    }
    if (desc.acceptsExtraKwargs()) {
      n++;
    }
    if (desc.isUseStarlarkThread()) {
      n++;
    }
    Object[] vector = new Object[n];

    // positional arguments
    int paramIndex = 0;
    int argIndex = 0;
    if (obj instanceof String) {
      // String methods get the string as an extra argument
      // because their true receiver is StringModule.INSTANCE.
      vector[paramIndex++] = obj;
    }
    for (; argIndex < positional.length && paramIndex < parameters.length; paramIndex++) {
      ParamDescriptor param = parameters[paramIndex];
      if (!param.isPositional()) {
        break;
      }

      // disabled?
      if (param.disabledByFlag() != null) {
        // Skip disabled parameter as if not present at all.
        // The default value will be filled in below.
        continue;
      }

      Object value = positional[argIndex++];
      checkParamValue(param, value);
      vector[paramIndex] = value;
    }

    // *args
    Tuple<Object> varargs = null;
    if (desc.acceptsExtraArgs()) {
      varargs = Tuple.wrap(Arrays.copyOfRange(positional, argIndex, positional.length));
    } else if (argIndex < positional.length) {
      if (argIndex == 0) {
        throw Starlark.errorf("%s() got unexpected positional argument", methodName);
      } else {
        throw Starlark.errorf(
            "%s() accepts no more than %d positional argument%s but got %d",
            methodName, argIndex, plural(argIndex), positional.length);
      }
    }

    // named arguments
    LinkedHashMap<String, Object> kwargs = desc.acceptsExtraKwargs() ? new LinkedHashMap<>() : null;
    for (int i = 0; i < named.length; i += 2) {
      String name = (String) named[i]; // safe
      Object value = named[i + 1];

      // look up parameter
      int index = desc.getParameterIndex(name);
      // unknown parameter?
      if (index < 0) {
        // spill to **kwargs
        if (kwargs == null) {
          List<String> allNames =
              Arrays.stream(parameters)
                  .map(ParamDescriptor::getName)
                  .collect(ImmutableList.toImmutableList());
          throw Starlark.errorf(
              "%s() got unexpected keyword argument '%s'%s",
              methodName, name, SpellChecker.didYouMean(name, allNames));
        }

        // duplicate named argument?
        if (kwargs.put(name, value) != null) {
          throw Starlark.errorf(
              "%s() got multiple values for keyword argument '%s'", methodName, name);
        }
        continue;
      }
      ParamDescriptor param = parameters[index];

      // positional-only param?
      if (!param.isNamed()) {
        // spill to **kwargs
        if (kwargs == null) {
          throw Starlark.errorf(
              "%s() got named argument for positional-only parameter '%s'", methodName, name);
        }

        // duplicate named argument?
        if (kwargs.put(name, value) != null) {
          throw Starlark.errorf(
              "%s() got multiple values for keyword argument '%s'", methodName, name);
        }
        continue;
      }

      // disabled?
      String flag = param.disabledByFlag();
      if (flag != null) {
        // spill to **kwargs
        if (kwargs == null) {
          throw Starlark.errorf(
              "in call to %s(), parameter '%s' is %s",
              methodName, param.getName(), disabled(flag, thread.getSemantics()));
        }

        // duplicate named argument?
        if (kwargs.put(name, value) != null) {
          throw Starlark.errorf(
              "%s() got multiple values for keyword argument '%s'", methodName, name);
        }
        continue;
      }

      checkParamValue(param, value);

      // duplicate?
      if (vector[index] != null) {
        throw Starlark.errorf("%s() got multiple values for argument '%s'", methodName, name);
      }

      vector[index] = value;
    }

    // Set default values for missing parameters,
    // and report any that are still missing.
    List<String> missingPositional = null;
    List<String> missingNamed = null;
    for (int i = 0; i < parameters.length; i++) {
      if (vector[i] == null) {
        ParamDescriptor param = parameters[i];
        vector[i] = param.getDefaultValue();
        if (vector[i] == null) {
          if (param.isPositional()) {
            if (missingPositional == null) {
              missingPositional = new ArrayList<>();
            }
            missingPositional.add(param.getName());
          } else {
            if (missingNamed == null) {
              missingNamed = new ArrayList<>();
            }
            missingNamed.add(param.getName());
          }
        }
      }
    }
    if (missingPositional != null) {
      throw Starlark.errorf(
          "%s() missing %d required positional argument%s: %s",
          methodName,
          missingPositional.size(),
          plural(missingPositional.size()),
          Joiner.on(", ").join(missingPositional));
    }
    if (missingNamed != null) {
      throw Starlark.errorf(
          "%s() missing %d required named argument%s: %s",
          methodName,
          missingNamed.size(),
          plural(missingNamed.size()),
          Joiner.on(", ").join(missingNamed));
    }

    // special parameters
    int i = parameters.length;
    if (desc.acceptsExtraArgs()) {
      vector[i++] = varargs;
    }
    if (desc.acceptsExtraKwargs()) {
      vector[i++] = Dict.wrap(thread.mutability(), kwargs);
    }
    if (desc.isUseStarlarkThread()) {
      vector[i++] = thread;
    }

    return vector;
  }

  private static String plural(int n) {
    return n == 1 ? "" : "s";
  }

  private void checkParamValue(ParamDescriptor param, Object value) throws EvalException {
    // Value must belong to one of the specified classes.
    boolean ok = false;
    for (Class<?> cls : param.getAllowedClasses()) {
      if (cls.isInstance(value)) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      throw Starlark.errorf(
          "in call to %s(), parameter '%s' got value of type '%s', want '%s'",
          methodName, param.getName(), Starlark.type(value), param.getTypeErrorMessage());
    }

    // None is valid if and only if the parameter is marked noneable,
    // in which case the above check passes as the list of classes will include NoneType.
    // The reason for this check is to ensure that merely having type=Object.class
    // does not allow None as an argument value; I'm not sure why, that but that's the
    // historical behavior.
    //
    // We do this check second because the first check prints a better error
    // that enumerates the allowed types.
    if (value == Starlark.NONE && !param.isNoneable()) {
      throw Starlark.errorf(
          "in call to %s(), parameter '%s' cannot be None", methodName, param.getName());
    }
  }

  // Returns a phrase meaning "disabled" appropriate to the specified flag.
  private static String disabled(String flag, StarlarkSemantics semantics) {
    // If the flag is True, it must be a deprecation flag. Otherwise it's an experimental flag.
    // TODO(adonovan): is that assumption sound?
    if (semantics.flagValue(flag)) {
      return String.format(
          "deprecated and will be removed soon. It may be temporarily re-enabled by setting"
              + " --%s=false",
          flag);
    } else {
      return String.format(
          "experimental and thus unavailable with the current flags. It may be enabled by setting"
              + " --%s",
          flag);
    }
  }
}
