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
package net.starlark.java.eval;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.spelling.SpellChecker;

/**
 * A BuiltinFunction is a callable Starlark value that reflectively invokes a {@link
 * StarlarkMethod}-annotated method of a Java object. The Java object may or may not itself be a
 * Starlark value. BuiltinFunctions are not produced for Java methods for which {@link
 * StarlarkMethod#structField} is true.
 */
// TODO(adonovan): support annotated static methods.
@StarlarkBuiltin(
    name = "builtin_function_or_method", // (following Python)
    category = "core",
    doc = "The type of a built-in function, defined by Java code.")
public final class BuiltinFunction implements StarlarkCallable {

  private final Object obj;
  private final String methodName;
  @Nullable private final MethodDescriptor desc;

  /**
   * Constructs a BuiltinFunction for a StarlarkMethod-annotated method of the given name (as seen
   * by Starlark, not Java).
   */
  BuiltinFunction(Object obj, String methodName) {
    this.obj = obj;
    this.methodName = methodName;
    this.desc = null; // computed later
  }

  /**
   * Constructs a BuiltinFunction for a StarlarkMethod-annotated method (not field) of the given
   * name (as seen by Starlark, not Java).
   *
   * <p>This constructor should be used only for ephemeral BuiltinFunction values created
   * transiently during a call such as {@code x.f()}, when the caller has already looked up the
   * MethodDescriptor using the same semantics as the thread that will be used in the call. Use the
   * other (slower) constructor if there is any possibility that the semantics of the {@code x.f}
   * operation differ from those of the thread used in the call.
   */
  BuiltinFunction(Object obj, String methodName, MethodDescriptor desc) {
    Preconditions.checkArgument(!desc.isStructField());
    this.obj = obj;
    this.methodName = methodName;
    this.desc = desc;
  }

  private MethodDescriptor getMethodDescriptor(StarlarkSemantics semantics) {
    MethodDescriptor desc = this.desc;
    if (desc == null) {
      desc = CallUtils.getAnnotatedMethods(semantics, obj.getClass()).get(methodName);
      Preconditions.checkArgument(
          !desc.isStructField(),
          "BuiltinFunction constructed for MethodDescriptor(structField=True)");
    }
    return desc;
  }

  /**
   * Returns the StarlarkMethod annotation of this Starlark-callable Java method.
   */
  public StarlarkMethod getAnnotation() {
    return getMethodDescriptor(StarlarkSemantics.DEFAULT).getAnnotation();
  }

  @Override
  public String getName() {
    return methodName;
  }

  @Override
  public void repr(Printer printer) {
    if (obj instanceof StarlarkValue || obj instanceof String) {
      printer
          .append("<built-in method ")
          .append(methodName)
          .append(" of ")
          .append(Starlark.type(obj))
          .append(" value>");
    } else {
      printer.append("<built-in function ").append(methodName).append(">");
    }
  }

  @Override
  public String toString() {
    return methodName;
  }

  @Override
  public StarlarkCallable.ArgumentProcessor requestArgumentProcessor(StarlarkThread thread) {
    return new ArgumentProcessor(this, thread, getMethodDescriptor(thread.getSemantics()));
  }

  /**
   * ArgumentProcessor for a call to the StarlarkMethod-annotated Java method.
   *
   * <p>Allocation of the vector of actual arguments with the correct size happens at a constructor
   * time.
   *
   * <p>Processing of positional arguments happens in {@link #addPositionalArg}.
   *
   * <p>Processing of named arguments happens in {@link #addNamedArg}.
   *
   * <p>Setting default values for missing optionals, and setting special parameters happens in
   * {@link #call}.
   *
   * <p>Static checks ensure that positional parameters appear before named, and mandatory
   * positionals appear before optional. No additional memory allocation occurs in the common
   * (success) case. Flag-disabled parameters are skipped during argument matching, as if they do
   * not exist. They are instead assigned their flag-disabled values.
   */
  static class ArgumentProcessor extends StarlarkCallable.ArgumentProcessor {
    private final BuiltinFunction owner;
    private final MethodDescriptor desc;
    private final ParamDescriptor[] parameters;
    private final Object[] vector;
    @Nullable private ArrayList<Object> varArgs;
    @Nullable private LinkedHashMap<String, Object> kwargs;
    private int paramIndex;
    private int argIndex;
    private boolean allPositionalParamsFilled;
    private int unexpectedPositionalArgCount;

    /**
     * Constructs an ArgumentProcessor for a call to the StarlarkMethod-annotated Java method.
     *
     * <p>The only work done at construction time is allocating the argument vector, and, only if
     * the method accepts extra args and/or extra kwargs, allocating the varArgs list and/or kwargs
     * map.
     *
     * @param thread the Starlark thread for the call
     * @param desc descriptor for the StarlarkMethod-annotated method
     */
    ArgumentProcessor(BuiltinFunction owner, StarlarkThread thread, MethodDescriptor desc) {
      super(thread);
      this.owner = owner;
      this.desc = desc;
      this.parameters = desc.getParameters();
      varArgs = null;
      kwargs = null;
      paramIndex = 0;
      argIndex = 0;
      allPositionalParamsFilled = false;
      unexpectedPositionalArgCount = 0;

      int n = parameters.length;
      if (desc.acceptsExtraArgs()) {
        varArgs = new ArrayList<>();
        n++;
      }
      if (desc.acceptsExtraKwargs()) {
        kwargs = Maps.newLinkedHashMapWithExpectedSize(1);
        n++;
      }
      if (desc.isUseStarlarkThread()) {
        n++;
      }
      this.vector = new Object[n];

      if (owner.obj instanceof String) {
        // String methods get the string as an extra argument
        // because their true receiver is StringModule.INSTANCE.
        vector[paramIndex++] = owner.obj;
      }
    }

    @Nullable
    private ParamDescriptor getNextEnabledPositionalParam() {
      while (!allPositionalParamsFilled && paramIndex < parameters.length) {
        ParamDescriptor param = parameters[paramIndex];
        if (!param.isPositional()) {
          allPositionalParamsFilled = true;
          return null;
        }
        if (param.disabledByFlag() == null) {
          return param;
        }
        paramIndex++;
      }
      return null;
    }

    @Override
    public void addPositionalArg(Object value) throws EvalException {
      ParamDescriptor param = getNextEnabledPositionalParam();
      if (param != null) {
        checkParamValue(param, value);
        vector[paramIndex++] = value;
        argIndex++;
      } else if (varArgs != null) {
        varArgs.add(value);
      } else {
        unexpectedPositionalArgCount++;
      }
    }

    @Override
    public void addNamedArg(String name, Object value) throws EvalException {
      // look up parameter
      int index = desc.getParameterIndex(name);
      // unknown parameter?
      if (index < 0) {
        // spill to **kwargs
        if (kwargs == null) {
          ImmutableList<String> allNames =
              Arrays.stream(parameters)
                  .map(ParamDescriptor::getName)
                  .collect(ImmutableList.toImmutableList());
          pushCallableAndThrow(
              Starlark.errorf(
                  "%s() got unexpected keyword argument '%s'%s",
                  owner.methodName, name, SpellChecker.didYouMean(name, allNames)));
        }

        // duplicate named argument?
        if (kwargs.put(name, value) != null) {
          pushCallableAndThrow(
              Starlark.errorf(
                  "%s() got multiple values for keyword argument '%s'", owner.methodName, name));
        }
        return;
      }
      ParamDescriptor param = parameters[index];

      // positional-only param?
      if (!param.isNamed()) {
        // spill to **kwargs
        if (kwargs == null) {
          pushCallableAndThrow(
              Starlark.errorf(
                  "%s() got named argument for positional-only parameter '%s'",
                  owner.methodName, name));
        }

        // duplicate named argument?
        if (kwargs.put(name, value) != null) {
          pushCallableAndThrow(
              Starlark.errorf(
                  "%s() got multiple values for keyword argument '%s'", owner.methodName, name));
        }
        return;
      }

      // disabled?
      String flag = param.disabledByFlag();
      if (flag != null) {
        // spill to **kwargs
        // TODO(b/380824219): Disabled named parameters should be skipped, no matter whether kwargs
        // is null or not. Disabled parameters should not be spilled to **kwargs.
        if (kwargs == null) {
          pushCallableAndThrow(
              Starlark.errorf(
                  "in call to %s(), parameter '%s' is %s",
                  owner.methodName, param.getName(), disabled(flag, thread.getSemantics())));
        }

        // duplicate named argument?
        if (kwargs.put(name, value) != null) {
          pushCallableAndThrow(
              Starlark.errorf(
                  "%s() got multiple values for keyword argument '%s'", owner.methodName, name));
        }
        return;
      }

      checkParamValue(param, value);

      // duplicate?
      if (vector[index] != null) {
        pushCallableAndThrow(
            Starlark.errorf("%s() got multiple values for argument '%s'", owner.methodName, name));
      }

      vector[index] = value;
    }

    private void checkParamValue(ParamDescriptor param, Object value) throws EvalException {
      List<Class<?>> allowedClasses = param.getAllowedClasses();
      if (allowedClasses == null) {
        return;
      }

      // Value must belong to one of the specified classes.
      boolean ok = false;
      for (Class<?> cls : allowedClasses) {
        if (cls.isInstance(value)) {
          ok = true;
          break;
        }
      }
      if (!ok) {
        pushCallableAndThrow(
            Starlark.errorf(
                "in call to %s(), parameter '%s' got value of type '%s', want '%s'",
                owner.methodName,
                param.getName(),
                Starlark.type(value),
                param.getTypeErrorMessage()));
      }
    }

    @Override
    public StarlarkCallable getCallable() {
      return owner;
    }

    @Override
    public Object call(StarlarkThread thread) throws EvalException, InterruptedException {
      if (unexpectedPositionalArgCount > 0) {
        if (argIndex == 0) {
          throw Starlark.errorf("%s() got unexpected positional argument", owner.methodName);
        } else {
          throw Starlark.errorf(
              "%s() accepts no more than %d positional argument%s but got %d",
              owner.methodName,
              argIndex,
              plural(argIndex),
              argIndex + unexpectedPositionalArgCount);
        }
      }

      owner.applyDefaultsReportMissingArgs(parameters, vector);

      int i = parameters.length;
      if (desc.acceptsExtraArgs()) {
        vector[i++] = Tuple.wrap(varArgs.toArray());
      }
      if (desc.acceptsExtraKwargs()) {
        vector[i++] = Dict.wrap(thread.mutability(), kwargs);
      }
      if (desc.isUseStarlarkThread()) {
        vector[i++] = thread;
      }

      return desc.call(
          owner.obj instanceof String ? StringModule.INSTANCE : owner.obj,
          vector,
          thread.mutability());
    }
  }

  private void applyDefaultsReportMissingArgs(ParamDescriptor[] parameters, Object[] vector)
      throws EvalException {
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
  }

  private static String plural(int n) {
    return n == 1 ? "" : "s";
  }

  // Returns a phrase meaning "disabled" appropriate to the specified flag.
  private static String disabled(String flag, StarlarkSemantics semantics) {
    // If the flag is True, it must be a deprecation flag. Otherwise it's an experimental flag.
    // TODO(adonovan): is that assumption sound?
    if (semantics.getBool(flag)) {
      return String.format(
          "deprecated and will be removed soon. It may be temporarily re-enabled by setting"
              + " --%s=false",
          flag.substring(1)); // remove [+-] prefix
    } else {
      return String.format(
          "experimental and thus unavailable with the current flags. It may be enabled by setting"
              + " --%s",
          flag.substring(1)); // remove [+-] prefix
    }
  }
}
