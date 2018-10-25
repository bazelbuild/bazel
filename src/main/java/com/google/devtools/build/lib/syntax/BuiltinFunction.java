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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.Environment.LexicalFrame;
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.NoSuchElementException;
import javax.annotation.Nullable;

/**
 * A class for Skylark functions provided as builtins by the Skylark implementation. Instances of
 * this class do not need to be serializable because they should effectively be treated as
 * constants.
 */
public class BuiltinFunction extends BaseFunction {

  /** ExtraArgKind so you can tweek your function's own calling convention */
  public enum ExtraArgKind {
    LOCATION,
    SYNTAX_TREE,
    ENVIRONMENT;
  }
  // Predefined system add-ons to function signatures
  public static final ExtraArgKind[] USE_LOC =
      new ExtraArgKind[] {ExtraArgKind.LOCATION};
  public static final ExtraArgKind[] USE_LOC_ENV =
      new ExtraArgKind[] {ExtraArgKind.LOCATION, ExtraArgKind.ENVIRONMENT};
  public static final ExtraArgKind[] USE_AST =
      new ExtraArgKind[] {ExtraArgKind.SYNTAX_TREE};
  public static final ExtraArgKind[] USE_AST_ENV =
      new ExtraArgKind[] {ExtraArgKind.SYNTAX_TREE, ExtraArgKind.ENVIRONMENT};

  // Builtins cannot create or modify variable bindings. So it's sufficient to use a shared
  // instance.
  private static final LexicalFrame SHARED_LEXICAL_FRAME_FOR_BUILTIN_FUNCTION_CALLS =
      LexicalFrame.create(Mutability.IMMUTABLE);

  // The underlying invoke() method.
  @Nullable private Method invokeMethod;

  // extra arguments required beside signature.
  @Nullable private ExtraArgKind[] extraArgs;

  // The count of arguments in the inner invoke method,
  // to be used as size of argument array by the outer call method.
  private int innerArgumentCount;

  // The returnType of the method.
  private Class<?> returnType;

  // True if the function is a rule class
  private boolean isRule;

  /** Create unconfigured function from its name */
  public BuiltinFunction(String name) {
    super(name);
  }

  /** Creates an unconfigured BuiltinFunction with the given name and defaultValues */
  public BuiltinFunction(String name, Iterable<Object> defaultValues) {
    super(name, defaultValues);
  }

  /** Creates a BuiltinFunction with the given name and signature */
  public BuiltinFunction(String name, FunctionSignature signature) {
    super(name, signature);
    configure();
  }

  /** Creates a BuiltinFunction with the given name and signature with values */
  public BuiltinFunction(String name,
      FunctionSignature.WithValues<Object, SkylarkType> signature) {
    super(name, signature);
    configure();
  }

  /** Creates a BuiltinFunction with the given name and signature and extra arguments */
  public BuiltinFunction(String name, FunctionSignature signature, ExtraArgKind[] extraArgs) {
    super(name, signature);
    this.extraArgs = extraArgs;
    configure();
  }

  /** Creates a BuiltinFunction with the given name, signature, extra arguments, and a rule flag */
  public BuiltinFunction(
      String name, FunctionSignature signature, ExtraArgKind[] extraArgs, boolean isRule) {
    super(name, signature);
    this.extraArgs = extraArgs;
    this.isRule = isRule;
    configure();
  }

  /** Creates a BuiltinFunction with the given name, signature with values, and extra arguments */
  public BuiltinFunction(String name,
      FunctionSignature.WithValues<Object, SkylarkType> signature, ExtraArgKind[] extraArgs) {
    super(name, signature);
    this.extraArgs = extraArgs;
    configure();
  }

  /** Creates a BuiltinFunction from the given name and a Factory */
  public BuiltinFunction(String name, Factory factory) {
    super(name);
    configure(factory);
  }

  @Override
  protected int getArgArraySize () {
    return innerArgumentCount;
  }

  protected ExtraArgKind[] getExtraArgs () {
    return extraArgs;
  }

  @Override
  @Nullable
  public Object call(Object[] args, @Nullable FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(env);

    // ast is null when called from Java (as there's no Skylark call site).
    Location loc = ast == null ? Location.BUILTIN : ast.getLocation();

    // Add extra arguments, if needed
    if (extraArgs != null) {
      int i = args.length - extraArgs.length;
      for (BuiltinFunction.ExtraArgKind extraArg : extraArgs) {
        switch(extraArg) {
          case LOCATION:
            args[i] = loc;
            break;

          case SYNTAX_TREE:
            args[i] = ast;
            break;

          case ENVIRONMENT:
            args[i] = env;
            break;
        }
        i++;
      }
    }

    // Last but not least, actually make an inner call to the function with the resolved arguments.
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.SKYLARK_BUILTIN_FN, getName())) {
      env.enterScope(this, SHARED_LEXICAL_FRAME_FOR_BUILTIN_FUNCTION_CALLS, ast, env.getGlobals());
      return invokeMethod.invoke(this, args);
    } catch (InvocationTargetException x) {
      Throwable e = x.getCause();

      if (e instanceof EvalException) {
        throw ((EvalException) e).ensureLocation(loc);
      } else if (e instanceof IllegalArgumentException) {
        throw new EvalException(loc, "illegal argument in call to " + getName(), e);
      }
      Throwables.throwIfInstanceOf(e, InterruptedException.class);
      Throwables.throwIfUnchecked(e);
      throw badCallException(loc, e, args);
    } catch (IllegalArgumentException e) {
      // Either this was thrown by Java itself, or it's a bug
      // To cover the first case, let's manually check the arguments.
      final int len = args.length - ((extraArgs == null) ? 0 : extraArgs.length);
      final Class<?>[] types = invokeMethod.getParameterTypes();
      for (int i = 0; i < args.length; i++) {
        if (args[i] != null && !types[i].isAssignableFrom(args[i].getClass())) {
          String paramName =
              i < len ? signature.getSignature().getNames().get(i) : extraArgs[i - len].name();
          throw new EvalException(
              loc,
              String.format(
                  "argument '%s' has type '%s', but should be '%s'\nin call to builtin %s %s",
                  paramName,
                  EvalUtils.getDataTypeName(args[i]),
                  EvalUtils.getDataTypeNameFromClass(types[i]),
                  hasSelfArgument() ? "method" : "function",
                  getShortSignature()));
        }
      }
      throw badCallException(loc, e, args);
    } catch (IllegalAccessException e) {
      throw badCallException(loc, e, args);
    } finally {
      env.exitScope();
    }
  }

  private static String stacktraceToString(StackTraceElement[] elts) {
    StringBuilder b = new StringBuilder();
    for (StackTraceElement e : elts) {
      b.append(e);
      b.append("\n");
    }
    return b.toString();
  }

  private IllegalStateException badCallException(Location loc, Throwable e, Object... args) {
    // If this happens, it's a bug in our code.
    return new IllegalStateException(
        String.format(
            "%s%s (%s)\n"
                + "while calling %s with args %s\n"
                + "Java parameter types: %s\nStarlark type checks: %s",
            (loc == null) ? "" : loc + ": ",
            Arrays.asList(args),
            e.getClass().getName(),
            stacktraceToString(e.getStackTrace()),
            this,
            Arrays.asList(invokeMethod.getParameterTypes()),
            signature.getTypes()),
        e);
  }

  /** Configure the reflection mechanism */
  @Override
  public void configure(SkylarkSignature annotation) {
    Preconditions.checkState(!isConfigured()); // must not be configured yet
    enforcedArgumentTypes = new ArrayList<>();
    this.extraArgs = SkylarkSignatureProcessor.getExtraArgs(annotation);
    this.returnType = annotation.returnType();
    super.configure(annotation);
  }

  // finds the method and makes it accessible (which is needed to find it, and later to use it)
  protected Method findMethod(final String name) {
    Method found = null;
    for (Method method : this.getClass().getDeclaredMethods()) {
      method.setAccessible(true);
      if (name.equals(method.getName())) {
        if (found != null) {
          throw new IllegalArgumentException(String.format(
              "function %s has more than one method named %s", getName(), name));
        }
        found = method;
      }
    }
    if (found == null) {
      throw new NoSuchElementException(String.format(
          "function %s doesn't have a method named %s", getName(), name));
    }
    return found;
  }

  /** Configure the reflection mechanism */
  @Override
  protected void configure() {
    invokeMethod = findMethod("invoke");

    int arguments = signature.getSignature().getShape().getArguments();
    innerArgumentCount = arguments + (extraArgs == null ? 0 : extraArgs.length);
    Class<?>[] parameterTypes = invokeMethod.getParameterTypes();
    if (innerArgumentCount != parameterTypes.length) {
      // Guard message construction by check to avoid autoboxing two integers.
      throw new IllegalStateException(
          String.format(
              "bad argument count for %s: method has %s arguments, type list has %s",
              getName(),
              innerArgumentCount,
              parameterTypes.length));
    }

    if (enforcedArgumentTypes != null) {
      for (int i = 0; i < arguments; i++) {
        SkylarkType enforcedType = enforcedArgumentTypes.get(i);
        if (enforcedType != null) {
          Class<?> parameterType = parameterTypes[i];
          String msg = String.format(
              "fun %s(%s), param %s, enforcedType: %s (%s); parameterType: %s",
              getName(), signature, signature.getSignature().getNames().get(i),
              enforcedType, enforcedType.getType(), parameterType);
          if (enforcedType instanceof SkylarkType.Simple
              || enforcedType instanceof SkylarkFunctionType) {
            Preconditions.checkArgument(
                enforcedType.getType() == parameterType, msg);
            // No need to enforce Simple types on the Skylark side, the JVM will do it for us.
            enforcedArgumentTypes.set(i, null);
          } else if (enforcedType instanceof SkylarkType.Combination) {
            Preconditions.checkArgument(enforcedType.getType() == parameterType, msg);
          } else {
            Preconditions.checkArgument(
                parameterType == Object.class || parameterType == null, msg);
          }
        }
      }
    }
    // No need for the enforcedArgumentTypes List if all the types were Simple
    enforcedArgumentTypes = FunctionSignature.valueListOrNull(enforcedArgumentTypes);

    if (returnType != null) {
      Class<?> type = returnType;
      Class<?> methodReturnType = invokeMethod.getReturnType();
      Preconditions.checkArgument(type == methodReturnType,
          "signature for function %s says it returns %s but its invoke method returns %s",
          getName(), returnType, methodReturnType);
    }
  }

  /** Configure by copying another function's configuration */
  // Alternatively, we could have an extension BuiltinFunctionSignature of FunctionSignature,
  // and use *that* instead of a Factory.
  public void configure(BuiltinFunction.Factory factory) {
    // this function must not be configured yet, but the factory must be
    Preconditions.checkState(!isConfigured());
    Preconditions.checkState(factory.isConfigured(),
        "function factory is not configured for %s", getName());

    this.paramDoc = factory.getParamDoc();
    this.signature = factory.getSignature();
    this.extraArgs = factory.getExtraArgs();
    this.objectType = factory.getObjectType();
    configure();
  }

  /**
   * A Factory allows for a @SkylarkSignature annotation to be provided and processed in advance
   * for a function that will be defined later as a closure (see e.g. in PackageFactory).
   *
   * <p>Each instance of this class must define a method create that closes over some (final)
   * variables and returns a BuiltinFunction.
   */
  public abstract static class Factory extends BuiltinFunction {
    @Nullable private Method createMethod;

    /** Create unconfigured function Factory from its name */
    public Factory(String name) {
      super(name);
    }

    /** Creates an unconfigured function Factory with the given name and defaultValues */
    public Factory(String name, Iterable<Object> defaultValues) {
      super(name, defaultValues);
    }

    @Override
    public void configure() {
      if (createMethod != null) {
        return;
      }
      createMethod = findMethod("create");
    }

    @Override
    public Object call(Object[] args, @Nullable FuncallExpression ast, Environment env)
        throws EvalException {
      throw new EvalException(null, "tried to invoke a Factory for function " + this);
    }

    /** Instantiate the Factory
     * @param args arguments to pass to the create method
     * @return a new BuiltinFunction that closes over the arguments
     */
    public BuiltinFunction apply(Object... args) {
      try {
        return (BuiltinFunction) createMethod.invoke(this, args);
      } catch (InvocationTargetException | IllegalArgumentException | IllegalAccessException e) {
        throw new RuntimeException(String.format(
            "Exception while applying BuiltinFunction.Factory %s: %s",
            this, e.getMessage()), e);
      }
    }
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    if (isRule) {
      printer.append("<built-in rule " + getName() + ">");
    } else {
      printer.append("<built-in function " + getName() + ">");
    }
  }
}
