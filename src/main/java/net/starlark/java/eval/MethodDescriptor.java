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

import com.google.common.base.Preconditions;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.WrongMethodTypeException;
import java.lang.reflect.Method;
import java.util.Arrays;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;

/**
 * A value class to store Methods with their corresponding {@link StarlarkMethod} annotation
 * metadata. This is needed because the annotation is sometimes in a superclass.
 *
 * <p>The annotation metadata is duplicated in this class to avoid usage of Java dynamic proxies
 * which are ~7× slower.
 */
final class MethodDescriptor {
  private final Method method;
  private final MethodHandle methodHandle;
  private final StarlarkMethod annotation;

  private final String name;
  private final String doc;
  private final boolean documented;
  private final boolean structField;
  private final ParamDescriptor[] parameters;
  private final boolean extraPositionals;
  private final boolean extraKeywords;
  private final boolean selfCall;
  private final boolean allowReturnNones;
  private final boolean useStarlarkThread;
  private final boolean useStarlarkSemantics;

  private MethodDescriptor(
      Method method,
      StarlarkMethod annotation,
      String name,
      String doc,
      boolean documented,
      boolean structField,
      ParamDescriptor[] parameters,
      boolean extraPositionals,
      boolean extraKeywords,
      boolean selfCall,
      boolean allowReturnNones,
      boolean useStarlarkThread,
      boolean useStarlarkSemantics) {
    boolean isStringModule = method.getDeclaringClass() == StringModule.class;

    this.method = method;
    this.annotation = annotation;
    this.name = name;
    this.doc = doc;
    this.documented = documented;
    this.structField = structField;
    this.parameters =
        !isStringModule ? parameters : Arrays.copyOfRange(parameters, 1, parameters.length);
    this.extraPositionals = extraPositionals;
    this.extraKeywords = extraKeywords;
    this.selfCall = selfCall;
    this.allowReturnNones = allowReturnNones;
    this.useStarlarkThread = useStarlarkThread;
    this.useStarlarkSemantics = useStarlarkSemantics;

    MethodHandle mh;
    try {
      mh = LOOKUP.unreflect(method);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    }

    if (isStringModule) {
      // String methods get the string as an extra argument
      // because their true receiver is StringModule.INSTANCE.
      mh = MethodHandles.insertArguments(mh, 0, StringModule.INSTANCE);
    }

    // We generate different method handles for field calls and for method calls for performance and
    // simplicity

    if (!structField) {
      // Resulting method handle accepts StarlarkThread argument
      // (to avoid conditional inclusion of StarlarkThread argument at call time)
      // so we need to drop the StarlarkThread argument if underlying method does not accept
      // StarlarkThread.
      if (!useStarlarkThread) {
        mh = MethodHandles.dropArguments(mh, mh.type().parameterCount(), StarlarkThread.class);
      }

      // mh: (self, args..., StarlarkThread) -> ?

      // Move non-special arguments to the end of parameter list so we could fold them into a
      // single array parameter.
      mh = moveArgument(mh, mh.type().parameterCount() - 1, 1);

      // mh: (self, StarlarkThread, args...) -> ?

      // Convert non-special arguments into single Object[] argument
      mh = mh.asSpreader(Object[].class, mh.type().parameterCount() - 2);

      // mh: (self, StarlarkThread, args) -> ?

      // This operation "filters" return value of the underlying method.
      // The resuling operation has extra Mutability argument,
      // because fromJava functions requires it.
      mh = filterReturnTypeToCompatibleWithStarlark(mh, allowReturnNones, name);

      // mh: (self, StarlarkThread, args, Mutability) -> ?

      // Now we convert Mutability argument to StarlarkThread argument.
      // We do that to reduce the number of call arguments by one to make reflective invocation a
      // little cheaper
      mh = MethodHandles.filterArguments(mh, 3, THREAD_GET_MUTABILITY);

      // mh: (self, StarlarkThread, args, StarlarkThread) -> ?

      // Fold two StarlarkArguments into single StarlarkThread argument
      mh = MethodHandles.permuteArguments(mh, mh.type().dropParameterTypes(3, 4), 0, 1, 2, 1);

      // mh: (self, StarlarkThread, args) -> ?

      // The final step converts return type to Object (to be able to use invokeExact)
      // but also validates that the method handle we have created has a correct type.
      mh = MethodHandles.explicitCastArguments(mh, CALL_NON_STRUCT_FIELD_METHOD_TYPE);

      // mh: (self, StarlarkThread, args) -> Object
    } else {
      // Similarly to above, if underlying method does not accept StarlarkSemantics,
      // we add ignored StarlarkSemantics argument to make resulting method type the same
      // regardles of underlying method signature.
      if (!useStarlarkSemantics) {
        mh = MethodHandles.dropArguments(mh, mh.type().parameterCount(), StarlarkSemantics.class);
      }

      // mh: (self, StarlarkSemantics) -> ?

      // Convert return type, same as above
      mh = filterReturnTypeToCompatibleWithStarlark(mh, allowReturnNones, name);

      // mh: (self, StarlarkSemantics, Mutability) -> ?

      // Convert return to Object and assert type correct
      mh = MethodHandles.explicitCastArguments(mh, CALL_STRUCT_FIELD_METHOD_TYPE);

      // mh: (self, StarlarkSemantics, Mutability) -> Object
    }

    this.methodHandle = mh;
  }

  /**
   * Take a method handle and return a method handle with additional {@link Mutability} argument.
   * Resulting method handle ensures that method returns a valid starlark object.
   */
  private static MethodHandle filterReturnTypeToCompatibleWithStarlark(
      MethodHandle mh, boolean allowReturnNones, String name) {
    if (mh.type().returnType() == void.class) {
      // The easiest case: if underlying method returns void,
      // we convert MethodHandle to return None object.
      mh =
          MethodHandles.filterReturnValue(
              mh, MethodHandles.constant(NoneType.class, Starlark.NONE));
      mh = MethodHandles.dropArguments(mh, mh.type().parameterCount(), Mutability.class);
    } else if (methodReturnTypeIsGoodAsIsForStarlark(mh.type().returnType())) {
      // If the underlying method return type is a value compatible with Starlark runtime,
      // we only perform null-handling.
      mh = handleNullReturn(mh, allowReturnNones, name);
      mh = MethodHandles.dropArguments(mh, mh.type().parameterCount(), Mutability.class);
    } else if (mh.type().returnType() == int.class) {
      mh = MethodHandles.filterReturnValue(mh, STARLARK_INT_OF_INT);
      mh = MethodHandles.dropArguments(mh, mh.type().parameterCount(), Mutability.class);
    } else if (mh.type().returnType() == long.class) {
      mh = MethodHandles.filterReturnValue(mh, STARLARK_INT_OF_LONG);
      mh = MethodHandles.dropArguments(mh, mh.type().parameterCount(), Mutability.class);
    } else {
      // Otherwise we invoke fromJava function to convert method return type
      // a value compatible with Starlark runtime.
      // Note if an underlying method returns a concrete type,
      // then fromJava invocation is efficiently inlined in MethodHandle object.
      mh = handleNullReturn(mh, allowReturnNones, name);

      MethodHandle fromJava =
          MethodHandles.explicitCastArguments(
              FROM_JAVA,
              MethodType.methodType(Object.class, mh.type().returnType(), Mutability.class));
      mh = MethodHandles.collectArguments(fromJava, 0, mh);
    }
    return mh;
  }

  /** @see Starlark#valid(Object) */
  private static boolean methodReturnTypeIsGoodAsIsForStarlark(Class<?> x) {
    return x == String.class
        || x == Boolean.class
        || x == boolean.class
        || StarlarkValue.class.isAssignableFrom(x);
  }

  private static MethodHandle handleNullReturn(
      MethodHandle mh, boolean allowReturnNones, String name) {
    // Primitive types are never null
    if (mh.type().returnType().isPrimitive()) {
      return mh;
    }

    if (allowReturnNones) {
      return nullToNone(mh);
    } else {
      return nullToException(mh, name);
    }
  }

  private static Object nullToNone(Object object) {
    return object != null ? object : Starlark.NONE;
  }

  private static Object nullToException(Object object, String methodName) {
    if (object == null) {
      throw new IllegalStateException("method invocation returned null: " + methodName);
    }
    return object;
  }

  private static MethodHandle nullToNone(MethodHandle mh) {
    MethodHandle filter =
        MethodHandles.explicitCastArguments(
            NULL_TO_NONE, MethodType.methodType(Object.class, mh.type().returnType()));
    return MethodHandles.filterReturnValue(mh, filter);
  }

  private static MethodHandle nullToException(MethodHandle mh, String name) {
    MethodHandle filter = MethodHandles.insertArguments(NULL_TO_EXCEPTION, 1, name);
    filter =
        MethodHandles.explicitCastArguments(
            filter, MethodType.methodType(mh.type().returnType(), mh.type().returnType()));
    return MethodHandles.filterReturnValue(mh, filter);
  }

  private static final MethodType CALL_NON_STRUCT_FIELD_METHOD_TYPE =
      MethodType.methodType(
          // return type
          Object.class,
          // self
          Object.class,
          StarlarkThread.class,
          // arguments
          Object[].class);

  private static final MethodType CALL_STRUCT_FIELD_METHOD_TYPE =
      MethodType.methodType(
          // return type
          Object.class,
          // self
          Object.class,
          StarlarkSemantics.class,
          Mutability.class);

  private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();

  // Cache method handles to avoid looking them up again and again
  private static final MethodHandle NULL_TO_NONE;
  private static final MethodHandle NULL_TO_EXCEPTION;
  private static final MethodHandle FROM_JAVA;
  private static final MethodHandle THREAD_GET_MUTABILITY;
  private static final MethodHandle STARLARK_INT_OF_INT;
  private static final MethodHandle STARLARK_INT_OF_LONG;

  static {
    try {
      NULL_TO_NONE =
          LOOKUP.findStatic(
              MethodDescriptor.class,
              "nullToNone",
              MethodType.methodType(Object.class, Object.class));
      NULL_TO_EXCEPTION =
          LOOKUP.findStatic(
              MethodDescriptor.class,
              "nullToException",
              MethodType.methodType(Object.class, Object.class, String.class));
      FROM_JAVA =
          LOOKUP.findStatic(
              Starlark.class,
              "fromJava",
              MethodType.methodType(Object.class, Object.class, Mutability.class));
      STARLARK_INT_OF_INT =
        LOOKUP.findStatic(StarlarkInt.class, "of", MethodType.methodType(StarlarkInt.class, int.class));
      STARLARK_INT_OF_LONG =
        LOOKUP.findStatic(StarlarkInt.class, "of", MethodType.methodType(StarlarkInt.class, long.class));
      THREAD_GET_MUTABILITY =
          LOOKUP.findVirtual(
              StarlarkThread.class, "mutability", MethodType.methodType(Mutability.class));
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  /** Permutate method parameters, by moving {@code oldPos} parameter to {@code newPos}. */
  private static MethodHandle moveArgument(MethodHandle mh, int oldPos, int newPos) {
    if (oldPos == newPos) {
      return mh;
    } else if (newPos > oldPos) {
      throw new IllegalArgumentException("newPost must be <= oldPos");
    }

    int[] reorder = new int[mh.type().parameterCount()];
    for (int i = 0; i != reorder.length; ++i) {
      if (i < newPos) {
        reorder[i] = i;
      } else if (i < oldPos) {
        reorder[i] = i + 1;
      } else if (i == oldPos) {
        reorder[i] = newPos;
      } else {
        reorder[i] = i;
      }
    }
    return MethodHandles.permuteArguments(
        mh,
        mh.type()
            .dropParameterTypes(oldPos, oldPos + 1)
            .insertParameterTypes(newPos, mh.type().parameterType(oldPos)),
        reorder);
  }

  /** Returns the StarlarkMethod annotation corresponding to this method. */
  StarlarkMethod getAnnotation() {
    return annotation;
  }

  /** @return Starlark method descriptor for provided Java method and signature annotation. */
  static MethodDescriptor of(
      Method method, StarlarkMethod annotation, StarlarkSemantics semantics) {
    // This happens when the interface is public but the implementation classes
    // have reduced visibility.
    method.setAccessible(true);

    Class<?>[] paramClasses = method.getParameterTypes();
    Param[] paramAnnots = annotation.parameters();
    ParamDescriptor[] params = new ParamDescriptor[paramAnnots.length];
    Arrays.setAll(params, i -> ParamDescriptor.of(paramAnnots[i], paramClasses[i], semantics));

    return new MethodDescriptor(
        method,
        annotation,
        annotation.name(),
        annotation.doc(),
        annotation.documented(),
        annotation.structField(),
        params,
        !annotation.extraPositionals().name().isEmpty(),
        !annotation.extraKeywords().name().isEmpty(),
        annotation.selfCall(),
        annotation.allowReturnNones(),
        annotation.useStarlarkThread(),
        annotation.useStarlarkSemantics());
  }

  /**
   * Calls this method, which must have {@code structField=true}.
   *
   * <p>The Mutability is used if it is necessary to allocate a Starlark copy of a Java result.
   */
  Object callField(Object obj, StarlarkSemantics semantics, @Nullable Mutability mu)
      throws EvalException, InterruptedException {
    if (!structField) {
      throw new IllegalStateException("not a struct field: " + name);
    }
    Preconditions.checkNotNull(obj);
    try {
      return methodHandle.invokeExact(obj, semantics, mu);
    } catch (RuntimeException | Error | EvalException | InterruptedException e) {
      throw e;
    } catch (Throwable e) {
      throw wrapOtherException(e);
    }
  }

  /**
   * Invokes this method using {@code obj} as a target and {@code args} as Java arguments.
   *
   * <p>Methods with {@code void} return type return {@code None} following Python convention.
   */
  Object call(Object obj, StarlarkThread starlarkThread, Object[] args)
      throws EvalException, InterruptedException {
    if (structField) {
      throw new IllegalStateException("a struct field: " + name);
    }
    Preconditions.checkNotNull(obj);
    try {
      return methodHandle.invokeExact(obj, starlarkThread, args);
    } catch (WrongMethodTypeException ex) {
      // "Can't happen": unexpected type mismatch.
      // Show details to aid debugging (see e.g. b/162444744).
      StringBuilder buf = new StringBuilder();
      buf.append(
          String.format(
              "IllegalArgumentException (%s) in Starlark call of %s, obj=%s (%s), args=[",
              ex.getMessage(), method, Starlark.repr(obj), Starlark.type(obj)));
      String sep = "";
      for (Object arg : args) {
        buf.append(String.format("%s%s (%s)", sep, Starlark.repr(arg), Starlark.type(arg)));
        sep = ", ";
      }
      buf.append(']');
      throw new IllegalArgumentException(buf.toString());
    } catch (RuntimeException | Error | EvalException | InterruptedException e) {
      throw e;
    } catch (Throwable e) {
      // All other checked exceptions (e.g. LabelSyntaxException) are reported to Starlark.
      throw wrapOtherException(e);
    }
  }

  private EvalException wrapOtherException(Throwable e) throws EvalException {
    // All other checked exceptions (e.g. LabelSyntaxException) are reported to Starlark.
    return new EvalException(e);
  }

  /** @see StarlarkMethod#name() */
  String getName() {
    return name;
  }

  Method getMethod() {
    return method;
  }

  /** @see StarlarkMethod#structField() */
  boolean isStructField() {
    return structField;
  }

  /** @see StarlarkMethod#useStarlarkThread() */
  boolean isUseStarlarkThread() {
    return useStarlarkThread;
  }

  /** @see StarlarkMethod#useStarlarkSemantics() */
  boolean isUseStarlarkSemantics() {
    return useStarlarkSemantics;
  }

  /** @see StarlarkMethod#allowReturnNones() */
  boolean isAllowReturnNones() {
    return allowReturnNones;
  }

  /** @return {@code true} if this method accepts extra arguments ({@code *args}) */
  boolean acceptsExtraArgs() {
    return extraPositionals;
  }

  /** @see StarlarkMethod#extraKeywords() */
  boolean acceptsExtraKwargs() {
    return extraKeywords;
  }

  /** @see StarlarkMethod#parameters() */
  ParamDescriptor[] getParameters() {
    return parameters;
  }

  /** Returns the index of the named parameter or -1 if not found. */
  int getParameterIndex(String name) {
    for (int i = 0; i < parameters.length; i++) {
      if (parameters[i].getName().equals(name)) {
        return i;
      }
    }
    return -1;
  }

  /** @see StarlarkMethod#documented() */
  boolean isDocumented() {
    return documented;
  }

  /** @see StarlarkMethod#doc() */
  String getDoc() {
    return doc;
  }

  /** @see StarlarkMethod#selfCall() */
  boolean isSelfCall() {
    return selfCall;
  }
}
