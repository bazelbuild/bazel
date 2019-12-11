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

import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A BuiltinCallable is a callable Starlark value that reflectively invokes a
 * SkylarkCallable-annotated method of a Java object.
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
  public Object callImpl(
      StarlarkThread thread, FuncallExpression call, List<Object> args, Map<String, Object> kwargs)
      throws EvalException, InterruptedException {
    MethodDescriptor methodDescriptor =
        desc != null ? desc : getMethodDescriptor(thread.getSemantics());
    Class<?> clazz;
    Object objValue;

    if (obj instanceof String) {
      args.add(0, obj);
      clazz = StringModule.class;
      objValue = StringModule.INSTANCE;
    } else {
      clazz = obj.getClass();
      objValue = obj;
    }

    // TODO(cparsons): Profiling should be done at the MethodDescriptor level.
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.STARLARK_BUILTIN_FN, methodName)) {
      Object[] javaArguments =
          CallUtils.convertStarlarkArgumentsToJavaMethodArguments(
              thread, call, methodDescriptor, clazz, args, kwargs);
      return methodDescriptor.call(
          objValue, javaArguments, call.getLocation(), thread.mutability());
    }
  }

  private MethodDescriptor getMethodDescriptor(StarlarkSemantics semantics) {
    return CallUtils.getMethod(semantics, obj.getClass(), methodName);
  }

  /**
   * Returns the SkylarkCallable annotation of this Starlark-callable Java method.
   *
   * @deprecated This method is intended only for docgen, and uses the default semantics.
   */
  @Deprecated
  public SkylarkCallable getAnnotation() {
    return getMethodDescriptor(StarlarkSemantics.DEFAULT_SEMANTICS).getAnnotation();
  }

  @Override
  public String getName() {
    return methodName;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<built-in function " + methodName + ">");
  }
}
