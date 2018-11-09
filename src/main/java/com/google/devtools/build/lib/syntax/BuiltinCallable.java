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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.Environment.LexicalFrame;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A function-object abstraction on object methods exposed to skylark using {@link SkylarkCallable}.
 */
public class BuiltinCallable extends BaseFunction {

  /** Represents a required interpreter parameter as dictated by {@link SkylarkCallable} */
  public enum ExtraArgKind {
    LOCATION, // SkylarkCallable.useLocation
    SYNTAX_TREE, // SkylarkCallable.useAst
    ENVIRONMENT, // SkylarkCallable.useEnvironment
    SEMANTICS; // SkylarkCallable.useSemantics
  }

  // Builtins cannot create or modify variable bindings. So it's sufficient to use a shared
  // instance.
  private static final LexicalFrame SHARED_LEXICAL_FRAME_FOR_BUILTIN_METHOD_CALLS =
      LexicalFrame.create(Mutability.IMMUTABLE);

  private final Object obj;
  private final MethodDescriptor descriptor;
  private final List<ExtraArgKind> extraArgs;
  private int innerArgumentCount;

  public BuiltinCallable(String name, Object obj, MethodDescriptor descriptor) {
    super(name);
    this.obj = obj;
    this.descriptor = descriptor;
    this.extraArgs = getExtraArgs(descriptor);
    configure(obj, descriptor);
  }

  @Override
  protected int getArgArraySize () {
    return innerArgumentCount;
  }

  private static List<ExtraArgKind> getExtraArgs(MethodDescriptor method) {
    ImmutableList.Builder<ExtraArgKind> extraArgs = ImmutableList.builder();
    if (method.isUseLocation()) {
      extraArgs.add(ExtraArgKind.LOCATION);
    }
    if (method.isUseAst()) {
      extraArgs.add(ExtraArgKind.SYNTAX_TREE);
    }
    if (method.isUseEnvironment()) {
      extraArgs.add(ExtraArgKind.ENVIRONMENT);
    }
    if (method.isUseSkylarkSemantics()) {
      extraArgs.add(ExtraArgKind.SEMANTICS);
    }
    return extraArgs.build();
  }

  /** Configure a BaseFunction from a @SkylarkCallable-annotated method */
  private void configure(Object obj, MethodDescriptor descriptor) {
    Preconditions.checkState(!isConfigured()); // must not be configured yet

    this.paramDoc = new ArrayList<>();
    this.signature = SkylarkSignatureProcessor.getSignatureForCallable(
        getName(), descriptor, paramDoc, getEnforcedArgumentTypes());
    this.objectType = obj.getClass();
    this.innerArgumentCount = signature.getSignature().getShape().getArguments() + extraArgs.size();
    configure();
  }

  @Override
  @Nullable
  public Object call(Object[] args, @Nullable FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(env);

    // ast is null when called from Java (as there's no Skylark call site).
    Location loc = ast == null ? Location.BUILTIN : ast.getLocation();

    // Add extra arguments, if needed
    int index = args.length - extraArgs.size();
    for (ExtraArgKind extraArg : extraArgs) {
      switch(extraArg) {
        case LOCATION:
          args[index] = loc;
          break;

        case SYNTAX_TREE:
          args[index] = ast;
          break;

        case ENVIRONMENT:
          args[index] = env;
          break;

        case SEMANTICS:
          args[index] = env.getSemantics();
          break;
      }
      index++;
    }

    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.STARLARK_BUILTIN_FN, getName())) {
      env.enterScope(this, SHARED_LEXICAL_FRAME_FOR_BUILTIN_METHOD_CALLS, ast, env.getGlobals());
      return descriptor.call(obj, args, ast.getLocation(), env);
    } finally {
      env.exitScope();
    }
  }
}
