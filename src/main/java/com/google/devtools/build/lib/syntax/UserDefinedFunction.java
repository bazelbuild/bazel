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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.LoopLabels;
import com.google.devtools.build.lib.syntax.compiler.ReflectionUtils;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.vfs.PathFragment;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.asm.ClassVisitorWrapper;
import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.description.modifier.MethodManifestation;
import net.bytebuddy.description.modifier.Ownership;
import net.bytebuddy.description.modifier.TypeManifestation;
import net.bytebuddy.description.modifier.Visibility;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.dynamic.DynamicType.Unloaded;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.MethodDelegation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.matcher.ElementMatchers;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.util.Textifier;
import org.objectweb.asm.util.TraceClassVisitor;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The actual function registered in the environment. This function is defined in the
 * parsed code using {@link FunctionDefStatement}.
 */
public class UserDefinedFunction extends BaseFunction {

  private final ImmutableList<Statement> statements;

  // we close over the globals at the time of definition
  private final Environment.Frame definitionGlobals;

  private Optional<Method> method;
  // TODO(bazel-team) make this configurable once the compiler is stable
  public static boolean debugCompiler = false;
  public static boolean debugCompilerPrintByteCode = false;
  private static File debugFolder;
  public static boolean enableCompiler = false;

  protected UserDefinedFunction(
      Identifier function,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      ImmutableList<Statement> statements,
      Environment.Frame definitionGlobals)
      throws EvalException {
    super(function.getName(), signature, function.getLocation());
    this.statements = statements;
    this.definitionGlobals = definitionGlobals;
    method = enableCompiler ? buildCompiledFunction() : Optional.<Method>absent();
  }

  public FunctionSignature.WithValues<Object, SkylarkType> getFunctionSignature() {
    return signature;
  }

  ImmutableList<Statement> getStatements() {
    return statements;
  }

  @Override
  public Object call(Object[] arguments, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    if (!env.mutability().isMutable()) {
      throw new EvalException(getLocation(), "Trying to call in frozen environment");
    }
    if (env.getStackTrace().contains(this)) {
      throw new EvalException(getLocation(),
          String.format("Recursion was detected when calling '%s' from '%s'",
              getName(), Iterables.getLast(env.getStackTrace()).getName()));
    }

    if (enableCompiler && method.isPresent()) {
      Object returnValue = callCompiledFunction(arguments, ast, env);
      if (returnValue != null) {
        return returnValue;
      }
    }

    Profiler.instance().startTask(ProfilerTask.SKYLARK_USER_FN, getName());
    try {
      env.enterScope(this, ast, definitionGlobals);
      ImmutableList<String> names = signature.getSignature().getNames();

      // Registering the functions's arguments as variables in the local Environment
      int i = 0;
      for (String name : names) {
        env.update(name, arguments[i++]);
      }

      try {
        for (Statement stmt : statements) {
          if (stmt instanceof ReturnStatement) {
            // Performance optimization.
            // Executing the statement would throw an exception, which is slow.
            return ((ReturnStatement) stmt).getReturnExpression().eval(env);
          } else {
            stmt.exec(env);
          }
        }
      } catch (ReturnStatement.ReturnException e) {
        return e.getValue();
      }
      return Runtime.NONE;
    } finally {
      Profiler.instance().completeTask(ProfilerTask.SKYLARK_USER_FN);
      env.exitScope();
    }
  }

  private Object callCompiledFunction(Object[] arguments, FuncallExpression ast, Environment env) {
    compilerDebug("Calling compiled function " + getLocationPathAndLine() + " " + getName());
    try {
      Profiler.instance().startTask(ProfilerTask.SKYLARK_USER_COMPILED_FN,
          getLocationPathAndLine() + "#" + getName());
      env.enterScope(this, ast, definitionGlobals);

      return method
          .get()
          .invoke(null, ImmutableList.builder().add(arguments).add(env).build().toArray());

    } catch (IllegalAccessException e) {
      // this should never happen
      throw new RuntimeException(
          "Compiler created code that could not be accessed reflectively.", e);
    } catch (InvocationTargetException e) {
      compilerDebug("Error running compiled version", e.getCause());
      return null;
    } finally {
      Profiler.instance().completeTask(ProfilerTask.SKYLARK_USER_COMPILED_FN);
      env.exitScope();
    }
  }

  /**
   * Generates a subclass of {@link CompiledFunction} with a static method "call" and static
   * methods for getting information from a {@link DebugInfo} instance.
   *
   * <p>The "call" method contains the compiled version of this function's AST.
   */
  private Optional<Method> buildCompiledFunction() throws EvalException {
    // replace the / character in the path so we have file system compatible class names
    // the java specification mentions that $ should be used in generated code
    // see http://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.8
    String path =
        location.getPath() != null ? location.getPath().getPathString().replace('/', '$') : "";
    String compiledFunctionClassName =
        CompiledFunction.class.getCanonicalName() + path + "$" + getName();
    compilerDebug("Compiling " + getLocationPathAndLine() + " " + getName());
    try {
      int publicStatic = Visibility.PUBLIC.getMask() | Ownership.STATIC.getMask();
      TypeDescription.Latent latentCompiledFunctionClass =
          new TypeDescription.Latent(
              compiledFunctionClassName,
              publicStatic | TypeManifestation.FINAL.getMask(),
              new TypeDescription.ForLoadedType(CompiledFunction.class),
              Collections.<TypeDescription>emptyList());
      MethodDescription getAstNode =
          new MethodDescription.Latent(
              latentCompiledFunctionClass,
              new MethodDescription.Token(
                  "getAstNode",
                  publicStatic | MethodManifestation.FINAL.getMask(),
                  new TypeDescription.ForLoadedType(ASTNode.class),
                  Arrays.asList(new TypeDescription.ForLoadedType(int.class))));
      MethodDescription getLocation =
          new MethodDescription.Latent(
              latentCompiledFunctionClass,
              new MethodDescription.Token(
                  "getLocation",
                  publicStatic | MethodManifestation.FINAL.getMask(),
                  new TypeDescription.ForLoadedType(Location.class),
                  Arrays.asList(new TypeDescription.ForLoadedType(int.class))));

      DebugInfo debugInfo = new DebugInfo(getAstNode, getLocation);
      FunctionSignature sig = signature.getSignature();
      VariableScope scope = VariableScope.function(sig.getNames());
      Implementation compiledImplementation = compileBody(scope, debugInfo);

      List<Class<?>> parameterTypes = sig.getShape().toClasses();
      parameterTypes.add(Environment.class);
      Unloaded<CompiledFunction> unloadedImplementation =
          new ByteBuddy()
              .withClassVisitor(new StackMapFrameClassVisitor(debugCompilerPrintByteCode))
              .subclass(CompiledFunction.class)
              .name(compiledFunctionClassName)
              .defineMethod(
                  "call",
                  Object.class,
                  parameterTypes,
                  Visibility.PUBLIC,
                  Ownership.STATIC,
                  MethodManifestation.FINAL)
              .intercept(compiledImplementation)
              .defineMethod(getAstNode)
              // TODO(bazel-team) unify the two delegate fields into one, probably needs a custom
              // ImplementationDelegate that adds it only once? or just create the static field
              // itself with the correct value and create getAstNode & getLocation with a custom
              // implementation using it
              .intercept(
                  MethodDelegation.to(debugInfo, DebugInfo.class, "getAstNodeDelegate")
                      .filter(ElementMatchers.named("getAstNode")))
              .defineMethod(getLocation)
              .intercept(
                  MethodDelegation.to(debugInfo, DebugInfo.class, "getLocationDelegate")
                      .filter(ElementMatchers.named("getLocation")))
              .make();
      saveByteCode(unloadedImplementation);
      Class<? extends CompiledFunction> functionClass =
          unloadedImplementation
              .load(getClass().getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
              .getLoaded();

      return Optional.of(
          ReflectionUtils.getMethod(
                  functionClass,
                  "call",
                  parameterTypes.toArray(new Class<?>[parameterTypes.size()]))
              .getLoadedMethod());
    } catch (EvalException e) {
      // don't capture EvalExceptions
      throw e;
    } catch (Throwable e) {
      compilerDebug("Error while compiling", e);
      // TODO(bazel-team) don't capture all throwables? couldn't compile this, log somewhere?
    }
    return Optional.absent();
  }

  /**
   * Saves byte code to a temporary directory prefixed with "skylarkbytecode" in the system
   * default temporary directory.
   */
  private void saveByteCode(Unloaded<CompiledFunction> unloadedImplementation) {
    if (debugCompiler) {
      try {
        if (debugFolder == null) {
          debugFolder = Files.createTempDirectory("skylarkbytecode").toFile();
        }
        unloadedImplementation.saveIn(debugFolder);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /**
   * Builds a byte code implementation of the AST.
   */
  private Implementation compileBody(VariableScope scope, DebugInfo debugInfo)
      throws EvalException {
    List<ByteCodeAppender> code = new ArrayList<>(statements.size());
    code.add(null); // reserve space for later addition of the local variable initializer

    for (Statement statement : statements) {
      code.add(statement.compile(scope, LoopLabels.ABSENT, debugInfo));
    }
    // add a return None if there are no statements or the last one to ensure the method always
    // returns something. This implements the interpreters behavior.
    if (statements.isEmpty()
        || !(statements.get(statements.size() - 1) instanceof ReturnStatement)) {
      code.add(new ByteCodeAppender.Simple(Runtime.GET_NONE, MethodReturn.REFERENCE));
    }
    // we now know which variables we used in the method, so assign them "undefined" (i.e. null)
    // at the beginning of the method
    code.set(0, scope.createLocalVariablesUndefined());
    // TODO(bazel-team) wrap ByteCodeAppender in our own type including a reference to the ASTNode
    // it came from and verify the stack and local variables ourselves, because ASM does not help
    // with debugging much when its stack map frame calculation fails because of invalid byte code
    return new Implementation.Simple(ByteCodeUtils.compoundAppender(code));
  }

  /**
   * Returns the location (filename:line) of the BaseFunction's definition.
   *
   * <p>If such a location is not defined, this method returns an empty string.
   */
  private String getLocationPathAndLine() {
    if (location == null) {
      return "";
    }

    StringBuilder builder = new StringBuilder();
    PathFragment path = location.getPath();
    if (path != null) {
      builder.append(path.getPathString());
    }

    LineAndColumn position = location.getStartLineAndColumn();
    if (position != null) {
      builder.append(":").append(position.getLine());
    }
    return builder.toString();
  }

  private void compilerDebug(String message) {
    System.err.println(message);
  }

  private void compilerDebug(String message, Throwable e) {
    compilerDebug(message);
    e.printStackTrace();
  }

  /**
   * A simple super class for all compiled function's classes.
   */
  protected abstract static class CompiledFunction {}

  /**
   * A {@link Textifier} for printing the generated byte code that keeps the ASM-internal label
   * names in place for easier debugging with IDE debuggers.
   */
  private static class DebugTextifier extends Textifier {
    DebugTextifier() {
      super(Opcodes.ASM5);
    }

    @Override
    protected void appendLabel(Label l) {
      buf.append(l.toString());
    }

    @Override
    protected Textifier createTextifier() {
      return new DebugTextifier();
    }
  }

  /**
   * Passes the {@link ClassWriter#COMPUTE_FRAMES} hint to ASM and optionally prints generated
   * byte code to System.err.
   */
  private static class StackMapFrameClassVisitor implements ClassVisitorWrapper {

    private final boolean debug;

    private StackMapFrameClassVisitor(boolean debug) {
      this.debug = debug;
    }

    @Override
    public int mergeWriter(int hint) {
      return hint | ClassWriter.COMPUTE_FRAMES;
    }

    @Override
    public int mergeReader(int hint) {
      return hint;
    }

    @Override
    public ClassVisitor wrap(ClassVisitor classVisitor) {
      if (debug) {
        return new TraceClassVisitor(
            classVisitor, new DebugTextifier(), new PrintWriter(System.err, true));
      } else {
        return classVisitor;
      }
    }
  }
}
