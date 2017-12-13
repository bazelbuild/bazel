// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Global constants and support for static registration of builtin symbols.
 */
// TODO(bazel-team): Rename to SkylarkRuntime to avoid conflict with java.lang.Runtime.
public final class Runtime {

  private Runtime() {}

  @SkylarkSignature(name = "True", returnType = Boolean.class,
      doc = "Literal for the boolean true.")
  private static final Boolean TRUE = true;

  @SkylarkSignature(name = "False", returnType = Boolean.class,
      doc = "Literal for the boolean false.")
  private static final Boolean FALSE = false;

  /** There should be only one instance of this type to allow "== None" tests. */
  @SkylarkModule(
    name = "NoneType",
    documented = false,
    doc = "Unit type, containing the unique value None."
  )
  @Immutable
  public static final class NoneType implements SkylarkValue {
    private NoneType() {}

    @Override
    public String toString() {
      return "None";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("None");
    }
  }

  /** Marker for unbound variables in cases where neither Java null nor Skylark None is suitable. */
  @Immutable
  public static final class UnboundMarker implements SkylarkValue {
    private UnboundMarker() {}

    @Override
    public String toString() {
      return "<unbound>";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("<unbound>");
    }
  }

  @SkylarkSignature(name = "<unbound>", returnType = UnboundMarker.class, documented = false,
      doc = "Marker for unbound values in cases where neither Skylark None nor Java null can do.")
  public static final UnboundMarker UNBOUND = new UnboundMarker();

  @SkylarkSignature(name = "None", returnType = NoneType.class,
      doc = "Literal for the None value.")
  public static final NoneType NONE = new NoneType();

  @SkylarkSignature(name = "PACKAGE_NAME", returnType = String.class,
      doc = "<b>Deprecated. Use <a href=\"native.html#package_name\">package_name()</a> "
          + "instead.</b> The name of the package being evaluated. "
          + "For example, in the BUILD file <code>some/package/BUILD</code>, its value "
          + "will be <code>some/package</code>. "
          + "If the BUILD file calls a function defined in a .bzl file, PACKAGE_NAME will "
          + "match the caller BUILD file package. "
          + "In .bzl files, do not access PACKAGE_NAME at the file-level (outside of functions), "
          + "either directly or by calling a function at the file-level that accesses "
          + "PACKAGE_NAME (PACKAGE_NAME is only defined during BUILD file evaluation)."
          + "Here is an example of a .bzl file:<br>"
          + "<pre class=language-python>"
          + "# a = PACKAGE_NAME  # not allowed outside functions\n"
          + "def extension():\n"
          + "  return PACKAGE_NAME</pre>"
          + "In this case, <code>extension()</code> can be called from a BUILD file (even "
          + "indirectly), but not in a file-level expression in the .bzl file. "
          + "When implementing a rule, use <a href=\"ctx.html#label\">ctx.label</a> to know where "
          + "the rule comes from. ")
  public static final String PKG_NAME = "PACKAGE_NAME";

  @SkylarkSignature(
    name = "REPOSITORY_NAME",
    returnType = String.class,
    doc =
        "<b>Deprecated. Use <a href=\"native.html#repository_name\">repository_name()</a> "
            + "instead.</b> The name of the repository the rule or build extension is called from. "
            + "For example, in packages that are called into existence by the WORKSPACE stanza "
            + "<code>local_repository(name='local', path=...)</code> it will be set to "
            + "<code>@local</code>. In packages in the main repository, it will be set to "
            + "<code>@</code>. It can only be accessed in functions (transitively) called from "
            + "BUILD files, i.e. it follows the same restrictions as "
            + "<a href=\"#PACKAGE_NAME\">PACKAGE_NAME</a>"
  )
  public static final String REPOSITORY_NAME = "REPOSITORY_NAME";

  /**
   * Set up a given environment for supported class methods.
   */
  static Environment setupConstants(Environment env) {
    // In Python 2.x, True and False are global values and can be redefined by the user.
    // In Python 3.x, they are keywords. We implement them as values, for the sake of
    // simplicity. We define them as Boolean objects.
    return env.setup("False", FALSE).setup("True", TRUE).setup("None", NONE);
  }


  /**
   * Returns the canonical class representing the namespace associated with the given class, i.e.,
   * the class under which builtins should be registered.
   */
  public static Class<?> getSkylarkNamespace(Class<?> clazz) {
    return String.class.isAssignableFrom(clazz)
        ? MethodLibrary.StringModule.class
        : EvalUtils.getSkylarkType(clazz);
  }

  /**
   * A registry of builtins, including both global builtins and builtins that are under some
   * namespace.
   *
   * <p>This object is unsynchronized, but concurrent reads are fine.
   */
  public static class BuiltinRegistry {

    /**
     * All registered builtins, keyed and sorted by an identifying (but otherwise unimportant)
     * string.
     *
     * <p>The string is typically formed from the builtin's simple name and the Java class in which
     * it is defined. The Java class need not correspond to a namespace. (This map includes global
     * builtins that have no namespace.)
     */
    private final Map<String, Object> allBuiltins = new TreeMap<>();

    /** All non-global builtin functions, keyed by their namespace class and their name. */
    private final Map<Class<?>, Map<String, BaseFunction>> functions = new HashMap<>();

    /** Registers a builtin with the given simple name, that was defined in the given Java class. */
    public void registerBuiltin(Class<?> definingClass, String name, Object builtin) {
      String key = String.format("%s.%s", definingClass.getName(), name);
      Preconditions.checkArgument(
          !allBuiltins.containsKey(key),
          "Builtin '%s' registered multiple times",
          key);
      allBuiltins.put(key, builtin);
    }

    /**
     * Registers a function underneath a namespace.
     *
     * <p>This is independent of {@link #registerBuiltin}.
     */
    public void registerFunction(Class<?> namespace, BaseFunction function) {
      Preconditions.checkNotNull(namespace);
      Preconditions.checkNotNull(function.getObjectType());
      Class<?> skylarkNamespace = getSkylarkNamespace(namespace);
      Preconditions.checkArgument(skylarkNamespace.equals(namespace));
      Class<?> objType = getSkylarkNamespace(function.getObjectType());
      Preconditions.checkArgument(objType.equals(skylarkNamespace));

      functions.computeIfAbsent(namespace, k -> new HashMap<>());
      functions.get(namespace).put(function.getName(), function);
    }

    /** Returns a set of all registered builtins, in a deterministic order. */
    public ImmutableSet<Object> getBuiltins() {
      return ImmutableSet.copyOf(allBuiltins.values());
    }

    @Nullable
    private Map<String, BaseFunction> getFunctionsInNamespace(Class<?> namespace) {
      return functions.get(getSkylarkNamespace(namespace));
    }

    /**
     * Given a namespace, returns the function with the given name.
     *
     * <p>If the namespace does not exist or has no function with that name, returns null.
     */
    public BaseFunction getFunction(Class<?> namespace, String name) {
      Map<String, BaseFunction> namespaceFunctions = getFunctionsInNamespace(namespace);
      return namespaceFunctions != null ? namespaceFunctions.get(name) : null;
    }

    /**
     * Given a namespace, returns all function names.
     *
     * <p>If the namespace does not exist, returns an empty set.
     */
    public ImmutableSet<String> getFunctionNames(Class<?> namespace) {
      Map<String, BaseFunction> namespaceFunctions = getFunctionsInNamespace(namespace);
      if (namespaceFunctions == null) {
        return ImmutableSet.of();
      }
      return ImmutableSet.copyOf(namespaceFunctions.keySet());
    }
  }

  /**
   * All Skylark builtins.
   *
   * <p>Note that just because a symbol is registered here does not necessarily mean that it is
   * accessible in a particular {@link Environment}. This registry should include any builtin that
   * is available in any environment.
   *
   * <p>Thread safety: This object is unsynchronized. The register functions are typically called
   * from within static initializer blocks, which should be fine.
   */
  private static final BuiltinRegistry builtins = new BuiltinRegistry();

  /** Retrieve the static instance containing information on all known Skylark builtins. */
  public static BuiltinRegistry getBuiltinRegistry() {
    return builtins;
  }

  /**
   * Registers global fields with SkylarkSignature into the specified Environment.
   * @param env the Environment into which to register fields.
   * @param moduleClass the Class object containing globals.
   */
  public static void setupModuleGlobals(Environment env, Class<?> moduleClass) {
    try {
      if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
        env.setup(
            moduleClass.getAnnotation(SkylarkModule.class).name(),
            moduleClass.getConstructor().newInstance());
      }
      for (Field field : moduleClass.getDeclaredFields()) {
        if (field.isAnnotationPresent(SkylarkSignature.class)) {
          // Fields in Skylark modules are sometimes private.
          // Nevertheless they have to be annotated with SkylarkSignature.
          field.setAccessible(true);
          SkylarkSignature annotation = field.getAnnotation(SkylarkSignature.class);
          Object value = field.get(null);
          // Ignore function factories and non-global functions
          if (!(value instanceof BuiltinFunction.Factory
              || (value instanceof BaseFunction
                  && !annotation.objectType().equals(Object.class)))) {
            env.setup(annotation.name(), value);
          }
        }
      }
    } catch (ReflectiveOperationException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Registers global fields with SkylarkSignature into the specified Environment. Alias for
   * {@link #setupModuleGlobals}.
   *
   * @deprecated Use {@link #setupModuleGlobals} instead.
   */
  @Deprecated
  // TODO(bazel-team): Remove after all callers updated.
  public static void registerModuleGlobals(Environment env, Class<?> moduleClass) {
    setupModuleGlobals(env, moduleClass);
  }

  static void setupMethodEnvironment(
      Environment env, Iterable<BaseFunction> functions) {
    for (BaseFunction function : functions) {
      env.setup(function.getName(), function);
    }
  }
}
