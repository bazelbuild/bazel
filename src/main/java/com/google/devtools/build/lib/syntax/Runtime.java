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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
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

  @SkylarkSignature(
      name = "<unbound>",
      returnType = UnboundMarker.class,
      documented = false,
      doc = "Marker for unbound values in cases where neither Starlark None nor Java null can do.")
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
              + "instead.</b> The name of the repository the rule or build extension is called "
              + "from. "
              + "For example, in packages that are called into existence by the WORKSPACE stanza "
              + "<code>local_repository(name='local', path=...)</code> it will be set to "
              + "<code>@local</code>. In packages in the main repository, it will be set to "
              + "<code>@</code>. It can only be accessed in functions (transitively) called from "
              + "BUILD files, i.e. it follows the same restrictions as "
              + "<a href=\"#PACKAGE_NAME\">PACKAGE_NAME</a>.")
  public static final String REPOSITORY_NAME = "REPOSITORY_NAME";

  /** Adds bindings for False/True/None constants to the given map builder. */
  public static void addConstantsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    // In Python 2.x, True and False are global values and can be redefined by the user.
    // In Python 3.x, they are keywords. We implement them as values. Currently they can't be
    // redefined because builtins can't be overridden. In the future we should permit shadowing of
    // most builtins but still prevent shadowing of these constants.
    builder
        .put("False", FALSE)
        .put("True", TRUE)
        .put("None", NONE);
  }


  /**
   * Returns the canonical class representing the namespace associated with the given class, i.e.,
   * the class under which builtins should be registered.
   */
  public static Class<?> getSkylarkNamespace(Class<?> clazz) {
    return String.class.isAssignableFrom(clazz)
        ? StringModule.class
        : EvalUtils.getSkylarkType(clazz);
  }

  /**
   * A registry of builtins, including both global builtins and builtins that are under some
   * namespace.
   *
   * <p>Concurrency model: This object is thread-safe. Read accesses are always allowed, while write
   * accesses are only allowed before this object has been frozen ({@link #freeze}). Prior to
   * freezing, all operations are synchronized, while after freezing they are lockless.
   */
  public static class BuiltinRegistry {

    /**
     * Whether the registry's construction has completed.
     *
     * <p>Mutating methods may only be called while this is still false. Accessor methods may be
     * called at any time.
     *
     * <p>We use {@code volatile} rather than {@link AtomicBoolean} because the bit flip only
     * happens once, and doesn't require correlated reads and writes.
     */
    private volatile boolean frozen = false;

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

    /**
     * Marks the registry as initialized, if it wasn't already.
     *
     * <p>It is guaranteed that after this method returns, all accessor methods are safe without
     * synchronization; i.e. no mutation operation can touch the data structures.
     */
    public void freeze() {
      // Similar to double-checked locking, but no need to check again on the inside since we don't
      // care if two threads set the bit at once. The synchronized block is only to provide
      // exclusion with mutations.
      if (!this.frozen) {
        synchronized (this) {
          this.frozen = true;
        }
      }
    }

    /** Registers a builtin with the given simple name, that was defined in the given Java class. */
    public synchronized void registerBuiltin(Class<?> definingClass, String name, Object builtin) {
      String key = String.format("%s.%s", definingClass.getName(), name);
      Preconditions.checkArgument(
          !allBuiltins.containsKey(key),
          "Builtin '%s' registered multiple times",
          key);

      Preconditions.checkState(
          !frozen,
          "Attempted to register builtin '%s' after registry has already been frozen",
          key);

      allBuiltins.put(key, builtin);
    }

    /**
     * Registers a function underneath a namespace.
     *
     * <p>This is independent of {@link #registerBuiltin}.
     */
    public synchronized void registerFunction(Class<?> namespace, BaseFunction function) {
      Preconditions.checkNotNull(namespace);
      Preconditions.checkNotNull(function.getObjectType());
      Class<?> skylarkNamespace = getSkylarkNamespace(namespace);
      Preconditions.checkArgument(skylarkNamespace.equals(namespace));
      Class<?> objType = getSkylarkNamespace(function.getObjectType());
      Preconditions.checkArgument(objType.equals(skylarkNamespace));

      Preconditions.checkState(
          !frozen,
          "Attempted to register function '%s' in namespace '%s' after registry has already been "
              + "frozen",
          function,
          namespace);

      functions.computeIfAbsent(namespace, k -> new HashMap<>());
      functions.get(namespace).put(function.getName(), function);
    }

    /** Returns a list of all registered builtins, in a deterministic order. */
    public ImmutableList<Object> getBuiltins() {
      if (frozen) {
        return ImmutableList.copyOf(allBuiltins.values());
      } else {
        synchronized (this) {
          return ImmutableList.copyOf(allBuiltins.values());
        }
      }
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
      if (frozen) {
        Map<String, BaseFunction> namespaceFunctions = getFunctionsInNamespace(namespace);
        return namespaceFunctions != null ? namespaceFunctions.get(name) : null;
      } else {
        synchronized (this) {
          Map<String, BaseFunction> namespaceFunctions = getFunctionsInNamespace(namespace);
          return namespaceFunctions != null ? namespaceFunctions.get(name) : null;
        }
      }
    }

    /**
     * Given a namespace, returns all function names.
     *
     * <p>If the namespace does not exist, returns an empty set.
     */
    public ImmutableSet<String> getFunctionNames(Class<?> namespace) {
      if (frozen) {
        Map<String, BaseFunction> namespaceFunctions = getFunctionsInNamespace(namespace);
        if (namespaceFunctions == null) {
          return ImmutableSet.of();
        }
        return ImmutableSet.copyOf(namespaceFunctions.keySet());
      } else {
        synchronized (this) {
          Map<String, BaseFunction> namespaceFunctions = getFunctionsInNamespace(namespace);
          if (namespaceFunctions == null) {
            return ImmutableSet.of();
          }
          return ImmutableSet.copyOf(namespaceFunctions.keySet());
        }
      }
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

  /**
   * Retrieve the static instance containing information on all known Skylark builtins.
   *
   * @deprecated do not use a static singleton registry -- instead set up the Skylark environment
   *     with 'global' objects
   */
  @Deprecated
  public static BuiltinRegistry getBuiltinRegistry() {
    return builtins;
  }

  /**
   * Convenience overload of {@link #setupModuleGlobals(ImmutableMap.Builder, Class)} to add
   * bindings directly to an {@link Environment}.
   *
   * @param env the Environment into which to register fields
   * @param moduleClass the Class object containing globals
   * @deprecated use {@link #setupSkylarkLibrary} instead (and {@link SkylarkCallable} instead of
   *     {@link SkylarkSignature})
   */
  @Deprecated
  public static void setupModuleGlobals(Environment env, Class<?> moduleClass) {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    setupModuleGlobals(envBuilder, moduleClass);
    for (Map.Entry<String, Object> envEntry : envBuilder.build().entrySet()) {
      env.setup(envEntry.getKey(), envEntry.getValue());
    }
  }

  /**
   * Adds global (top-level) symbols, provided by the given class object, to the given bindings
   * builder.
   *
   * <p>Global symbols may be provided by the given class in the following ways:
   * <ul>
   *   <li>If the class is annotated with {@link SkylarkModule}, an instance of that object is
   *       a global object with the module's name.</li>
   *   <li>If the class has fields annotated with {@link SkylarkSignature}, each of these
   *       fields is a global object with the signature's name.</li>
   *   <li>If the class is annotated with {@link SkylarkGlobalLibrary}, then all of its methods
   *       which are annotated with
   *       {@link com.google.devtools.build.lib.skylarkinterface.SkylarkCallable} are global
   *       callables.</li>
   * </ul>
   *
   * <p>On collisions, this method throws an {@link AssertionError}. Collisions may occur if
   * multiple global libraries have functions of the same name, two modules of the same name
   * are given, or if two subclasses of the same module are given.
   *
   * @param builder the builder for the "bindings" map, which maps from symbol names to objects,
   *     and which will be built into a global frame
   * @param moduleClass the Class object containing globals
   * @deprecated use {@link #setupSkylarkLibrary} instead (and {@link SkylarkCallable} instead of
   *     {@link SkylarkSignature})
   */
  @Deprecated
  public static void setupModuleGlobals(ImmutableMap.Builder<String, Object> builder,
      Class<?> moduleClass) {
    try {
      if (SkylarkInterfaceUtils.getSkylarkModule(moduleClass) != null
          || SkylarkInterfaceUtils.hasSkylarkGlobalLibrary(moduleClass)) {
        setupSkylarkLibrary(builder, moduleClass.getConstructor().newInstance());
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
            builder.put(annotation.name(), value);
          }
        }
      }
    } catch (ReflectiveOperationException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Adds global (top-level) symbols, provided by the given object, to the given bindings
   * builder.
   *
   * <p>Global symbols may be provided by the given object in the following ways:
   * <ul>
   *   <li>If its class is annotated with {@link SkylarkModule}, an instance of that object is
   *       a global object with the module's name.</li>
   *   <li>If its class is annotated with {@link SkylarkGlobalLibrary}, then all of its methods
   *       which are annotated with
   *       {@link com.google.devtools.build.lib.skylarkinterface.SkylarkCallable} are global
   *       callables.</li>
   * </ul>
   *
   * <p>On collisions, this method throws an {@link AssertionError}. Collisions may occur if
   * multiple global libraries have functions of the same name, two modules of the same name
   * are given, or if two subclasses of the same module are given.
   *
   * @param builder the builder for the "bindings" map, which maps from symbol names to objects,
   *     and which will be built into a global frame
   * @param moduleInstance the object containing globals
   * @throws AssertionError if there are name collisions
   * @throws IllegalArgumentException if {@code moduleInstance} is not annotated with
   *     {@link SkylarkGlobalLibrary} nor {@link SkylarkModule}
   */
  public static void setupSkylarkLibrary(ImmutableMap.Builder<String, Object> builder,
      Object moduleInstance) {
    Class<?> moduleClass = moduleInstance.getClass();
    SkylarkModule skylarkModule = SkylarkInterfaceUtils.getSkylarkModule(moduleClass);
    boolean hasSkylarkGlobalLibrary = SkylarkInterfaceUtils.hasSkylarkGlobalLibrary(moduleClass);

    Preconditions.checkArgument(hasSkylarkGlobalLibrary || skylarkModule != null,
        "%s must be annotated with @SkylarkGlobalLibrary or @SkylarkModule",
        moduleClass);

    if (skylarkModule != null) {
      builder.put(skylarkModule.name(), moduleInstance);
    }
    if (hasSkylarkGlobalLibrary) {
      for (String methodName : FuncallExpression.getMethodNames(moduleClass)) {
        builder.put(methodName, FuncallExpression.getBuiltinCallable(moduleInstance, methodName));
      }
    }
  }
}
