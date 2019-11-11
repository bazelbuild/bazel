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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;

/** Global constants and support for static registration of builtin symbols. */
// TODO(adonovan): migrate None and Unbound to Starlark.java, and NoneType and
// UnboundMarker to top level, then delete this class.
public final class Runtime {

  private Runtime() {}

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
    public boolean truth() {
      return false;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("None");
    }
  }

  /* The Starlark None value. */
  public static final NoneType NONE = new NoneType();

  /** Marker for unbound variables in cases where neither Java null nor Skylark None is suitable. */
  @Immutable
  private static final class UnboundMarker implements SkylarkValue {
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

  public static final UnboundMarker UNBOUND = new UnboundMarker();

  /**
   * Adds global (top-level) symbols, provided by the given object, to the given bindings builder.
   *
   * <p>Global symbols may be provided by the given object in the following ways:
   *
   * <ul>
   *   <li>If its class is annotated with {@link SkylarkModule}, an instance of that object is a
   *       global object with the module's name.
   *   <li>If its class is annotated with {@link SkylarkGlobalLibrary}, then all of its methods
   *       which are annotated with {@link SkylarkCallable} are global callables.
   * </ul>
   *
   * <p>On collisions, this method throws an {@link AssertionError}. Collisions may occur if
   * multiple global libraries have functions of the same name, two modules of the same name are
   * given, or if two subclasses of the same module are given.
   *
   * @param builder the builder for the "bindings" map, which maps from symbol names to objects, and
   *     which will be built into a global frame
   * @param moduleInstance the object containing globals
   * @throws AssertionError if there are name collisions
   * @throws IllegalArgumentException if {@code moduleInstance} is not annotated with {@link
   *     SkylarkGlobalLibrary} nor {@link SkylarkModule}
   */
  public static void setupSkylarkLibrary(
      ImmutableMap.Builder<String, Object> builder, Object moduleInstance) {
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
      for (String methodName : CallUtils.getMethodNames(moduleClass)) {
        builder.put(methodName, CallUtils.getBuiltinCallable(moduleInstance, methodName));
      }
    }
  }
}
