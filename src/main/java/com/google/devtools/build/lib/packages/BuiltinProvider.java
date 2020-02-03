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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeProvider.NativeKey;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import javax.annotation.Nullable;

/**
 * Base class for declared providers {@see Provider} defined in native code.
 *
 * <p>Every subclass of {@link BuiltinProvider} corresponds to a single declared provider. This is
 * enforced by final {@link #equals(Object)} and {@link #hashCode()}.
 *
 * <p>Implementations of native declared providers should subclass this class, and define a method
 * in the subclass definition to create instances of its corresponding Info object. The method
 * should be annotated with {@link SkylarkCallable} with {@link SkylarkCallable#selfCall} set to
 * true, and with {@link SkylarkConstructor} for the info type it constructs.
 */
@Immutable
public abstract class BuiltinProvider<T extends Info> implements Provider {
  private final NativeKey key;
  private final String name;
  private final Class<T> valueClass;

  public Class<T> getValueClass() {
    return valueClass;
  }

  public BuiltinProvider(String name, Class<T> valueClass) {
    @SuppressWarnings("unchecked")
    Class<? extends BuiltinProvider<?>> clazz = (Class<? extends BuiltinProvider<?>>) getClass();
    key = new NativeKey(name, clazz);
    this.name = name;
    this.valueClass = valueClass;
  }

  /**
   * equals() implements singleton class semantics.
   */
  @Override
  public final boolean equals(@Nullable Object other) {
    return other != null && this.getClass().equals(other.getClass());
  }

  /**
   * hashCode() implements singleton class semantics.
   */
  @Override
  public final int hashCode() {
    return getClass().hashCode();
  }

  @Override
  public boolean isExported() {
    return true;
  }

  @Override
  public NativeKey getKey() {
    return key;
  }

  @Override
  public Location getLocation() {
    return Location.BUILTIN;
  }

  @Override
  public String getPrintableName() {
    return name;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<function " + getPrintableName() + ">");
  }

  /**
   * Convenience method for subclasses of this class to throw a consistent error when a provider is
   * unable to be constructed from skylark.
   */
  protected final T throwUnsupportedConstructorException() throws EvalException {
    throw Starlark.errorf("'%s' cannot be constructed from Starlark", getPrintableName());
  }

  /**
   * Returns the identifier of this provider.
   */
  public SkylarkProviderIdentifier id() {
    return SkylarkProviderIdentifier.forKey(getKey());
  }
}
