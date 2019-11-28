// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.util.Pair;
import javax.annotation.Nullable;

/**
 * Base class for declared providers {@see Provider} defined in native code.
 *
 * <p>Every non-abstract derived class of {@link NativeProvider} corresponds to a single declared
 * provider. This is enforced by final {@link #equals(Object)} and {@link #hashCode()}.
 *
 * <p>Typical implementation of a non-constructable from Skylark declared provider is as follows:
 *
 * <pre>
 *     public static final Provider PROVIDER =
 *       new NativeProvider("link_params") { };
 * </pre>
 *
 * To allow construction from Skylark and custom construction logic, override the {@link #call}
 * method.
 *
 * @deprecated use {@link BuiltinProvider} instead.
 */
@Immutable
@Deprecated
public abstract class NativeProvider<V extends InfoInterface> extends BaseFunction
    implements Provider {
  private final String name;
  private final NativeKey key;
  private final String errorMessageFormatForUnknownField;

  private final Class<V> valueClass;

  public Class<V> getValueClass() {
    return valueClass;
  }

  /**
   * Implement this to mark that a native provider should be exported with certain name to Skylark.
   * Broken: only works for rules, not for aspects. DO NOT USE FOR NEW CODE!
   *
   * <p>Use native declared providers mechanism exclusively to expose providers to both native and
   * Skylark code.
   */
  @Deprecated
  public interface WithLegacySkylarkName {
    String getSkylarkName();
  }

  protected NativeProvider(Class<V> valueClass, String name) {
    super(FunctionSignature.KWARGS, /*defaultValues=*/ null);
    this.name = name;
    this.key = new NativeKey(name, getClass());
    this.valueClass = valueClass;
    this.errorMessageFormatForUnknownField =
        String.format("'%s' object has no attribute '%%s'", name);
  }

  public final SkylarkProviderIdentifier id() {
    return SkylarkProviderIdentifier.forKey(getKey());
  }

  /**
   * equals() implements singleton class semantics.
   *
   * <p>Every non-abstract derived class of {@link NativeProvider} corresponds to a single declared
   * provider.
   */
  @Override
  public final boolean equals(@Nullable Object other) {
    return other != null && this.getClass().equals(other.getClass());
  }

  /**
   * hashCode() implements singleton class semantics.
   *
   * <p>Every non-abstract derived class of {@link NativeProvider} corresponds to a single declared
   * provider.
   */
  @Override
  public final int hashCode() {
    return getClass().hashCode();
  }

  @Override
  public String getPrintableName() {
    return name; // for provider-related errors
  }

  @Override
  public String getName() {
    return name; // for stack traces
  }

  @Override
  public String getErrorMessageFormatForUnknownField() {
    return errorMessageFormatForUnknownField;
  }

  @Override
  public Location getLocation() {
    return Location.BUILTIN;
  }

  @Override
  public boolean isExported() {
    return true;
  }

  @Override
  public NativeKey getKey() {
    return key;
  }

  /**
   * Override this method to provide logic that is used to instantiate a declared provider from
   * Skylark.
   *
   * <p>This is a method that is called when a constructor {@code c} is invoked as<br>
   * {@code c(arg1 = val1, arg2 = val2, ...)}.
   *
   * @param args an array of argument values sorted as per the signature ({@see BaseFunction#call})
   */
  @Override
  protected Object call(Object[] args, @Nullable FuncallExpression ast, StarlarkThread thread)
      throws EvalException, InterruptedException {
    Location loc = ast != null ? ast.getLocation() : Location.BUILTIN;
    throw new EvalException(
        loc, String.format("'%s' cannot be constructed from Starlark", getPrintableName()));
  }

  public static Pair<String, String> getSerializedRepresentationForNativeKey(NativeKey key) {
    return Pair.of(key.name, key.aClass.getName());
  }

  @SuppressWarnings("unchecked")
  public static NativeKey getNativeKeyFromSerializedRepresentation(Pair<String, String> serialized)
      throws ClassNotFoundException {
    Class<? extends Provider> aClass = Class.forName(serialized.second).asSubclass(Provider.class);
    return new NativeKey(serialized.first, aClass);
  }

  /**
   * A serializable representation of {@link NativeProvider}.
   *
   * <p>Just a wrapper around its class.
   */
  @AutoCodec
  @Immutable
  // TODO(cparsons): Move this class, as NativeProvider is deprecated.
  public static final class NativeKey extends Key {
    private final String name;
    private final Class<? extends Provider> aClass;

    @VisibleForSerialization
    NativeKey(String name, Class<? extends Provider> aClass) {
      this.name = name;
      this.aClass = aClass;
    }

    @Override
    public int hashCode() {
      return aClass.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof NativeKey && aClass.equals(((NativeKey) obj).aClass);
    }

    @Override
    public String toString() {
      return name;
    }
  }
}
