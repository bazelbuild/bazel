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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.util.Pair;
import javax.annotation.Nullable;

/**
 * Base class for declared providers {@see Provider} defined in native code.
 *
 * <p>Every non-abstract derived class of {@link NativeProvider} corresponds to a single declared
 * provider. This is enforced by final {@link #equals(Object)} and {@link #hashCode()}.
 *
 * <p>Typical implementation of a non-constructable from Starlark declared provider is as follows:
 *
 * <pre>
 *     public static final Provider PROVIDER = new NativeProvider("link_params") {};
 * </pre>
 *
 * @deprecated use {@link BuiltinProvider} instead.
 */
@Immutable
@Deprecated
public abstract class NativeProvider<V extends Info> implements StarlarkValue, Provider {
  private final String name;
  private final NativeKey key;
  private final String errorMessageFormatForUnknownField;

  private final Class<V> valueClass;

  public Class<V> getValueClass() {
    return valueClass;
  }

  /**
   * Implement this to mark that a native provider should be exported with certain name to Starlark.
   * Broken: only works for rules, not for aspects. DO NOT USE FOR NEW CODE!
   *
   * <p>Use native declared providers mechanism exclusively to expose providers to both native and
   * Starlark code.
   */
  @Deprecated
  public interface WithLegacySkylarkName {
    String getSkylarkName();
  }

  protected NativeProvider(Class<V> valueClass, String name) {
    this.name = name;
    this.key = new NativeKey(name, getClass());
    this.valueClass = valueClass;
    this.errorMessageFormatForUnknownField =
        String.format("'%s' value has no field or method '%%s'", name);
  }

  public final StarlarkProviderIdentifier id() {
    return StarlarkProviderIdentifier.forKey(getKey());
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
