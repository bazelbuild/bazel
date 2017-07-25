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
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.Pair;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Base class for declared providers {@see ClassObjectConstructor} defined in native code.
 *
 * <p>Every non-abstract derived class of {@link NativeClassObjectConstructor} corresponds to a
 * single declared provider. This is enforced by final {@link #equals(Object)} and {@link
 * #hashCode()}.
 *
 * <p>Typical implementation of a non-constructable from Skylark declared provider is as follows:
 *
 * <pre>
 *     public static final ClassObjectConstructor CC_LINK_PARAMS =
 *       new NativeClassObjectConstructor("link_params") { };
 * </pre>
 *
 * To allow construction from Skylark and custom construction logic, override {@link
 * #createInstanceFromSkylark(Object[], Location)} (see {@link #STRUCT} for an example.
 */
@Immutable
public abstract class NativeClassObjectConstructor<VALUE extends SkylarkClassObject>
    extends ClassObjectConstructor {
  private final NativeKey key;
  private final String errorMessageForInstances;

  /**
   * "struct" function.
   */
  public static final StructConstructor STRUCT = new StructConstructor();

  private final Class<VALUE> valueClass;

  public Class<VALUE> getValueClass() {
    return valueClass;
  }

  /**
   * Implement this to mark that a native provider should be exported with
   * certain name to Skylark.
   * Broken: only works for rules, not for aspects.
   * DO NOT USE FOR NEW CODE!
   *
   * Use native declared providers
   * ({@link NativeClassObjectConstructor}) to
   * expose providers to both native and Skylark code.
   */
  @Deprecated
  public static interface WithLegacySkylarkName {
    String getSkylarkName();
  }

  /**
   * A constructor for default {@code struct}s.
   *
   * <p>Singleton, instance is {@link #STRUCT}.
   */
  public static final class StructConstructor
      extends NativeClassObjectConstructor<SkylarkClassObject> {
    private StructConstructor() {
      super(SkylarkClassObject.class, "struct");
    }

    @Override
    protected SkylarkClassObject createInstanceFromSkylark(Object[] args, Location loc) {
      @SuppressWarnings("unchecked")
      Map<String, Object> kwargs = (Map<String, Object>) args[0];
      return new SkylarkClassObject(this, kwargs, loc);
    }

    public SkylarkClassObject create(Map<String, Object> values, String message) {
      return new SkylarkClassObject(this, values, message);
    }

    public SkylarkClassObject create(Location loc) {
      return new SkylarkClassObject(this, loc);
    }
  }

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(FunctionSignature.KWARGS);

  protected NativeClassObjectConstructor(Class<VALUE> clazz, String name) {
    this(clazz, name, SIGNATURE);
  }

  protected NativeClassObjectConstructor(
      Class<VALUE> valueClass,
      String name,
      FunctionSignature.WithValues<Object, SkylarkType> signature) {
    super(name, signature, Location.BUILTIN);
    key = new NativeKey(name, getClass());
    this.valueClass = valueClass;
    errorMessageForInstances = String.format("'%s' object has no attribute '%%s'", name);
  }

  /**
   * equals() implements singleton class semantics.
   *
   * Every non-abstract derived class of {@link NativeClassObjectConstructor}
   * corresponds to a single declared provider.
   */
  @Override
  public final boolean equals(@Nullable Object other) {
    return other != null && this.getClass().equals(other.getClass());
  }

  /**
   * hashCode() implements singleton class semantics.
   *
   * Every non-abstract derived class of {@link NativeClassObjectConstructor}
   * corresponds to a single declared provider.
   */
  @Override
  public final int hashCode() {
    return getClass().hashCode();
  }

  @Override
  public String getPrintableName() {
    return getName();
  }

  @Override
  public String getErrorMessageFormatForInstances() {
    return errorMessageForInstances;
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
  protected SkylarkClassObject createInstanceFromSkylark(Object[] args, Location loc)
      throws EvalException {
    throw new EvalException(loc,
        String.format("'%s' cannot be constructed from Skylark", getPrintableName()));
  }

  public static Pair<String, String> getSerializedRepresentationForNativeKey(NativeKey key) {
    return Pair.of(key.name, key.aClass.getName());
  }

  public static NativeKey getNativeKeyFromSerializedRepresentation(Pair<String, String> serialized)
      throws ClassNotFoundException {
    Class<? extends NativeClassObjectConstructor> aClass =
        Class.forName(serialized.second).asSubclass(NativeClassObjectConstructor.class);
    return new NativeKey(serialized.first, aClass);
  }

  /**
   * A serializable representation of {@link NativeClassObjectConstructor}.
   *
   * Just a wrapper around its class.
   */
  // todo(vladmos,dslomov): when we allow declared providers in `requiredProviders`,
  // we will need to serialize this somehow.
  @Immutable
  public static final class NativeKey extends Key {
    private final String name;
    private final Class<? extends NativeClassObjectConstructor> aClass;

    private NativeKey(
        String name,
        Class<? extends NativeClassObjectConstructor> aClass) {
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
