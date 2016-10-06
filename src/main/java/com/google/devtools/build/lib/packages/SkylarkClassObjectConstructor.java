// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Declared Provider (a constructor for {@link SkylarkClassObject}).
 */
@SkylarkModule(name = "provider",
    doc = "A constructor for simple value objects. "
        + "See the global <a href=\"globals.html#provider\">provider</a> function "
        + "for more details."
)
public final class SkylarkClassObjectConstructor extends BaseFunction implements SkylarkExportable {
  /**
   * "struct" function.
   */
  public static final SkylarkClassObjectConstructor STRUCT =
      createNativeConstructable("struct");


  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(FunctionSignature.KWARGS);

  @Nullable
  private Key key;

  /**
   * Some native declared providers are not constructable from Skylark.
   */
  private final boolean isConstructable;

  private SkylarkClassObjectConstructor(String name, Location location) {
    super(name, SIGNATURE, location);
    // All Skylark-defined declared providers are constructable.
    this.isConstructable = true;
  }

  private SkylarkClassObjectConstructor(String name, boolean isConstructable) {
    super(name, SIGNATURE, Location.BUILTIN);
    this.key = new NativeKey();
    this.isConstructable = isConstructable;
  }

  /**
   * Create a native Declared Provider ({@link SkylarkClassObject} constructor)
   */
  public static SkylarkClassObjectConstructor createNative(String name) {
    return new SkylarkClassObjectConstructor(name, false);
  }

  /**
   * Create a native Declared Provider ({@link SkylarkClassObject} constructor)
   * that can be constructed from Skylark.
   */
  public static SkylarkClassObjectConstructor createNativeConstructable(String name) {
    return new SkylarkClassObjectConstructor(name, true);
  }


  /**
   * Create a Skylark-defined Declared Provider ({@link SkylarkClassObject} constructor)
   *
   * Needs to be exported later.
   */
  public static SkylarkClassObjectConstructor createSkylark(String name, Location location) {
    return new SkylarkClassObjectConstructor(name, location);
  }

  @Override
  protected Object call(Object[] args, @Nullable FuncallExpression ast, @Nullable Environment env)
      throws EvalException, InterruptedException {
    if (!isConstructable) {
      Location loc = ast != null ? ast.getLocation() : Location.BUILTIN;
      throw new EvalException(loc,
          String.format("'%s' cannot be constructed from Skylark", getPrintableName()));
    }
    @SuppressWarnings("unchecked")
    Map<String, Object> kwargs = (Map<String, Object>) args[0];
    return new SkylarkClassObject(this, kwargs, ast != null ? ast.getLocation() : Location.BUILTIN);
  }

  /**
   * Creates a built-in class object (i.e. without creation loc). The errorMessage has to have
   * exactly one '%s' parameter to substitute the field name.
   */
  public SkylarkClassObject create(Map<String, Object> values, String message) {
    return new SkylarkClassObject(this, values, message);
  }

  @Override
  public boolean isExported() {
    return key != null;
  }

  public Key getKey() {
    Preconditions.checkState(isExported());
    return key;
  }

  public String getPrintableName() {
    return key != null ? key.getExportedName() : getName();
  }

  @Override
  public void export(Label extensionLabel, String exportedName) {
    Preconditions.checkState(!isExported());
    this.key = new SkylarkKey(extensionLabel, exportedName);
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }

  @Override
  public boolean equals(@Nullable Object other) {
    return other == this;
  }


  /**
   * A representation of {@link SkylarkClassObjectConstructor}.
   */
  // todo(vladmos,dslomov): when we allow declared providers in `requiredProviders`,
  // we will need to serialize this somehow.
  public abstract static class Key {
    private Key() {}

    public abstract String getExportedName();
  }

  /**
   * A serializable representation of Skylark-defined {@link SkylarkClassObjectConstructor}
   * that uniquely identifies all {@link SkylarkClassObjectConstructor}s that
   * are exposed to SkyFrame.
   */
  public static class SkylarkKey extends Key {
    private final Label extensionLabel;
    private final String exportedName;

    public SkylarkKey(Label extensionLabel, String exportedName) {
      this.extensionLabel = Preconditions.checkNotNull(extensionLabel);
      this.exportedName = Preconditions.checkNotNull(exportedName);
    }

    public Label getExtensionLabel() {
      return extensionLabel;
    }

    @Override
    public String getExportedName() {
      return exportedName;
    }

    @Override
    public int hashCode() {
      return Objects.hash(extensionLabel, exportedName);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }

      if (!(obj instanceof SkylarkKey)) {
        return false;
      }
      SkylarkKey other = (SkylarkKey) obj;
      return Objects.equals(this.extensionLabel, other.extensionLabel)
          && Objects.equals(this.exportedName, other.exportedName);
    }
  }

  /**
   * A representation of {@link SkylarkClassObjectConstructor} defined in native code.
   */
  // todo(vladmos,dslomov): when we allow declared providers in `requiredProviders`,
  // we will need to serialize this somehow.
  public final class NativeKey extends Key {
    private NativeKey() {
    }

    @Override
    public String getExportedName() {
      return SkylarkClassObjectConstructor.this.getName();
    }

    public SkylarkClassObjectConstructor getConstructor() {
      return SkylarkClassObjectConstructor.this;
    }
  }
}
