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
 * A constructor for {@link SkylarkClassObject}.
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
      new SkylarkClassObjectConstructor("struct");


  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(FunctionSignature.KWARGS);

  @Nullable
  private Key key;

  public SkylarkClassObjectConstructor(String name, Location location) {
    super(name, SIGNATURE, location);
  }

  public SkylarkClassObjectConstructor(String name) {
    this(name, Location.BUILTIN);
  }

  @Override
  protected Object call(Object[] args, @Nullable FuncallExpression ast, @Nullable Environment env)
      throws EvalException, InterruptedException {
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
    return key != null ? key.exportedName : getName();
  }

  @Override
  public void export(Label extensionLabel, String exportedName) {
    Preconditions.checkState(!isExported());
    this.key = new Key(extensionLabel, exportedName);
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
   * A serializable representation of {@link SkylarkClassObjectConstructor}
   * that uniquely identifies all {@link SkylarkClassObjectConstructor}s that
   * are exposed to SkyFrame.
   */
  public static class Key {
    private final Label extensionLabel;
    private final String exportedName;

    public Key(Label extensionLabel, String exportedName) {
      this.extensionLabel = Preconditions.checkNotNull(extensionLabel);
      this.exportedName = Preconditions.checkNotNull(exportedName);
    }

    public Label getExtensionLabel() {
      return extensionLabel;
    }

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

      if (!(obj instanceof Key)) {
        return false;
      }
      Key other = (Key) obj;
      return Objects.equals(this.extensionLabel, other.extensionLabel)
          && Objects.equals(this.exportedName, other.exportedName);
    }
  }
}
