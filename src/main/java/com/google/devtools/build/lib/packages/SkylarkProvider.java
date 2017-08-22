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
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Declared provider defined in Skylark.
 *
 * <p>This is a result of calling {@code provider()} function from Skylark ({@link
 * com.google.devtools.build.lib.analysis.skylark.SkylarkRuleClassFunctions#provider}).
 */
public class SkylarkProvider extends Provider implements SkylarkExportable {

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(FunctionSignature.KWARGS);

  @Nullable private SkylarkKey key;
  @Nullable private String errorMessageFormatForInstances;

  private static final String DEFAULT_ERROR_MESSAFE = "Object has no '%s' attribute.";

  /**
   * Creates a Skylark-defined Declared Provider ({@link Info} constructor).
   *
   * <p>Needs to be exported later.
   */
  public SkylarkProvider(String name, Location location) {
    this(name, SIGNATURE, location);
  }

  public SkylarkProvider(
      String name, FunctionSignature.WithValues<Object, SkylarkType> signature, Location location) {
    super(name, signature, location);
    this.errorMessageFormatForInstances = DEFAULT_ERROR_MESSAFE;
  }

  @Override
  protected Info createInstanceFromSkylark(Object[] args, Location loc) throws EvalException {
    @SuppressWarnings("unchecked")
    Map<String, Object> kwargs = (Map<String, Object>) args[0];
    return new SkylarkInfo(this, kwargs, loc);
  }

  @Override
  public boolean isExported() {
    return key != null;
  }

  @Override
  public SkylarkKey getKey() {
    Preconditions.checkState(isExported());
    return key;
  }

  @Override
  public String getPrintableName() {
    return key != null ? key.getExportedName() : getName();
  }

  @Override
  public String getErrorMessageFormatForInstances() {
    return errorMessageFormatForInstances;
  }

  @Override
  public void export(Label extensionLabel, String exportedName) {
    Preconditions.checkState(!isExported());
    this.key = new SkylarkKey(extensionLabel, exportedName);
    this.errorMessageFormatForInstances =
        String.format("'%s' object has no attribute '%%s'", exportedName);
  }

  @Override
  public int hashCode() {
    if (isExported()) {
      return getKey().hashCode();
    }
    return System.identityHashCode(this);
  }

  @Override
  public boolean equals(@Nullable Object otherObject) {
    if (!(otherObject instanceof SkylarkProvider)) {
      return false;
    }
    SkylarkProvider other = (SkylarkProvider) otherObject;

    if (this.isExported() && other.isExported()) {
      return this.getKey().equals(other.getKey());
    } else {
      return this == other;
    }
  }

  @Override
  public boolean isImmutable() {
    // Hash code for non exported constructors may be changed
    return isExported();
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<provider>");
  }

  /**
   * A serializable representation of Skylark-defined {@link SkylarkProvider} that uniquely
   * identifies all {@link SkylarkProvider}s that are exposed to SkyFrame.
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

    public String getExportedName() {
      return exportedName;
    }

    @Override
    public String toString() {
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
}
