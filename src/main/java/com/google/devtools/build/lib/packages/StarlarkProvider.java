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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Collection;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * A provider defined in Starlark rather than in native code.
 *
 * <p>This is a result of calling the {@code provider()} function from Starlark ({@link
 * com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions#provider}).
 *
 * <p>{@code StarlarkProvider}s may be either schemaless or schemaful. Instances of schemaless
 * providers can have any set of fields on them, whereas instances of schemaful providers may have
 * only the fields that are named in the schema.
 *
 * <p>Exporting a {@code StarlarkProvider} creates a key that is used to uniquely identify it.
 * Usually a provider is exported by calling {@link #export}, but a test may wish to just create a
 * pre-exported provider directly. Exported providers use only their key for {@link #equals} and
 * {@link #hashCode}.
 */
public final class StarlarkProvider implements StarlarkCallable, StarlarkExportable, Provider {

  private final Location location;

  // For schemaful providers, the sorted list of allowed field names.
  // The requirement for sortedness comes from StarlarkInfo.createFromNamedArgs,
  // as it lets us verify table ⊆ schema in O(n) time without temporaries.
  @Nullable private final ImmutableList<String> schema;

  /** Null iff this provider has not yet been exported. */
  @Nullable private Key key;

  /**
   * Creates an unexported {@link StarlarkProvider} with no schema.
   *
   * <p>The resulting object needs to be exported later (via {@link #export}).
   *
   * @param location the location of the Starlark definition for this provider (tests may use {@link
   *     Location#BUILTIN})
   */
  public static StarlarkProvider createUnexportedSchemaless(Location location) {
    return new StarlarkProvider(/*key=*/ null, /*schema=*/ null, location);
  }

  /**
   * Creates an unexported {@link StarlarkProvider} with a schema.
   *
   * <p>The resulting object needs to be exported later (via {@link #export}).
   *
   * @param schema the allowed field names for instances of this provider
   * @param location the location of the Starlark definition for this provider (tests may use {@link
   *     Location#BUILTIN})
   */
  // TODO(adonovan): in what sense is this "schemaful" if schema may be null?
  public static StarlarkProvider createUnexportedSchemaful(
      @Nullable Collection<String> schema, Location location) {
    return new StarlarkProvider(
        /*key=*/ null, schema == null ? null : ImmutableList.sortedCopyOf(schema), location);
  }

  /**
   * Creates an exported {@link StarlarkProvider} with no schema.
   *
   * @param key the key that identifies this provider
   * @param location the location of the Starlark definition for this provider (tests may use {@link
   *     Location#BUILTIN})
   */
  public static StarlarkProvider createExportedSchemaless(Key key, Location location) {
    return new StarlarkProvider(key, /*schema=*/ null, location);
  }

  /**
   * Creates an exported {@link StarlarkProvider} with no schema.
   *
   * @param key the key that identifies this provider
   * @param schema the allowed field names for instances of this provider
   * @param location the location of the Starlark definition for this provider (tests may use {@link
   *     Location#BUILTIN})
   */
  // TODO(adonovan): in what sense is this "schemaful" if schema may be null?
  public static StarlarkProvider createExportedSchemaful(
      Key key, @Nullable Collection<String> schema, Location location) {
    return new StarlarkProvider(
        key, schema == null ? null : ImmutableList.sortedCopyOf(schema), location);
  }

  /**
   * Constructs the provider.
   *
   * <p>If {@code key} is null, the provider is unexported. If {@code schema} is null, the provider
   * is schemaless.
   */
  private StarlarkProvider(
      @Nullable Key key, @Nullable ImmutableList<String> schema, Location location) {
    this.schema = schema;
    this.location = location;
    this.key = key;  // possibly null
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException, InterruptedException {
    if (positional.length > 0) {
      throw Starlark.errorf("%s: unexpected positional arguments", getName());
    }
    return StarlarkInfo.createFromNamedArgs(this, named, schema, thread.getCallerLocation());
  }

  @Override
  public Location getLocation() {
    return location;
  }

  @Override
  public boolean isExported() {
    return key != null;
  }

  @Override
  public Key getKey() {
    Preconditions.checkState(isExported());
    return key;
  }

  @Override
  public String getName() {
    return key != null ? key.getExportedName() : "<no name>";
  }

  @Override
  public String getPrintableName() {
    return getName();
  }

  /** Returns the list of fields allowed by this provider, or null if the provider is schemaless. */
  @Nullable
  // TODO(adonovan): rename getSchema.
  public ImmutableList<String> getFields() {
    return schema;
  }

  @Override
  public String getErrorMessageForUnknownField(String name) {
    return String.format(
        "'%s' value has no field or method '%s'",
        isExported() ? key.getExportedName() : "struct", name);
  }

  @Override
  public void export(Label extensionLabel, String exportedName) {
    Preconditions.checkState(!isExported());
    this.key = new Key(extensionLabel, exportedName);
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
    if (!(otherObject instanceof StarlarkProvider)) {
      return false;
    }
    StarlarkProvider other = (StarlarkProvider) otherObject;

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
  public void repr(Printer printer) {
    printer.append("<provider>");
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  /**
   * A serializable representation of Starlark-defined {@link StarlarkProvider} that uniquely
   * identifies all {@link StarlarkProvider}s that are exposed to SkyFrame.
   */
  @AutoCodec
  public static class Key extends Provider.Key {
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
    public String toString() {
      return exportedName;
    }

    @Override
    void fingerprint(Fingerprint fp) {
      // False => Not native.
      fp.addBoolean(false);
      fp.addString(extensionLabel.getCanonicalForm());
      fp.addString(exportedName);
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
