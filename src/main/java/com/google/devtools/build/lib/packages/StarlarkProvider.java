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
import com.google.devtools.build.lib.events.EventHandler;
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

  /** Null iff this provider has not yet been exported. Mutated by {@link export}. */
  @Nullable private Key key;

  /**
   * Returns a new empty builder.
   *
   * <p>By default (unless {@link Builder#setExported} is called), the builder will build a provider
   * which is unexported and would need to be exported later via {@link #export}.
   *
   * <p>By default (unless {@link Builder#setSchema} is called), the builder will build a provider
   * which is schemaless.
   *
   * @param location the location of the Starlark definition for this provider (tests may use {@link
   *     Location#BUILTIN})
   */
  public static Builder builder(Location location) {
    return new Builder(location);
  }

  /** A builder which may be used to construct a StarlarkProvider. */
  public static final class Builder {
    private final Location location;

    @Nullable private ImmutableList<String> schema;

    @Nullable private Key key;

    private Builder(Location location) {
      this.location = location;
    }

    /**
     * Sets the schema (the list of allowed field names) for instances of the provider built by this
     * builder.
     */
    public Builder setSchema(Collection<String> schema) {
      this.schema = ImmutableList.sortedCopyOf(schema);
      return this;
    }

    /** Sets the provider built by this builder to be exported with the given key. */
    public Builder setExported(Key key) {
      this.key = key;
      return this;
    }

    /** Builds a StarlarkProvider. */
    public StarlarkProvider build() {
      return new StarlarkProvider(location, schema, key);
    }
  }

  /**
   * Constructs the provider.
   *
   * <p>If {@code key} is null, the provider is unexported. If {@code schema} is null, the provider
   * is schemaless.
   */
  private StarlarkProvider(
      Location location, @Nullable ImmutableList<String> schema, @Nullable Key key) {
    this.location = location;
    this.schema = schema;
    this.key = key;
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException {
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
  public void export(EventHandler handler, Label extensionLabel, String exportedName) {
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
  public static final class Key extends Provider.Key {
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
