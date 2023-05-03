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


import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReferenceArray;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
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
 * <p>{@code StarlarkProvider} may have a custom initializer callback, which might perform
 * preprocessing or validation of field values. This callback (if defined) is automatically invoked
 * when the provider is called. To create instances of the provider without calling the initializer
 * callback, use the callable returned by {@code StarlarkProvider#createRawConstructor}.
 *
 * <p>Exporting a {@code StarlarkProvider} creates a key that is used to uniquely identify it.
 * Usually a provider is exported by calling {@link #export}, but a test may wish to just create a
 * pre-exported provider directly. Exported providers use only their key for {@link #equals} and
 * {@link #hashCode}.
 */
public final class StarlarkProvider implements StarlarkCallable, StarlarkExportable, Provider {

  private final Location location;

  @Nullable private final String documentation;

  // For schemaful providers, the sorted list of allowed field names.
  // The requirement for sortedness comes from StarlarkInfoWithSchema and lets us bisect the fields.
  @Nullable private final ImmutableList<String> fields;

  // For schemaful providers, an optional map from field names to documentation strings (if any). In
  // accordance with the provider() Starlark API, either all schema fields have documentation
  // strings (possibly empty strings), or none do. The iteration order is the order of fields in the
  // provider() invocation in Starlark - thus, *not* the order of the `fields` list above.
  @Nullable private final ImmutableMap<String, Optional<String>> schema;

  // Optional custom initializer callback. If present, it is invoked with the same positional and
  // keyword arguments as were passed to the provider constructor. The return value must be a
  // Starlark dict mapping field names (string keys) to their values.
  @Nullable private final StarlarkCallable init;

  /** Null iff this provider has not yet been exported. Mutated by {@link export}. */
  @Nullable private Key key;

  /**
   * For schemaful providers, an array of metadata concerning depset optimization.
   *
   * <p>Each index in the array holds an optional (nullable) depset element type. The value at that
   * index is initialized to be the element type of the first non-empty Depset to ever be stored in
   * the corresponding field from {@link #schema} on any instance of this provider, globally. If no
   * depsets (or only empty depsets) are ever stored in a field, the value at its index in this
   * array will remain null.
   *
   * <p>Whenever a field is stored in an instance of this provider type, if the value is a depset
   * whose element type matches the one stored in this array, it is optimized by unwrapping it down
   * to its {@code NestedSet}. Upon retrieval, the depset wrapper is reconstructed using this saved
   * element type.
   *
   * <p>The optimization may (harmlessly) fail to apply for provider fields that are not strongly
   * typed across all instances.
   *
   * <p>For large builds, this optimization has been observed to save half a percent in retained
   * heap.
   *
   * <p>In the future, the ad hoc heuristic of examining the first stored non-empty depset might be
   * replaced by stronger type information in the provider's Starlark declaration. However, this
   * optimization would remain relevant for provider declarations that do not supply such type info.
   */
  @Nullable private transient AtomicReferenceArray<Class<?>> depsetTypePredictor;

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

    @Nullable private String documentation;

    @Nullable private ImmutableMap<String, Optional<String>> schema;

    @Nullable private StarlarkCallable init;

    @Nullable private Key key;

    private Builder(Location location) {
      this.location = location;
    }

    /**
     * Sets the list of allowed fields for the provider built by this builder, and marks the fields'
     * documentation as empty.
     */
    @CanIgnoreReturnValue
    public Builder setSchema(Collection<String> fields) {
      ImmutableMap.Builder<String, Optional<String>> builder = ImmutableMap.builder();
      for (String field : fields) {
        builder.put(field, Optional.empty());
      }
      this.schema = builder.buildOrThrow();
      return this;
    }

    /**
     * Sets the list of allowed field names and their corresponding documentation strings for the
     * provider built by this builder.
     */
    @CanIgnoreReturnValue
    public Builder setSchema(Map<String, String> schemaWithDocumentation) {
      ImmutableMap.Builder<String, Optional<String>> builder = ImmutableMap.builder();
      for (Map.Entry<String, String> entry : schemaWithDocumentation.entrySet()) {
        builder.put(entry.getKey(), Optional.of(entry.getValue()));
      }
      this.schema = builder.buildOrThrow();
      return this;
    }

    /** Sets the documentation string for the provider built by this builder. */
    @CanIgnoreReturnValue
    public Builder setDocumentation(String documentation) {
      this.documentation = documentation;
      return this;
    }

    /**
     * Sets the custom initializer callback for the provider built by this builder.
     *
     * <p>The initializer callback will be automatically invoked when the provider is called. To
     * bypass the custom initializer callback, use the callable returned by {@link
     * StarlarkProvider#createRawConstructor}.
     *
     * @param init A callback that accepts the arguments passed to the provider constructor, and
     *     which returns a dict mapping field names to their values. The resulting provider instance
     *     is created as though the dict were passed as **kwargs to the raw constructor. In
     *     particular, for a schemaful provider, the dict may not contain keys not listed in the
     *     schema.
     */
    @CanIgnoreReturnValue
    public Builder setInit(StarlarkCallable init) {
      this.init = init;
      return this;
    }

    /** Sets the provider built by this builder to be exported with the given key. */
    @CanIgnoreReturnValue
    public Builder setExported(Key key) {
      this.key = key;
      return this;
    }

    /** Builds a StarlarkProvider. */
    public StarlarkProvider build() {
      return new StarlarkProvider(location, documentation, schema, init, key);
    }
  }

  /**
   * Constructs the provider.
   *
   * <p>If {@code schema} is null, the provider is schemaless. If {@code init} is null, no custom
   * initializer callback will be used (i.e., calling the provider is the same as simply calling the
   * raw constructor). If {@code key} is null, the provider is unexported.
   */
  private StarlarkProvider(
      Location location,
      @Nullable String documentation,
      @Nullable ImmutableMap<String, Optional<String>> schema,
      @Nullable StarlarkCallable init,
      @Nullable Key key) {
    this.location = location;
    this.documentation = documentation;
    this.fields = schema != null ? ImmutableList.sortedCopyOf(schema.keySet()) : null;
    this.schema = schema;
    this.init = init;
    this.key = key;
    if (schema != null) {
      depsetTypePredictor = new AtomicReferenceArray<>(schema.size());
    }
  }

  private static Object[] toNamedArgs(Object value, String descriptionForError)
      throws EvalException {
    Dict<String, Object> kwargs = Dict.cast(value, String.class, Object.class, descriptionForError);
    Object[] named = new Object[2 * kwargs.size()];
    int i = 0;
    for (Map.Entry<String, Object> e : kwargs.entrySet()) {
      named[i++] = e.getKey();
      named[i++] = e.getValue();
    }
    return named;
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws InterruptedException, EvalException {
    if (init == null) {
      return fastcallRawConstructor(thread, positional, named);
    }

    Object initResult = Starlark.fastcall(thread, init, positional, named);
    // The code-path for providers with schema could be optimised to skip the call to toNamedArgs.
    // As it is, we copy the map to an alternating key-value Object array, and then extract just
    // the values into another array.
    return createFromNamedArgs(
        toNamedArgs(initResult, "return value of provider init()"), thread.getCallerLocation());
  }

  private Object fastcallRawConstructor(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException {
    if (positional.length > 0) {
      throw Starlark.errorf("%s: unexpected positional arguments", getName());
    }
    return createFromNamedArgs(named, thread.getCallerLocation());
  }

  private StarlarkInfo createFromNamedArgs(Object[] named, Location loc) throws EvalException {
    return schema != null
        ? StarlarkInfoWithSchema.createFromNamedArgs(this, named, loc)
        : StarlarkInfoNoSchema.createFromNamedArgs(this, named, loc);
  }

  private static final class RawConstructor implements StarlarkCallable {
    private final StarlarkProvider provider;

    private RawConstructor(StarlarkProvider provider) {
      this.provider = provider;
    }

    @Override
    public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
        throws EvalException {
      return provider.fastcallRawConstructor(thread, positional, named);
    }

    @Override
    public String getName() {
      StringBuilder name = new StringBuilder("<raw constructor");
      if (provider.isExported()) {
        name.append(" for ").append(provider.getName());
      }
      name.append(">");
      return name.toString();
    }

    @Override
    public Location getLocation() {
      return provider.location;
    }
  }

  public StarlarkCallable createRawConstructor() {
    return new RawConstructor(this);
  }

  @Nullable
  @VisibleForTesting
  public StarlarkCallable getInit() {
    return init;
  }

  @Override
  public Location getLocation() {
    return location;
  }

  /**
   * Returns the value of the doc parameter passed to {@code provider()} in Starlark, or an empty
   * Optional if a doc parameter was not provided.
   */
  public Optional<String> getDocumentation() {
    return Optional.ofNullable(documentation);
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

  /**
   * Returns the sorted list of fields allowed by this provider, or null if the provider is
   * schemaless.
   */
  @Nullable
  public ImmutableList<String> getFields() {
    return fields;
  }

  /**
   * Returns the map of fields allowed by this provider mapping to their corresponding documentation
   * strings (if any), or null if this provider is schemaless.
   *
   * <p>The returned map's iteration order matches the order of fields in the {@code provider()}
   * invocation in Starlark - thus, different from the order of fields in {@link #getFields}.
   */
  @Nullable
  public ImmutableMap<String, Optional<String>> getSchema() {
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
   * For schemaful providers, given a value to store in the field identified by {@code index},
   * returns a possibly optimized version of the value. The result (optimized or not) should be
   * decoded by {@link #retrieveOptimizedField}.
   *
   * <p>Mutable values are never optimized.
   */
  Object optimizeField(int index, Object value) {
    if (value instanceof Depset) {
      Preconditions.checkArgument(depsetTypePredictor != null);
      Depset depset = (Depset) value;
      if (depset.isEmpty()) {
        // Most empty depsets have the empty (null) type. We can't store this type because it
        // would clash with whatever the actual element type is for non-empty depsets in that
        // field. So instead just store the optimized (unwrapped) NestedSet without any type
        // information, and assume it's the empty type upon retrieval.
        //
        // This only loses information in the relatively rare case of a native-constructed empty
        // depset with a type restriction (e.g. empty set of artifacts). In that scenario, an
        // empty depset retrieved from the provider may "incorrectly" allow itself to participate
        // in a union with depsets of other types, whereas the original depset would trigger a
        // Starlark eval error. This is a user-observable difference but a very minor one; the
        // hazard would be logical errors that are masked by the provider machinery but triggered
        // by a refactoring of Starlark code. See TODO in Depset#of(Class, NestedSet) for notes
        // about eliminating this semantic confusion.
        //
        // This problem shouldn't arise for non-empty depsets since distinct non-empty element
        // types are not compatible with one another (i.e. there's no Depset<Any> schema).
        return depset.getSet();
      }
      Class<?> elementClass = depset.getElementClass();
      if (depsetTypePredictor.compareAndExchange(index, null, elementClass) == elementClass) {
        return depset.getSet();
      }
    }
    return value;
  }

  Object retrieveOptimizedField(int index, Object value) {
    if (value instanceof NestedSet<?>) {
      // We subvert Depset.of()'s static type checking for consistency between the type token and
      // NestedSet type. This is safe because these values came from a previous Depset, so we
      // already know they're consistent.
      @SuppressWarnings("unchecked")
      NestedSet<Object> nestedSet = (NestedSet<Object>) value;
      if (nestedSet.isEmpty()) {
        // This matches empty depsets created in Starlark with `depset()`.
        return Depset.of(Object.class, nestedSet);
      }
      @SuppressWarnings("unchecked") // can't parametrize Class literal by a non-raw type
      Depset depset = Depset.of((Class<Object>) depsetTypePredictor.get(index), nestedSet);
      return depset;
    }
    return value;
  }

  boolean isOptimised(int index, Object value) {
    return value instanceof NestedSet<?>;
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
