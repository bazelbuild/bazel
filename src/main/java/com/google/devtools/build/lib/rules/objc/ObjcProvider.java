// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.collect.nestedset.Order.LINK_ORDER;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.BuiltinProvider.WithLegacyStarlarkName;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.starlarkbuildapi.objc.ObjcProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;

/**
 * A provider that provides all linking and miscellaneous information in the transitive closure of
 * its deps that are needed for building Objective-C rules. Most of the compilation information has
 * been migrated to {@code CcInfo}. The objc proto strict dependency include paths are still here
 * and stored in a special, non-propagated field {@code strictDependencyIncludes}.
 *
 * <p>The rest of the information is stored in two generic maps indexed by {@code ObjcProvider.Key}:
 *
 * <ul>
 *   <li>{@code items}: This map contains items that are propagated transitively to all dependent
 *       ObjcProviders.
 *   <li>{@code directItems}: This multimap contains items whose values originate from this
 *       ObjcProvider (as opposed to those that came from a dependent ObjcProvider). {@link
 *       #KEYS_FOR_DIRECT} contains the keys whose items are inserted into this map. The map is
 *       created as a performance optimization for IDEs (i.e. Tulsi), so that the IDEs don't have to
 *       flatten large transitive nested sets returned by ObjcProvider queries. It does not
 *       materially affect other operations of the ObjcProvider.
 * </ul>
 */
// TODO(adonovan): this is an info, not a provider; rename.
@Immutable
public final class ObjcProvider implements Info, ObjcProviderApi<Artifact> {

  /** Starlark name for the ObjcProvider. */
  public static final String STARLARK_NAME = "objc";

  /** Expected suffix for a framework-containing directory. */
  public static final String FRAMEWORK_SUFFIX = ".framework";

  /**
   * Represents one of the things this provider can provide transitively. Things are provided as
   * {@link NestedSet}s of type E.
   */
  @Immutable
  public static class Key<E> {
    private final Order order;
    private final String starlarkKeyName;
    private final Class<E> type;

    private Key(Order order, String starlarkKeyName, Class<E> type) {
      this.order = Preconditions.checkNotNull(order);
      this.starlarkKeyName = starlarkKeyName;
      this.type = type;
    }

    /** Returns the name of the collection represented by this key in the Starlark provider. */
    public String getStarlarkKeyName() {
      return starlarkKeyName;
    }

    /** Returns the type of nested set keyed in the ObjcProvider by this key. */
    public Class<E> getType() {
      return type;
    }
  }

  /** Contains all source files. */
  public static final Key<Artifact> SOURCE = new Key<>(STABLE_ORDER, "source", Artifact.class);

  /**
   * Include search paths {@code -I} that are stored specially in their own field, and not
   * propagated transitively.
   */
  @SerializationConstant
  public static final Key<PathFragment> STRICT_INCLUDE =
      new Key<>(LINK_ORDER, "strict_include", PathFragment.class);

  /**
   * Clang umbrella header. Public headers are #included in umbrella headers to be compatible with
   * J2ObjC segmented headers.
   */
  @SerializationConstant
  public static final Key<Artifact> UMBRELLA_HEADER =
      new Key<>(STABLE_ORDER, "umbrella_header", Artifact.class);

  /** Clang module maps, used to enforce proper use of private header files. */
  @SerializationConstant
  public static final Key<Artifact> MODULE_MAP =
      new Key<>(STABLE_ORDER, "module_map", Artifact.class);

  /** Static libraries that are built from J2ObjC-translated Java code. */
  @SerializationConstant
  public static final Key<Artifact> J2OBJC_LIBRARY =
      new Key<>(LINK_ORDER, "j2objc_library", Artifact.class);

  // Items which are propagated transitively to dependents.
  private final ImmutableMap<Key<?>, NestedSet<?>> items;

  /** Strict dependency includes */
  private final ImmutableList<PathFragment> strictDependencyIncludes;

  /**
   * This is intended to be used by clients which need to collect transitive information without
   * paying the O(n^2) behavior to flatten it during analysis time.
   *
   * <p>For example, IDEs may use this to identify all direct header files for a target and fetch
   * all transitive headers from its dependencies by recursing through this field.
   */
  private final ImmutableListMultimap<Key<?>, ?> directItems;

  /** All keys in ObjcProvider that will be passed in the corresponding Starlark provider. */
  static final ImmutableList<Key<?>> KEYS_FOR_STARLARK =
      ImmutableList.<Key<?>>of(J2OBJC_LIBRARY, MODULE_MAP, SOURCE, UMBRELLA_HEADER);

  /**
   * Keys that should be kept as directItems. This is limited to a few keys that have larger
   * performance implications when flattened in a transitive fashion and/or require non-transitive
   * access (e.g. what module map did a target generate?).
   *
   * <p>Keys:
   *
   * <ul>
   *   <li>SOURCE: To expose all source files, including generated J2Objc source files, to IDEs.
   *   <li>MODULE_MAP: To expose generated module maps to IDEs (only one is expected per target).
   * </ul>
   */
  static final ImmutableSet<Key<?>> KEYS_FOR_DIRECT = ImmutableSet.<Key<?>>of(MODULE_MAP, SOURCE);

  public ImmutableList<PathFragment> getStrictDependencyIncludes() {
    return strictDependencyIncludes;
  }

  @Override
  public Depset /*<String>*/ strictIncludeForStarlark() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getStrictDependencyIncludes().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset /*<Artifact>*/ j2objcLibrary() {
    return Depset.of(Artifact.class, get(J2OBJC_LIBRARY));
  }

  @Override
  public Depset /*<Artifact>*/ moduleMap() {
    return Depset.of(Artifact.class, get(MODULE_MAP));
  }

  @Override
  public Sequence<Artifact> directModuleMaps() {
    return getDirect(MODULE_MAP);
  }

  @Override
  public Depset /*<Artifact>*/ sourceForStarlark() {
    return Depset.of(Artifact.class, source());
  }

  NestedSet<Artifact> source() {
    return get(SOURCE);
  }

  @Override
  public Sequence<Artifact> directSources() {
    return getDirect(SOURCE);
  }

  @Override
  public Depset /*<Artifact>*/ umbrellaHeader() {
    return Depset.of(Artifact.class, get(UMBRELLA_HEADER));
  }

  /**
   * All keys in ObjcProvider that are explicitly not exposed to Starlark. This is used for testing
   * and verification purposes to ensure that a conscious decision is made for all keys; by default,
   * keys should be exposed to Starlark: a comment outlining why a key is omitted from Starlark
   * should follow each such case.
   */
  @VisibleForTesting
  static final ImmutableList<Key<?>> KEYS_NOT_IN_STARLARK =
      ImmutableList.<Key<?>>of(
          // Strict include is handled specially.
          STRICT_INCLUDE);

  /**
   * Returns the Starlark key for the given string, or null if no such key exists or is available to
   * Starlark.
   */
  @Nullable
  static Key<?> getStarlarkKeyForString(String keyName) {
    for (Key<?> candidateKey : KEYS_FOR_STARLARK) {
      if (candidateKey.getStarlarkKeyName().equals(keyName)) {
        return candidateKey;
      }
    }
    return null;
  }

  /** Starlark constructor and identifier for ObjcProvider. */
  public static final BuiltinProvider<ObjcProvider> STARLARK_CONSTRUCTOR = new Constructor();

  private ObjcProvider(
      ImmutableMap<Key<?>, NestedSet<?>> items,
      ImmutableList<PathFragment> strictDependencyIncludes,
      ImmutableListMultimap<Key<?>, ?> directItems) {
    this.items = Preconditions.checkNotNull(items);
    this.strictDependencyIncludes = Preconditions.checkNotNull(strictDependencyIncludes);
    this.directItems = Preconditions.checkNotNull(directItems);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public BuiltinProvider<ObjcProvider> getProvider() {
    return STARLARK_CONSTRUCTOR;
  }

  /** All artifacts, bundleable files, etc. of the type specified by {@code key}. */
  @SuppressWarnings("unchecked")
  public <E> NestedSet<E> get(Key<E> key) {
    Preconditions.checkNotNull(key);
    if (items.containsKey(key)) {
      return (NestedSet<E>) items.get(key);
    } else {
      return new NestedSetBuilder<E>(key.order).build();
    }
  }

  /** All direct artifacts, bundleable files, etc. of the type specified by {@code key}. */
  @SuppressWarnings({"rawtypes", "unchecked"})
  public <E> Sequence<E> getDirect(Key<E> key) {
    if (directItems.containsKey(key)) {
      return StarlarkList.immutableCopyOf((List) directItems.get(key));
    }
    return StarlarkList.empty();
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public static class Builder {

    private final Map<Key<?>, NestedSetBuilder<?>> items = new HashMap<>();
    private final ImmutableList.Builder<PathFragment> strictDependencyIncludes =
        ImmutableList.builder();

    // Only includes items or lists added directly, never flattens any NestedSets.
    private final ImmutableListMultimap.Builder<Key<?>, ?> directItems =
        new ImmutableListMultimap.Builder<>();

    public Builder() {}

    private static void maybeAddEmptyBuilder(Map<Key<?>, NestedSetBuilder<?>> set, Key<?> key) {
      set.computeIfAbsent(key, k -> new NestedSetBuilder<>(k.order));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void uncheckedAddAll(Key key, Iterable toAdd) {
      maybeAddEmptyBuilder(items, key);
      items.get(key).addAll(toAdd);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    protected void uncheckedAddAllDirect(Key key, Iterable<?> toAdd) {
      directItems.putAll(key, (Iterable) toAdd);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    protected void uncheckedAddTransitive(Key key, NestedSet toAdd) {
      maybeAddEmptyBuilder(items, key);
      items.get(key).addTransitive(toAdd);
    }

    /**
     * Add all elements from providers, and propagate them to any (transitive) dependers on this
     * ObjcProvider.
     */
    @CanIgnoreReturnValue
    public Builder addTransitiveAndPropagate(Iterable<ObjcProvider> providers) {
      for (ObjcProvider provider : providers) {
        addTransitiveAndPropagate(provider);
      }
      return this;
    }

    /**
     * Add all keys and values from provider, and propagate them to any (transitive) dependers on
     * this ObjcProvider.
     */
    @CanIgnoreReturnValue
    public Builder addTransitiveAndPropagate(ObjcProvider provider) {
      for (Map.Entry<Key<?>, NestedSet<?>> typeEntry : provider.items.entrySet()) {
        uncheckedAddTransitive(typeEntry.getKey(), typeEntry.getValue());
      }
      return this;
    }

    /**
     * Add all elements from a single key of the given provider, and propagate them to any
     * (transitive) dependers on this ObjcProvider.
     */
    @CanIgnoreReturnValue
    public Builder addTransitiveAndPropagate(Key<?> key, ObjcProvider provider) {
      if (provider.items.containsKey(key)) {
        uncheckedAddTransitive(key, provider.items.get(key));
      }
      return this;
    }

    /**
     * Adds elements in items, and propagate them to any (transitive) dependers on this
     * ObjcProvider.
     */
    @CanIgnoreReturnValue
    public <E> Builder addTransitiveAndPropagate(Key<E> key, NestedSet<E> items) {
      uncheckedAddTransitive(key, items);
      return this;
    }

    /** Add element, and propagate it to any (transitive) dependers on this ObjcProvider. */
    @CanIgnoreReturnValue
    public <E> Builder add(Key<E> key, E toAdd) {
      uncheckedAddAll(key, ImmutableList.of(toAdd));
      return this;
    }

    @CanIgnoreReturnValue
    public <E> Builder addDirect(Key<E> key, E toAdd) {
      Preconditions.checkState(KEYS_FOR_DIRECT.contains(key));
      uncheckedAddAllDirect(key, ImmutableList.of(toAdd));
      return this;
    }

    /**
     * Add elements in toAdd, and propagate them to any (transitive) dependers on this ObjcProvider.
     */
    public <E> Builder addAll(Key<E> key, NestedSet<? extends E> toAdd) {
      return addAll(key, toAdd.toList());
    }

    /**
     * Add elements in toAdd, and propagate them to any (transitive) dependers on this ObjcProvider.
     */
    @CanIgnoreReturnValue
    public <E> Builder addAll(Key<E> key, Iterable<? extends E> toAdd) {
      uncheckedAddAll(key, toAdd);
      return this;
    }

    @CanIgnoreReturnValue
    public <E> Builder addAllDirect(Key<E> key, Iterable<? extends E> toAdd) {
      Preconditions.checkState(KEYS_FOR_DIRECT.contains(key));
      uncheckedAddAllDirect(key, toAdd);
      return this;
    }

    @CanIgnoreReturnValue
    protected Builder addStrictDependencyIncludes(Iterable<PathFragment> includes) {
      strictDependencyIncludes.addAll(includes);
      return this;
    }

    ObjcProvider build() {
      ImmutableMap.Builder<Key<?>, NestedSet<?>> propagatedBuilder = new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : items.entrySet()) {
        propagatedBuilder.put(typeEntry.getKey(), typeEntry.getValue().build());
      }
      return new ObjcProvider(
          propagatedBuilder.buildOrThrow(), strictDependencyIncludes.build(), directItems.build());
    }
  }

  /** A builder for this context, specialized for Starlark use. */
  public static final class StarlarkBuilder extends Builder {
    public StarlarkBuilder() {}

    /**
     * Add elements in toAdd with the given key from Starlark. An error is thrown if toAdd is not an
     * appropriate Depset.
     */
    void addElementsFromStarlark(Key<?> key, Object starlarkToAdd) throws EvalException {
      NestedSet<?> toAdd = ObjcProviderStarlarkConverters.convertToJava(key, starlarkToAdd);
      uncheckedAddTransitive(key, toAdd);

      if (KEYS_FOR_DIRECT.contains(key)) {
        uncheckedAddAllDirect(key, toAdd.toList());
      }
    }

    /**
     * Adds the given providers from Starlark. An error is thrown if toAdd is not an iterable of
     * ObjcProvider instances.
     */
    @SuppressWarnings("unchecked")
    void addProvidersFromStarlark(Object toAdd) throws EvalException {
      if (!(toAdd instanceof Iterable)) {
        throw Starlark.errorf(AppleStarlarkCommon.BAD_PROVIDERS_ITER_ERROR, Starlark.type(toAdd));
      } else {
        Iterable<Object> toAddIterable = (Iterable<Object>) toAdd;
        for (Object toAddObject : toAddIterable) {
          if (!(toAddObject instanceof ObjcProvider objcProvider)) {
            throw Starlark.errorf(
                AppleStarlarkCommon.BAD_PROVIDERS_ELEM_ERROR, Starlark.type(toAddObject));
          } else {
            this.addTransitiveAndPropagate(objcProvider);
          }
        }
      }
    }

    /**
     * Adds the given strict include paths from Starlark. An error is thrown if starlarkToAdd is not
     * an appropriate Depset.
     */
    @SuppressWarnings("unchecked")
    void addStrictIncludeFromStarlark(Object starlarkToAdd) throws EvalException {
      NestedSet<PathFragment> toAdd =
          (NestedSet<PathFragment>)
              ObjcProviderStarlarkConverters.convertToJava(STRICT_INCLUDE, starlarkToAdd);

      addStrictDependencyIncludes(toAdd.toList());
    }
  }

  private static class Constructor extends BuiltinProvider<ObjcProvider>
      implements WithLegacyStarlarkName {
    public Constructor() {
      super(ObjcProvider.STARLARK_NAME, ObjcProvider.class);
    }

    @Override
    public String getStarlarkName() {
      return STARLARK_NAME;
    }

    @Override
    public String getErrorMessageForUnknownField(String name) {
      return String.format("ObjcProvider field '%s' could not be instantiated", name);
    }
  }
}
