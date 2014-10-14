// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

import java.util.HashMap;
import java.util.Map;

/**
 * A provider that provides all compiling and linking information in the transitive closure of its
 * deps that are needed for building Objective-C rules.
 */
@Immutable
final class ObjcProvider implements TransitiveInfoProvider {
  /**
   * Represents one of the things this provider can provide transitively. Things are provided as
   * {@link NestedSet}s of type E.
   */
  public static class Key<E> {
    private final Order order;

    private Key(Order order) {
      this.order = Preconditions.checkNotNull(order);
    }
  }

  public static final Key<Artifact> LIBRARY = new Key<>(LINK_ORDER);
  public static final Key<Artifact> IMPORTED_LIBRARY = new Key<>(LINK_ORDER);
  public static final Key<Artifact> HEADER = new Key<>(STABLE_ORDER);
  public static final Key<PathFragment> INCLUDE = new Key<>(LINK_ORDER);
  public static final Key<Artifact> ASSET_CATALOG = new Key<>(STABLE_ORDER);

  /**
   * Added to {@link TargetControl#getGeneralResourceFileList()} when running Xcodegen.
   */
  public static final Key<Artifact> GENERAL_RESOURCE_FILE = new Key<>(STABLE_ORDER);

  /**
   * Exec paths of {@code .bundle} directories corresponding to imported bundles to link.
   * These are passed to Xcodegen.
   */
  public static final Key<PathFragment> BUNDLE_IMPORT_DIR = new Key<>(STABLE_ORDER);

  /**
   * Files that are plopped into the final bundle at some arbitrary bundle path. Note that these are
   * not passed to Xcodegen, and these don't include information about where the file originated
   * from.
   */
  public static final Key<BundleableFile> BUNDLE_FILE = new Key<>(STABLE_ORDER);

  public static final Key<PathFragment> XCASSETS_DIR = new Key<>(STABLE_ORDER);
  public static final Key<String> SDK_DYLIB = new Key<>(STABLE_ORDER);
  public static final Key<SdkFramework> SDK_FRAMEWORK = new Key<>(STABLE_ORDER);
  public static final Key<Xcdatamodel> XCDATAMODEL = new Key<>(STABLE_ORDER);
  public static final Key<Flag> FLAG = new Key<>(STABLE_ORDER);
  public static final Key<Artifact> STORYBOARD_OUTPUT_ZIP = new Key<>(STABLE_ORDER);

  /**
   * Exec paths of {@code .framework} directories corresponding to frameworks to link. These cause
   * -F arguments (framework search paths) to be added to each compile action, and -framework (link
   * framework) arguments to be added to each link action.
   */
  public static final Key<PathFragment> FRAMEWORK_DIR = new Key<>(LINK_ORDER);

  /**
   * Files in {@code .framework} directories that should be included as inputs when compiling and
   * linking.
   */
  public static final Key<Artifact> FRAMEWORK_FILE = new Key<>(STABLE_ORDER);

  /**
   * Bundles which should be linked in as a nested bundle to the final application.
   */
  public static final Key<Bundling> NESTED_BUNDLE = new Key<>(STABLE_ORDER);

  /**
   * Flags that apply to a transitive build dependency tree. Each item in the enum corresponds to a
   * flag. If the item is included in the key {@link #FLAG}, then the flag is considered set.
   */
  public enum Flag {
    /**
     * Indicates that C++ (or Objective-C++) is used in any source file. This affects how the linker
     * is invoked.
    */
    USES_CPP;
  }

  private final ImmutableMap<Key<?>, NestedSet<?>> items;

  private ObjcProvider(ImmutableMap<Key<?>, NestedSet<?>> items) {
    this.items = Preconditions.checkNotNull(items);
  }

  /**
   * All artifacts, bundleable files, etc. of the type specified by {@code key}.
   */
  @SuppressWarnings("unchecked")
  public <E> NestedSet<E> get(Key<E> key) {
    Preconditions.checkNotNull(key);
    if (!items.containsKey(key)) {
      return NestedSetBuilder.emptySet(key.order);
    }
    return (NestedSet<E>) items.get(key);
  }

  /**
   * Indicates whether {@code flag} is set on this provider.
   */
  public boolean is(Flag flag) {
    return Iterables.contains(get(FLAG), flag);
  }

  // This exists only to support objc_filegroup - it is horribly hacky and should be removed when
  // objc_filegroup is.
  NestedSet<Artifact> allArtifactsForObjcFilegroup() {
    NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.<Artifact>stableOrder();
    for (Map.Entry<?, NestedSet<?>> entry : items.entrySet()) {
      for (Object o : entry.getValue()) {
        if (o instanceof Artifact) {
          artifacts.add((Artifact) o);
        }
      }
    }
    return artifacts.build();
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public static final class Builder {
    private final Map<Key<?>, NestedSetBuilder<?>> items = new HashMap<>();

    private void maybeAddEmptyBuilder(Key<?> key) {
      if (!items.containsKey(key)) {
        items.put(key, new NestedSetBuilder<>(key.order));
      }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void uncheckedAddAll(Key key, Iterable toAdd) {
      maybeAddEmptyBuilder(key);
      items.get(key).addAll(toAdd);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void uncheckedAddTransitive(Key key, NestedSet toAdd) {
      maybeAddEmptyBuilder(key);
      items.get(key).addTransitive(toAdd);
    }

    public Builder addTransitive(ObjcProvider provider) {
      for (Map.Entry<Key<?>, NestedSet<?>> typeEntry : provider.items.entrySet()) {
        uncheckedAddTransitive(typeEntry.getKey(), typeEntry.getValue());
      }
      return this;
    }

    public Builder addTransitive(Iterable<ObjcProvider> providers) {
      for (ObjcProvider provider : providers) {
        addTransitive(provider);
      }
      return this;
    }

    public <E> Builder addTransitive(Key<E> key, NestedSet<E> items) {
      uncheckedAddTransitive(key, items);
      return this;
    }

    public <E> Builder add(Key<E> key, E toAdd) {
      uncheckedAddAll(key, ImmutableList.of(toAdd));
      return this;
    }

    public <E> Builder addAll(Key<E> key, Iterable<? extends E> toAdd) {
      uncheckedAddAll(key, toAdd);
      return this;
    }

    public ObjcProvider build() {
      ImmutableMap.Builder<Key<?>, NestedSet<?>> result = new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : items.entrySet()) {
        result.put(typeEntry.getKey(), typeEntry.getValue().build());
      }
      return new ObjcProvider(result.build());
    }
  }
}
