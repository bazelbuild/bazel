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
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider.WithLegacySkylarkName;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink.CcLinkingContext;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * A provider that provides all compiling and linking information in the transitive closure of its
 * deps that are needed for building Objective-C rules.
 */
@Immutable
public final class ObjcProvider extends Info implements ObjcProviderApi<Artifact> {

  /** Skylark name for the ObjcProvider. */
  public static final String SKYLARK_NAME = "objc";

  /** Expected suffix for a framework-containing directory. */
  public static final String FRAMEWORK_SUFFIX = ".framework";

  /**
   * Represents one of the things this provider can provide transitively. Things are provided as
   * {@link NestedSet}s of type E.
   */
  @Immutable
  public static class Key<E> {
    private final Order order;
    private final String skylarkKeyName;
    private final Class<E> type;

    private Key(Order order, String skylarkKeyName, Class<E> type) {
      this.order = Preconditions.checkNotNull(order);
      this.skylarkKeyName = skylarkKeyName;
      this.type = type;
    }

    /**
     * Returns the name of the collection represented by this key in the Skylark provider.
     */
    public String getSkylarkKeyName() {
      return skylarkKeyName;
    }

    /**
     * Returns the type of nested set keyed in the ObjcProvider by this key.
     */
    public Class<E> getType() {
      return type;
    }
  }

  public static final Key<Artifact> LIBRARY = new Key<>(LINK_ORDER, "library", Artifact.class);

  public static final Key<Artifact> IMPORTED_LIBRARY =
      new Key<>(LINK_ORDER, "imported_library", Artifact.class);

  /**
   * J2ObjC JRE emulation libraries and their dependencies. Separate from LIBRARY because these
   * dependencies are specified further up the tree from where the dependency actually exists and
   * they must be forced to the end of the link order.
   */
  public static final Key<Artifact> JRE_LIBRARY =
      new Key<>(LINK_ORDER, "jre_library", Artifact.class);

  /**
   * Single-architecture linked binaries to be combined for the final multi-architecture binary.
   */
  public static final Key<Artifact> LINKED_BINARY =
      new Key<>(STABLE_ORDER, "linked_binary", Artifact.class);

  /** Combined-architecture binaries to include in the final bundle. */
  public static final Key<Artifact> MULTI_ARCH_LINKED_BINARIES =
      new Key<>(STABLE_ORDER, "combined_arch_linked_binary", Artifact.class);
  /** Combined-architecture dynamic libraries to include in the final bundle. */
  public static final Key<Artifact> MULTI_ARCH_DYNAMIC_LIBRARIES =
      new Key<>(STABLE_ORDER, "combined_arch_dynamic_library", Artifact.class);
  /** Combined-architecture archives to include in the final bundle. */
  public static final Key<Artifact> MULTI_ARCH_LINKED_ARCHIVES =
      new Key<>(STABLE_ORDER, "combined_arch_linked_archive", Artifact.class);

  /**
   * Indicates which libraries to load with {@code -force_load}. This is a subset of the union of
   * the {@link #LIBRARY} and {@link #IMPORTED_LIBRARY} sets.
   */
  public static final Key<Artifact> FORCE_LOAD_LIBRARY =
      new Key<>(LINK_ORDER, "force_load_library", Artifact.class);

  /**
   * Contains all header files. These may be either public or private headers.
   */
  public static final Key<Artifact> HEADER = new Key<>(STABLE_ORDER, "header", Artifact.class);

  /**
   * Contains all source files.
   */
  public static final Key<Artifact> SOURCE = new Key<>(STABLE_ORDER, "source", Artifact.class);

  /**
   * Include search paths specified with {@code -I} on the command line. Also known as header search
   * paths (and distinct from <em>user</em> header search paths).
   */
  public static final Key<PathFragment> INCLUDE =
      new Key<>(LINK_ORDER, "include", PathFragment.class);

  /**
   * Include search paths specified with {@code -iquote} on the command line. Also known as user
   * header search paths.
   */
  public static final Key<PathFragment> IQUOTE =
      new Key<>(LINK_ORDER, "iquote", PathFragment.class);

  /**
   * Include search paths specified with {@code -isystem} on the command line.
   */
  public static final Key<PathFragment> INCLUDE_SYSTEM =
      new Key<>(LINK_ORDER, "include_system", PathFragment.class);

  /**
   * Key for values in {@code defines} attributes. These are passed as {@code -D} flags to all
   * invocations of the compiler for this target and all depending targets.
   */
  public static final Key<String> DEFINE = new Key<>(STABLE_ORDER, "define", String.class);

  public static final Key<Artifact> ASSET_CATALOG =
      new Key<>(STABLE_ORDER, "asset_catalog", Artifact.class);

  /**
   * Files that are plopped into the final bundle at some arbitrary bundle path. Do not include
   * information about where the file originated from.
   */
  public static final Key<BundleableFile> BUNDLE_FILE =
      new Key<>(STABLE_ORDER, "bundle_file", BundleableFile.class);

  public static final Key<PathFragment> XCASSETS_DIR =
      new Key<>(STABLE_ORDER, "xcassets_dir", PathFragment.class);
  public static final Key<String> SDK_DYLIB = new Key<>(STABLE_ORDER, "sdk_dylib", String.class);
  public static final Key<SdkFramework> SDK_FRAMEWORK =
      new Key<>(STABLE_ORDER, "sdk_framework", SdkFramework.class);
  public static final Key<SdkFramework> WEAK_SDK_FRAMEWORK =
      new Key<>(STABLE_ORDER, "weak_sdk_framework", SdkFramework.class);
  public static final Key<Artifact> XCDATAMODEL =
      new Key<>(STABLE_ORDER, "xcdatamodel", Artifact.class);
  public static final Key<Flag> FLAG = new Key<>(STABLE_ORDER, "flag", Flag.class);

  /**
   * Clang umbrella header. Public headers are #included in umbrella headers to be compatible with
   * J2ObjC segmented headers.
   */
  public static final Key<Artifact> UMBRELLA_HEADER =
      new Key<>(STABLE_ORDER, "umbrella_header", Artifact.class);

  /**
   * Clang module maps, used to enforce proper use of private header files.
   */
  public static final Key<Artifact> MODULE_MAP =
      new Key<>(STABLE_ORDER, "module_map", Artifact.class);

  /**
   * Information about this provider's module map, in the form of a {@link CppModuleMap}. This
   * is intransitive, and can be used to get just the target's module map to pass to clang or to
   * get the module maps for direct but not transitive dependencies. You should only add module maps
   * for this key using {@link Builder#addWithoutPropagating}.
   */
  public static final Key<CppModuleMap> TOP_LEVEL_MODULE_MAP =
      new Key<>(STABLE_ORDER, "top_level_module_map", CppModuleMap.class);

  /**
   * Merge zips to include in the bundle. The entries of these zip files are included in the final
   * bundle with the same path. The entries in the merge zips should not include the bundle root
   * path (e.g. {@code Foo.app}).
   */
  public static final Key<Artifact> MERGE_ZIP =
      new Key<>(STABLE_ORDER, "merge_zip", Artifact.class);

  /**
   * Merge zips to include in the ipa and outside the bundle root.
   *
   * e.g. For a bundle Test.ipa, unzipped content will be in:
   *    Test.ipa/<unzipped>
   *    Test.ipa/Payload
   *    Test.ipa/Payload/Test.app
   */
  public static final Key<Artifact> ROOT_MERGE_ZIP =
      new Key<>(STABLE_ORDER, "root_merge_zip", Artifact.class);

  /**
   * Exec paths of {@code .framework} directories corresponding to frameworks to include in search
   * paths, but not to link.  These cause -F arguments (framework search paths) to be added to
   * each compile action, but do not cause -framework (link framework) arguments to be added to
   * link actions.
   */
  public static final Key<PathFragment> FRAMEWORK_SEARCH_PATH_ONLY =
      new Key<>(LINK_ORDER, "framework_search_paths", PathFragment.class);

  /** The static library files of user-specified static frameworks. */
  public static final Key<Artifact> STATIC_FRAMEWORK_FILE =
      new Key<>(STABLE_ORDER, "static_framework_file", Artifact.class);

  /**
   * Exec paths of {@code .framework} directories corresponding to dynamic frameworks to link. These
   * cause -F arguments (framework search paths) to be added to each compile action, and
   * -framework (link framework) arguments to be added to each link action. These differ from
   * static frameworks in that they are not statically linked into the binary.
   */
  public static final Key<PathFragment> DYNAMIC_FRAMEWORK_DIR =
      new Key<>(LINK_ORDER, "dynamic_framework_dir", PathFragment.class);

  /** The dynamic library files of user-specified dynamic frameworks. */
  public static final Key<Artifact> DYNAMIC_FRAMEWORK_FILE =
      new Key<>(STABLE_ORDER, "dynamic_framework_file", Artifact.class);

  /**
   * Bundles which should be linked in as a nested bundle to the final application.
   */
  public static final Key<Bundling> NESTED_BUNDLE =
      new Key<>(STABLE_ORDER, "nested_bundle", Bundling.class);

  /**
   * Debug artifacts that should be exported by the top-level target.
   */
  public static final Key<Artifact> EXPORTED_DEBUG_ARTIFACTS =
      new Key<>(STABLE_ORDER, "exported_debug_artifacts", Artifact.class);

  /**
   * Single-architecture link map for a binary.
   */
  public static final Key<Artifact> LINKMAP_FILE =
      new Key<>(STABLE_ORDER, "linkmap_file", Artifact.class);

  /**
   * Artifacts for storyboard sources.
   */
  public static final Key<Artifact> STORYBOARD =
      new Key<>(STABLE_ORDER, "storyboard", Artifact.class);

  /**
   * Artifacts for .xib file sources.
   */
  public static final Key<Artifact> XIB = new Key<>(STABLE_ORDER, "xib", Artifact.class);

  /**
   * Artifacts for strings source files.
   */
  public static final Key<Artifact> STRINGS = new Key<>(STABLE_ORDER, "strings", Artifact.class);

  /** Linking information from cc dependencies. */
  public static final Key<LibraryToLink> CC_LIBRARY =
      new Key<>(LINK_ORDER, "cc_library", LibraryToLink.class);

  /**
   * Linking options from dependencies.
   */
  public static final Key<String> LINKOPT = new Key<>(LINK_ORDER, "linkopt", String.class);

  /**
   * Link time artifacts from dependencies. These do not fall into any other category such as
   * libraries or archives, rather provide a way to add arbitrary data (e.g. Swift AST files)
   * to the linker. The rule that adds these is also responsible to add the necessary linker flags
   * in {@link #LINKOPT}.
   */
  public static final Key<Artifact> LINK_INPUTS =
      new Key<>(LINK_ORDER, "link_inputs", Artifact.class);

  /** Static libraries that are built from J2ObjC-translated Java code. */
  public static final Key<Artifact> J2OBJC_LIBRARY =
      new Key<>(LINK_ORDER, "j2objc_library", Artifact.class);

  /**
   * Flags that apply to a transitive build dependency tree. Each item in the enum corresponds to a
   * flag. If the item is included in the key {@link #FLAG}, then the flag is considered set.
   */
  public enum Flag {
    /**
     * Indicates that C++ (or Objective-C++) is used in any source file. This affects how the linker
     * is invoked.
     */
    USES_CPP,

    /**
     * Indicates that Objective-C (or Objective-C++) is used in any source file. This affects how
     * the linker is invoked.
     */
    USES_OBJC,

    /** Indicates that Swift dependencies are present. This affects bundling actions. */
    USES_SWIFT,

    /**
     * Indicates that a watchOS 1 extension is present in the bundle. (There can only be one
     * extension for any given watchOS version in a given bundle).
     */
    HAS_WATCH1_EXTENSION,

    /**
     * Indicates that a watchOS 2 extension is present in the bundle. (There can only be one
     * extension for any given watchOS version in a given bundle).
     */
    HAS_WATCH2_EXTENSION,
  }

  private final StarlarkSemantics semantics;
  private final ImmutableMap<Key<?>, NestedSet<?>> items;

  // Items which should not be propagated to dependents.
  private final ImmutableMap<Key<?>, NestedSet<?>> nonPropagatedItems;
  /** All keys in ObjcProvider that will be passed in the corresponding Skylark provider. */
  static final ImmutableList<Key<?>> KEYS_FOR_SKYLARK =
      ImmutableList.<Key<?>>of(
          ASSET_CATALOG,
          BUNDLE_FILE,
          DEFINE,
          DYNAMIC_FRAMEWORK_DIR,
          DYNAMIC_FRAMEWORK_FILE,
          EXPORTED_DEBUG_ARTIFACTS,
          FRAMEWORK_SEARCH_PATH_ONLY,
          FORCE_LOAD_LIBRARY,
          HEADER,
          IMPORTED_LIBRARY,
          INCLUDE,
          INCLUDE_SYSTEM,
          IQUOTE,
          J2OBJC_LIBRARY,
          JRE_LIBRARY,
          LIBRARY,
          LINK_INPUTS,
          LINKED_BINARY,
          LINKMAP_FILE,
          LINKOPT,
          MERGE_ZIP,
          MODULE_MAP,
          MULTI_ARCH_DYNAMIC_LIBRARIES,
          MULTI_ARCH_LINKED_ARCHIVES,
          MULTI_ARCH_LINKED_BINARIES,
          ROOT_MERGE_ZIP,
          SDK_DYLIB,
          SDK_FRAMEWORK,
          SOURCE,
          STATIC_FRAMEWORK_FILE,
          STORYBOARD,
          STRINGS,
          UMBRELLA_HEADER,
          WEAK_SDK_FRAMEWORK,
          XCASSETS_DIR,
          XCDATAMODEL,
          XIB);

  /** Deprecated keys in ObjcProvider pertaining to resource files. */
  static final ImmutableList<Key<?>> DEPRECATED_RESOURCE_KEYS =
      ImmutableList.<Key<?>>of(
          ASSET_CATALOG,
          BUNDLE_FILE,
          // TODO(kaipi): Add this back once we have migrated usages of merge_zip from custom rules.
          // MERGE_ZIP,
          ROOT_MERGE_ZIP,
          STORYBOARD,
          STRINGS,
          XCASSETS_DIR,
          XCDATAMODEL,
          XIB);

  @Override
  public NestedSet<Artifact> assetCatalog() {
    return get(ASSET_CATALOG);
  }

  @Override
  public SkylarkNestedSet bundleFile() {
    return (SkylarkNestedSet) ObjcProviderSkylarkConverters.convertToSkylark(
        BUNDLE_FILE, get(BUNDLE_FILE));
  }

  @Override
  public NestedSet<String> define() {
    return get(DEFINE);
  }

  @Override
  public SkylarkNestedSet dynamicFrameworkDir() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(get(DYNAMIC_FRAMEWORK_DIR));
  }

  @Override
  public NestedSet<Artifact> dynamicFrameworkFile() {
    return get(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public NestedSet<Artifact> exportedDebugArtifacts() {
    return get(EXPORTED_DEBUG_ARTIFACTS);
  }

  @Override
  public SkylarkNestedSet frameworkSearchPathOnly() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(
        get(FRAMEWORK_SEARCH_PATH_ONLY));
  }

  @Override
  public NestedSet<Artifact> forceLoadLibrary() {
    return get(FORCE_LOAD_LIBRARY);
  }

  @Override
  public NestedSet<Artifact> header() {
    return get(HEADER);
  }

  @Override
  public NestedSet<Artifact> importedLibrary() {
    return get(IMPORTED_LIBRARY);
  }

  @Override
  public SkylarkNestedSet include() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(get(INCLUDE));
  }

  @Override
  public SkylarkNestedSet includeSystem() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(get(INCLUDE_SYSTEM));
  }

  @Override
  public SkylarkNestedSet iquote() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(get(IQUOTE));
  }

  @Override
  public NestedSet<Artifact> j2objcLibrary() {
    return get(J2OBJC_LIBRARY);
  }

  @Override
  public NestedSet<Artifact> jreLibrary() {
    return get(JRE_LIBRARY);
  }

  @Override
  public NestedSet<Artifact> library() {
    return get(LIBRARY);
  }

  @Override
  public NestedSet<Artifact> linkInputs() {
    return get(LINK_INPUTS);
  }

  @Override
  public NestedSet<Artifact> linkedBinary() {
    return get(LINKED_BINARY);
  }

  @Override
  public NestedSet<Artifact> linkmapFile() {
    return get(LINKMAP_FILE);
  }

  @Override
  public NestedSet<String> linkopt() {
    return get(LINKOPT);
  }

  @Override
  public NestedSet<Artifact> mergeZip() {
    return get(MERGE_ZIP);
  }

  @Override
  public NestedSet<Artifact> moduleMap() {
    return get(MODULE_MAP);
  }

  @Override
  public NestedSet<Artifact> multiArchDynamicLibraries() {
    return get(MULTI_ARCH_DYNAMIC_LIBRARIES);
  }

  @Override
  public NestedSet<Artifact> multiArchLinkedArchives() {
    return get(MULTI_ARCH_LINKED_ARCHIVES);
  }

  @Override
  public NestedSet<Artifact> multiArchLinkedBinaries() {
    return get(MULTI_ARCH_LINKED_BINARIES);
  }

  @Override
  public NestedSet<Artifact> rootMergeZip() {
    return get(ROOT_MERGE_ZIP);
  }

  @Override
  public NestedSet<String> sdkDylib() {
    return get(SDK_DYLIB);
  }

  @Override
  public SkylarkNestedSet sdkFramework() {
    return (SkylarkNestedSet) ObjcProviderSkylarkConverters.convertToSkylark(SDK_FRAMEWORK,
        get(SDK_FRAMEWORK));
  }

  @Override
  public NestedSet<Artifact> source() {
    return get(SOURCE);
  }

  @Override
  public NestedSet<Artifact> staticFrameworkFile() {
    return get(STATIC_FRAMEWORK_FILE);
  }

  @Override
  public NestedSet<Artifact> storyboard() {
    return get(STORYBOARD);
  }

  @Override
  public NestedSet<Artifact> strings() {
    return get(STRINGS);
  }

  @Override
  public NestedSet<Artifact> umbrellaHeader() {
    return get(UMBRELLA_HEADER);
  }

  @Override
  public SkylarkNestedSet weakSdkFramework() {
    return (SkylarkNestedSet) ObjcProviderSkylarkConverters.convertToSkylark(WEAK_SDK_FRAMEWORK,
        get(WEAK_SDK_FRAMEWORK));
  }

  @Override
  public SkylarkNestedSet xcassetsDir() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(get(XCASSETS_DIR));
  }

  @Override
  public NestedSet<Artifact> xcdatamodel() {
    return get(XCDATAMODEL);
  }

  @Override
  public NestedSet<Artifact> xib() {
    return get(XIB);
  }

  /**
   * All keys in ObjcProvider that are explicitly not exposed to skylark. This is used for
   * testing and verification purposes to ensure that a conscious decision is made for all keys;
   * by default, keys should be exposed to skylark: a comment outlining why a key is omitted
   * from skylark should follow each such case.
   **/
  @VisibleForTesting
  static final ImmutableList<Key<?>> KEYS_NOT_IN_SKYLARK = ImmutableList.<Key<?>>of(
      // LibraryToLink not exposed to skylark.
      CC_LIBRARY,
      // Flag enum is not exposed to skylark.
      FLAG,
      // Bundle not exposed to skylark.
      NESTED_BUNDLE,
      // CppModuleMap is not exposed to skylark.
      TOP_LEVEL_MODULE_MAP);

  /**
   * Set of {@link ObjcProvider} whose values are not subtracted via {@link #subtractSubtrees}.
   *
   * <p>Only keys which are unrelated to statically-linked library dependencies should be listed.
   * For example, LIBRARY is a subtractable key because it contains objects of individual
   * objc_library dependencies, but keys pertaining to resources are non-subtractable keys, because
   * the top level binary will need these resources whether or not the library is statically or
   * dynamically linked.
   */
  private static final ImmutableSet<Key<?>> NON_SUBTRACTABLE_KEYS =
      ImmutableSet.<Key<?>>of(
          DEFINE,
          DYNAMIC_FRAMEWORK_DIR,
          DYNAMIC_FRAMEWORK_FILE,
          FLAG,
          MERGE_ZIP,
          ROOT_MERGE_ZIP,
          FRAMEWORK_SEARCH_PATH_ONLY,
          HEADER,
          INCLUDE,
          INCLUDE_SYSTEM,
          IQUOTE,
          LINKOPT,
          LINK_INPUTS,
          SDK_DYLIB,
          SDK_FRAMEWORK,
          WEAK_SDK_FRAMEWORK);

  /**
   * Returns the skylark key for the given string, or null if no such key exists or is available
   * to Skylark.
   */
  static Key<?> getSkylarkKeyForString(String keyName) {
    for (Key<?> candidateKey : KEYS_FOR_SKYLARK) {
      if (candidateKey.getSkylarkKeyName().equals(keyName)) {
        return candidateKey;
      }
    }
    return null;
  }

  static boolean isDeprecatedResourceKey(Key<?> key) {
    return DEPRECATED_RESOURCE_KEYS.contains(key);
  }

  // Items which should be passed to strictly direct dependers, but not transitive dependers.
  private final ImmutableMap<Key<?>, NestedSet<?>> strictDependencyItems;

  /** Skylark constructor and identifier for ObjcProvider. */
  public static final BuiltinProvider<ObjcProvider> SKYLARK_CONSTRUCTOR = new Constructor();

  private ObjcProvider(
      StarlarkSemantics semantics,
      ImmutableMap<Key<?>, NestedSet<?>> items,
      ImmutableMap<Key<?>, NestedSet<?>> nonPropagatedItems,
      ImmutableMap<Key<?>, NestedSet<?>> strictDependencyItems) {
    super(SKYLARK_CONSTRUCTOR, Location.BUILTIN);
    this.semantics = semantics;
    this.items = Preconditions.checkNotNull(items);
    this.nonPropagatedItems = Preconditions.checkNotNull(nonPropagatedItems);
    this.strictDependencyItems = Preconditions.checkNotNull(strictDependencyItems);
  }

  /**
   * All artifacts, bundleable files, etc. of the type specified by {@code key}.
   */
  @SuppressWarnings("unchecked")
  public <E> NestedSet<E> get(Key<E> key) {
    Preconditions.checkNotNull(key);
    NestedSetBuilder<E> builder = new NestedSetBuilder<>(key.order);
    if (strictDependencyItems.containsKey(key)) {
      builder.addTransitive((NestedSet<E>) strictDependencyItems.get(key));
    }
    if (nonPropagatedItems.containsKey(key)) {
      builder.addTransitive((NestedSet<E>) nonPropagatedItems.get(key));
    }
    if (items.containsKey(key)) {
      builder.addTransitive((NestedSet<E>) items.get(key));
    }
    return builder.build();
  }

  /**
   * Returns all keys that have at least one value in this provider (values may be propagable,
   * non-propagable, or strict).
   */
  private Iterable<Key<?>> getValuedKeys() {
    return ImmutableSet.<Key<?>>builder()
        .addAll(strictDependencyItems.keySet())
        .addAll(nonPropagatedItems.keySet())
        .addAll(items.keySet())
        .build();
  }

  /**
   * All artifacts, bundleable files, etc, that should be propagated to transitive dependers, of
   * the type specified by {@code key}.
   */
  @SuppressWarnings("unchecked")
  private <E> NestedSet<E> getPropagable(Key<E> key) {
    Preconditions.checkNotNull(key);
    NestedSetBuilder<E> builder = new NestedSetBuilder<>(key.order);
    if (items.containsKey(key)) {
      builder.addTransitive((NestedSet<E>) items.get(key));
    }
    return builder.build();
  }

  /**
   * Indicates whether {@code flag} is set on this provider.
   */
  public boolean is(Flag flag) {
    return Iterables.contains(get(FLAG), flag);
  }

  /**
   * Indicates whether this provider has any asset catalogs. This is true whenever some target in
   * its transitive dependency tree specifies a non-empty {@code asset_catalogs} attribute.
   */
  public boolean hasAssetCatalogs() {
    return !get(XCASSETS_DIR).isEmpty();
  }

  /** Returns the list of .a files required for linking that arise from objc libraries. */
  ImmutableList<Artifact> getObjcLibraries() {
    // JRE libraries must be ordered after all regular objc libraries.
    NestedSet<Artifact> jreLibs = get(JRE_LIBRARY);
    return ImmutableList.<Artifact>builder()
        .addAll(Iterables.filter(
            get(LIBRARY), Predicates.not(Predicates.in(jreLibs.toSet()))))
        .addAll(jreLibs)
        .build();
  }

  /** Returns the list of .a files required for linking that arise from cc libraries. */
  List<Artifact> getCcLibraries() {
    NestedSetBuilder<LibraryToLink> libraryToLinkListBuilder = NestedSetBuilder.linkOrder();
    for (LibraryToLink libraryToLink : get(CC_LIBRARY)) {
      libraryToLinkListBuilder.add(libraryToLink);
    }
    CcLinkingContext ccLinkingContext =
        CcLinkingContext.builder().addLibraries(libraryToLinkListBuilder.build()).build();
    return ccLinkingContext.getStaticModeParamsForExecutableLibraries();
  }

  /**
   * Subtracts dependency subtrees from this provider and returns the result (subtraction does not
   * mutate this provider). Note that not all provider keys are subtracted; generally only keys
   * which correspond with compiled libraries will be subtracted.
   *
   * <p>This is an expensive operation, as it requires flattening of all nested sets contained in
   * each provider.
   *
   * @param avoidObjcProviders objc providers which contain the dependency subtrees to subtract
   * @param avoidCcProviders cc providers which contain the dependency subtrees to subtract
   */
  // TODO(b/65156211): Investigate subtraction generalized to NestedSet.
  @SuppressWarnings("unchecked") // Due to depending on Key types, when the keys map erases type.
  public ObjcProvider subtractSubtrees(
      Iterable<ObjcProvider> avoidObjcProviders, Iterable<CcLinkingContext> avoidCcProviders) {
    // LIBRARY and CC_LIBRARY need to be special cased for objc-cc interop.
    // A library which is a dependency of a cc_library may be present in all or any of
    // three possible locations (and may be duplicated!):
    // 1. ObjcProvider.LIBRARY
    // 2. ObjcProvider.CC_LIBRARY
    // 3. CcLinkingContext->LibraryToLink->getArtifact()
    // TODO(cpeyser): Clean up objc-cc interop.
    HashSet<PathFragment> avoidLibrariesSet = new HashSet<>();
    for (CcLinkingContext ccLinkingContext : avoidCcProviders) {
      List<Artifact> libraries = ccLinkingContext.getStaticModeParamsForExecutableLibraries();
      for (Artifact library : libraries) {
        avoidLibrariesSet.add(library.getRunfilesPath());
      }
    }
    for (ObjcProvider avoidProvider : avoidObjcProviders) {
      for (Artifact ccLibrary : avoidProvider.getCcLibraries()) {
        avoidLibrariesSet.add(ccLibrary.getRunfilesPath());
      }
      for (Artifact libraryToAvoid : avoidProvider.getPropagable(LIBRARY)) {
        avoidLibrariesSet.add(libraryToAvoid.getRunfilesPath());
      }
    }
    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder(semantics);
    for (Key<?> key : getValuedKeys()) {
      if (key == CC_LIBRARY) {
        addTransitiveAndFilter(objcProviderBuilder, CC_LIBRARY,
            ccLibraryNotYetLinked(avoidLibrariesSet));
      } else if (key == LIBRARY) {
        addTransitiveAndFilter(objcProviderBuilder, LIBRARY, notContainedIn(avoidLibrariesSet));
      } else if (NON_SUBTRACTABLE_KEYS.contains(key)) {
        addTransitiveAndAvoid(objcProviderBuilder, key, ImmutableList.<ObjcProvider>of());
      } else if (key.getType() == Artifact.class) {
        addTransitiveAndAvoidArtifacts(objcProviderBuilder, ((Key<Artifact>) key),
            avoidObjcProviders);
      } else {
        addTransitiveAndAvoid(objcProviderBuilder, key, avoidObjcProviders);
      }
    }
    return objcProviderBuilder.build();
  }

  /**
   * Returns a predicate which returns true for a given artifact if the artifact's runfiles path
   * is not contained in the given set.
   *
   * @param runfilesPaths if a given artifact has runfiles path present in this set, the predicate
   *     will return false
   */
  private static Predicate<Artifact> notContainedIn(
      final HashSet<PathFragment> runfilesPaths) {
    return libraryToLink -> !runfilesPaths.contains(libraryToLink.getRunfilesPath());
  }

  /**
   * Returns a predicate which returns true for a given {@link LibraryToLink} if the library's
   * runfiles path is not contained in the given set.
   *
   * @param runfilesPaths if a given library has runfiles path present in this set, the predicate
   *     will return false
   */
  private static Predicate<LibraryToLink> ccLibraryNotYetLinked(
      final HashSet<PathFragment> runfilesPaths) {
    return libraryToLink -> !checkIfLibraryIsInPaths(libraryToLink, runfilesPaths);
  }

  private static boolean checkIfLibraryIsInPaths(
      LibraryToLink libraryToLink, HashSet<PathFragment> runfilesPaths) {
    ImmutableList.Builder<PathFragment> libraryRunfilesPaths = ImmutableList.builder();
    if (libraryToLink.getStaticLibrary() != null) {
      libraryRunfilesPaths.add(libraryToLink.getStaticLibrary().getRunfilesPath());
    }
    if (libraryToLink.getPicStaticLibrary() != null) {
      libraryRunfilesPaths.add(libraryToLink.getPicStaticLibrary().getRunfilesPath());
    }
    if (libraryToLink.getDynamicLibrary() != null) {
      libraryRunfilesPaths.add(libraryToLink.getDynamicLibrary().getRunfilesPath());
    }
    if (libraryToLink.getResolvedSymlinkDynamicLibrary() != null) {
      libraryRunfilesPaths.add(libraryToLink.getResolvedSymlinkDynamicLibrary().getRunfilesPath());
    }
    if (libraryToLink.getInterfaceLibrary() != null) {
      libraryRunfilesPaths.add(libraryToLink.getInterfaceLibrary().getRunfilesPath());
    }
    if (libraryToLink.getResolvedSymlinkInterfaceLibrary() != null) {
      libraryRunfilesPaths.add(
          libraryToLink.getResolvedSymlinkInterfaceLibrary().getRunfilesPath());
    }

    return !Collections.disjoint(libraryRunfilesPaths.build(), runfilesPaths);
  }

  @SuppressWarnings("unchecked")
  private <T> void addTransitiveAndFilter(ObjcProvider.Builder objcProviderBuilder, Key<T> key,
      Predicate<T> filterPredicate) {
    NestedSet<T> propagableItems = (NestedSet<T>) items.get(key);
    NestedSet<T> nonPropagableItems = (NestedSet<T>) nonPropagatedItems.get(key);
    NestedSet<T> strictItems = (NestedSet<T>) strictDependencyItems.get(key);

    if (propagableItems != null) {
      objcProviderBuilder.addAll(key,
          Iterables.filter(propagableItems.toList(), filterPredicate));
    }
    if (nonPropagableItems != null) {
      objcProviderBuilder.addAllNonPropagable(key,
          Iterables.filter(nonPropagableItems.toList(), filterPredicate));
    }
    if (strictItems != null) {
      objcProviderBuilder.addAllForDirectDependents(key,
          Iterables.filter(strictItems.toList(), filterPredicate));
    }
  }

  private void addTransitiveAndAvoidArtifacts(ObjcProvider.Builder objcProviderBuilder,
      Key<Artifact> key, Iterable<ObjcProvider> avoidProviders) {
    // Artifacts to avoid may be in a different configuration and thus a different
    // root directory, hence only the path fragment after the root directory is compared.
    HashSet<PathFragment> avoidPathsSet = new HashSet<>();
    for (ObjcProvider avoidProvider : avoidProviders) {
      for (Artifact artifact : avoidProvider.getPropagable(key)) {
        avoidPathsSet.add(artifact.getRunfilesPath());
      }
    }
    addTransitiveAndFilter(objcProviderBuilder, key, notContainedIn(avoidPathsSet));
  }

  @SuppressWarnings("unchecked")
  private <T> void addTransitiveAndAvoid(ObjcProvider.Builder objcProviderBuilder, Key<T> key,
      Iterable<ObjcProvider> avoidProviders) {
    HashSet<T> avoidItemsSet = new HashSet<T>();
    for (ObjcProvider avoidProvider : avoidProviders) {
      avoidItemsSet.addAll(avoidProvider.getPropagable(key).toList());
    }
    addTransitiveAndFilter(objcProviderBuilder, key, Predicates.not(Predicates.in(avoidItemsSet)));
  }

  /**
   * Returns all unique static framework directories (directories ending in '.framework') for all
   * static framework files in this provider.
   */
  public Iterable<PathFragment> getStaticFrameworkDirs() {
    return ObjcCommon.uniqueContainers(get(STATIC_FRAMEWORK_FILE),
        ObjcCommon.FRAMEWORK_CONTAINER_TYPE);
  }

  /**
   * Returns all unique static framework directories (directories ending in '.framework') for all
   * static framework files in this provider.
   */
  @Override
  public SkylarkNestedSet getStaticFrameworkDirsForSkylark() {
    return ObjcProviderSkylarkConverters.convertPathFragmentsToSkylark(getStaticFrameworkDirs());
  }

  /**
   * Check whether that a path fragment is a framework directory (i.e. ends in FRAMEWORK_SUFFIX).
   */
  private static void checkIsFrameworkDirectory(PathFragment dir) {
    Preconditions.checkState(dir.getBaseName().endsWith(FRAMEWORK_SUFFIX));
  }

  /** The input path must be of the form <path>/<name>.FRAMEWORK_SUFFIX. Return the names. */
  private static String getFrameworkName(PathFragment frameworkPath) {
    String segment = frameworkPath.getBaseName();
    return segment.substring(0, segment.length() - FRAMEWORK_SUFFIX.length());
  }

  /** The input path must be of the form <path>/<name>.FRAMEWORK_SUFFIX. Return the paths. */
  private static String getFrameworkPath(PathFragment frameworkPath) {
    return frameworkPath.getParentDirectory().getSafePathString();
  }

  /**
   * @param key either DYNAMIC_FRAMEWORK_FILE or STATIC_FRAMEWORK_FILE. Return the corresponding
   *     framework names, i.e. for a given a file <path>/<name>.FRAMEWORK_SUFFIX/<name>, return
   *     <name>.
   */
  private NestedSet<String> getFrameworkNames(Key<Artifact> key) {
    NestedSetBuilder<String> names = new NestedSetBuilder<>(key.order);
    for (Artifact file : get(key)) {
      PathFragment frameworkDir = file.getExecPath().getParentDirectory();
      checkIsFrameworkDirectory(frameworkDir);
      names.add(getFrameworkName(frameworkDir));
    }
    return names.build();
  }

  /**
   * @param key either DYNAMIC_FRAMEWORK_FILE or STATIC_FRAMEWORK_FILE. Return the corresponding
   *     framework paths, i.e. for a given a file <path>/<name>.FRAMEWORK_SUFFIX/<name>, return
   *     <path>.
   */
  private NestedSet<String> getFrameworkPaths(Key<Artifact> key) {
    NestedSetBuilder<String> paths = new NestedSetBuilder<>(key.order);
    for (Artifact file : get(key)) {
      PathFragment frameworkDir = file.getExecPath().getParentDirectory();
      checkIsFrameworkDirectory(frameworkDir);
      paths.add(getFrameworkPath(frameworkDir));
    }
    return paths.build();
  }

  @Override
  public NestedSet<String> dynamicFrameworkNames() {
    return getFrameworkNames(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public NestedSet<String> dynamicFrameworkPaths() {
    return getFrameworkPaths(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public NestedSet<String> staticFrameworkNames() {
    return getFrameworkNames(STATIC_FRAMEWORK_FILE);
  }

  @Override
  public NestedSet<String> staticFrameworkPaths() {
    return getFrameworkPaths(STATIC_FRAMEWORK_FILE);
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public static final class Builder {
    private final StarlarkSemantics starlarkSemantics;
    private final Map<Key<?>, NestedSetBuilder<?>> items = new HashMap<>();
    private final Map<Key<?>, NestedSetBuilder<?>> nonPropagatedItems = new HashMap<>();
    private final Map<Key<?>, NestedSetBuilder<?>> strictDependencyItems = new HashMap<>();

    public Builder(StarlarkSemantics semantics) {
      this.starlarkSemantics = semantics;
    }

    private static void maybeAddEmptyBuilder(Map<Key<?>, NestedSetBuilder<?>> set, Key<?> key) {
      set.computeIfAbsent(key, k -> new NestedSetBuilder<>(k.order));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void uncheckedAddAll(Key key, Iterable toAdd, Map<Key<?>, NestedSetBuilder<?>> set) {
      maybeAddEmptyBuilder(set, key);
      set.get(key).addAll(toAdd);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void uncheckedAddTransitive(Key key, NestedSet toAdd,
        Map<Key<?>, NestedSetBuilder<?>> set) {
      maybeAddEmptyBuilder(set, key);
      set.get(key).addTransitive(toAdd);
    }

    /**
     * Add all elements from providers, and propagate them to any (transitive) dependers on this
     * ObjcProvider.
     */
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
    public Builder addTransitiveAndPropagate(ObjcProvider provider) {
      for (Map.Entry<Key<?>, NestedSet<?>> typeEntry : provider.items.entrySet()) {
        uncheckedAddTransitive(typeEntry.getKey(), typeEntry.getValue(), this.items);
      }
      for (Map.Entry<Key<?>, NestedSet<?>> typeEntry : provider.strictDependencyItems.entrySet()) {
        uncheckedAddTransitive(typeEntry.getKey(), typeEntry.getValue(), this.nonPropagatedItems);
      }
      return this;
    }

    /**
     * Add all elements from a single key of the given provider, and propagate them to any
     * (transitive) dependers on this ObjcProvider.
     */
    public Builder addTransitiveAndPropagate(Key key, ObjcProvider provider) {
      if (provider.items.containsKey(key)) {
        uncheckedAddTransitive(key, provider.items.get(key), this.items);
      }
      if (provider.strictDependencyItems.containsKey(key)) {
        uncheckedAddTransitive(
            key, provider.strictDependencyItems.get(key), this.nonPropagatedItems);
      }
      return this;
    }

    /**
     * Adds elements in items, and propagate them to any (transitive) dependers on this
     * ObjcProvider.
     */
    public <E> Builder addTransitiveAndPropagate(Key<E> key, NestedSet<E> items) {
      uncheckedAddTransitive(key, items, this.items);
      return this;
    }

    /**
     * Add all keys and values from the given provider, but propagate any normally-propagated items
     * only to direct dependers of this ObjcProvider.
     */
    public Builder addAsDirectDeps(ObjcProvider provider) {
      for (Map.Entry<Key<?>, NestedSet<?>> typeEntry : provider.items.entrySet()) {
        uncheckedAddTransitive(
            typeEntry.getKey(), typeEntry.getValue(), this.strictDependencyItems);
      }
      for (Map.Entry<Key<?>, NestedSet<?>> typeEntry : provider.strictDependencyItems.entrySet()) {
        uncheckedAddTransitive(typeEntry.getKey(), typeEntry.getValue(), this.nonPropagatedItems);
      }
      return this;
    }

    /**
     * Add element, and propagate it to any (transitive) dependers on this ObjcProvider.
     */
    public <E> Builder add(Key<E> key, E toAdd) {
      uncheckedAddAll(key, ImmutableList.of(toAdd), this.items);
      return this;
    }

    /**
     * Add elements in toAdd, and propagate them to any (transitive) dependers on this ObjcProvider.
     */
    public <E> Builder addAll(Key<E> key, Iterable<? extends E> toAdd) {
      uncheckedAddAll(key, toAdd, this.items);
      return this;
    }

    /**
     * Add elements in toAdd, and do not propagate to dependents of this provider.
     */
    public <E> Builder addAllNonPropagable(Key<E> key, Iterable<? extends E> toAdd) {
      uncheckedAddAll(key, toAdd, this.nonPropagatedItems);
      return this;
    }

    /**
     * Add element toAdd, and propagate it only to direct dependents of this provider.
     */
    public <E> Builder addForDirectDependents(Key<E> key, E toAdd) {
      uncheckedAddAll(key, ImmutableList.of(toAdd), this.strictDependencyItems);
      return this;
    }

    /**
     * Add elements in toAdd, and propagate them only to direct dependents of this provider.
     */
    public <E> Builder addAllForDirectDependents(Key<E> key, Iterable<? extends E> toAdd) {
      uncheckedAddAll(key, toAdd, this.strictDependencyItems);
      return this;
    }

    /**
     * Add elements in toAdd with the given key from skylark.  An error is thrown if toAdd is not
     * an appropriate SkylarkNestedSet.
     */
    void addElementsFromSkylark(Key<?> key, Object toAdd) {
      uncheckedAddAll(key, ObjcProviderSkylarkConverters.convertToJava(key, toAdd), this.items);
    }

    /**
     * Adds the given providers from skylark. An error is thrown if toAdd is not an iterable of
     * ObjcProvider instances.
     */
    @SuppressWarnings("unchecked")
    void addProvidersFromSkylark(Object toAdd) {
      if (!(toAdd instanceof Iterable)) {
        throw new IllegalArgumentException(
            String.format(
                AppleSkylarkCommon.BAD_PROVIDERS_ITER_ERROR, EvalUtils.getDataTypeName(toAdd)));
      } else {
        Iterable<Object> toAddIterable = (Iterable<Object>) toAdd;
        for (Object toAddObject : toAddIterable) {
          if (!(toAddObject instanceof ObjcProvider)) {
            throw new IllegalArgumentException(
                String.format(
                    AppleSkylarkCommon.BAD_PROVIDERS_ELEM_ERROR,
                    EvalUtils.getDataTypeName(toAddObject)));
          } else {
            this.addTransitiveAndPropagate((ObjcProvider) toAddObject);
          }
        }
      }
    }

    /**
     * Adds the given providers from skylark, but propagate any normally-propagated items
     * only to direct dependers. An error is thrown if toAdd is not an iterable of ObjcProvider
     * instances.
     */
    @SuppressWarnings("unchecked")
    void addDirectDepProvidersFromSkylark(Object toAdd) {
      if (!(toAdd instanceof Iterable)) {
        throw new IllegalArgumentException(
            String.format(
                AppleSkylarkCommon.BAD_PROVIDERS_ITER_ERROR, EvalUtils.getDataTypeName(toAdd)));
      } else {
        Iterable<Object> toAddIterable = (Iterable<Object>) toAdd;
        for (Object toAddObject : toAddIterable) {
          if (!(toAddObject instanceof ObjcProvider)) {
            throw new IllegalArgumentException(
                String.format(
                    AppleSkylarkCommon.BAD_PROVIDERS_ELEM_ERROR,
                    EvalUtils.getDataTypeName(toAddObject)));
          } else {
            this.addAsDirectDeps((ObjcProvider) toAddObject);
          }
        }
      }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public ObjcProvider build() {
      ImmutableMap.Builder<Key<?>, NestedSet<?>> propagatedBuilder = new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : items.entrySet()) {
        propagatedBuilder.put(typeEntry.getKey(), typeEntry.getValue().build());
      }
      ImmutableMap.Builder<Key<?>, NestedSet<?>> nonPropagatedBuilder =
          new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : nonPropagatedItems.entrySet()) {
        nonPropagatedBuilder.put(typeEntry.getKey(), typeEntry.getValue().build());
      }
      ImmutableMap.Builder<Key<?>, NestedSet<?>> strictDependencyBuilder =
          new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : strictDependencyItems.entrySet()) {
        strictDependencyBuilder.put(typeEntry.getKey(), typeEntry.getValue().build());
      }

      return new ObjcProvider(
          starlarkSemantics,
          propagatedBuilder.build(),
          nonPropagatedBuilder.build(),
          strictDependencyBuilder.build());
    }
  }

  private static class Constructor extends BuiltinProvider<ObjcProvider>
      implements WithLegacySkylarkName {
    public Constructor() {
      super(ObjcProvider.SKYLARK_NAME, ObjcProvider.class);
    }

    @Override
    public String getSkylarkName() {
      return SKYLARK_NAME;
    }

    @Override
    public String getErrorMessageFormatForUnknownField() {
      return "ObjcProvider field '%s' could not be instantiated";
    }
  }
}
