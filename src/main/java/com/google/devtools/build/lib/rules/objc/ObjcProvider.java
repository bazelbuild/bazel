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
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider.WithLegacySkylarkName;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.StarlarkList;
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
 *
 * <p>Most of the compilation information is stored in an embedded {@code CcCompilationContext}. The
 * objc proto strict dependency include paths are stored in a special, non-propagated field {@code
 * strictDependencyIncludes}.
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

  public static final Key<String> SDK_DYLIB = new Key<>(STABLE_ORDER, "sdk_dylib", String.class);
  public static final Key<SdkFramework> SDK_FRAMEWORK =
      new Key<>(STABLE_ORDER, "sdk_framework", SdkFramework.class);
  public static final Key<SdkFramework> WEAK_SDK_FRAMEWORK =
      new Key<>(STABLE_ORDER, "weak_sdk_framework", SdkFramework.class);
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
   * Exec paths of {@code .framework} directories corresponding to frameworks to include in search
   * paths, but not to link. These cause -F arguments (framework search paths) to be added to each
   * compile action, but do not cause -framework (link framework) arguments to be added to link
   * actions.
   */
  public static final Key<PathFragment> FRAMEWORK_SEARCH_PATHS =
      new Key<>(LINK_ORDER, "framework_search_paths", PathFragment.class);

  /** The static library files of user-specified static frameworks. */
  public static final Key<Artifact> STATIC_FRAMEWORK_FILE =
      new Key<>(STABLE_ORDER, "static_framework_file", Artifact.class);

  /** The dynamic library files of user-specified dynamic frameworks. */
  public static final Key<Artifact> DYNAMIC_FRAMEWORK_FILE =
      new Key<>(STABLE_ORDER, "dynamic_framework_file", Artifact.class);

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

  private final CcCompilationContext ccCompilationContext;

  /** Keys corresponding to compile information that has been migrated to CcCompilationContext. */
  static final ImmutableSet<Key<?>> KEYS_FOR_COMPILE_INFO =
      ImmutableSet.<Key<?>>of(
          DEFINE, FRAMEWORK_SEARCH_PATHS, HEADER, INCLUDE, INCLUDE_SYSTEM, IQUOTE);

  /** All keys in ObjcProvider that will be passed in the corresponding Skylark provider. */
  static final ImmutableList<Key<?>> KEYS_FOR_SKYLARK =
      ImmutableList.<Key<?>>of(
          DEFINE,
          DYNAMIC_FRAMEWORK_FILE,
          EXPORTED_DEBUG_ARTIFACTS,
          FRAMEWORK_SEARCH_PATHS,
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
          SDK_DYLIB,
          SDK_FRAMEWORK,
          SOURCE,
          STATIC_FRAMEWORK_FILE,
          UMBRELLA_HEADER,
          WEAK_SDK_FRAMEWORK);

  /**
   * Keys that should be kept as directItems. This is limited to a few keys that have larger
   * performance implications when flattened in a transitive fashion and/or require non-transitive
   * access (e.g. what module map did a target generate?).
   *
   * <p>Keys:
   *
   * <ul>
   *   <li>HEADER: To expose all header files, including generated proto header files, to IDEs.
   *   <li>SOURCE: To expose all source files, including generated J2Objc source files, to IDEs.
   *   <li>MODULE_MAP: To expose generated module maps to IDEs (only one is expected per target).
   * </ul>
   */
  static final ImmutableSet<Key<?>> KEYS_FOR_DIRECT =
      ImmutableSet.<Key<?>>of(HEADER, MODULE_MAP, SOURCE);

  public ImmutableList<PathFragment> getStrictDependencyIncludes() {
    return strictDependencyIncludes;
  }

  @Override
  public Depset /*<String>*/ defineForStarlark() {
    return getCcCompilationContext().getSkylarkDefines();
  }

  public NestedSet<String> define() {
    return getCcCompilationContext().getDefines();
  }

  @Override
  public Depset /*<Artifact>*/ dynamicFrameworkFileForStarlark() {
    return Depset.of(Artifact.TYPE, dynamicFrameworkFile());
  }

  NestedSet<Artifact> dynamicFrameworkFile() {
    return get(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<Artifact>*/ exportedDebugArtifacts() {
    return Depset.of(Artifact.TYPE, get(EXPORTED_DEBUG_ARTIFACTS));
  }

  @Override
  public Depset frameworkIncludeForStarlark() {
    // Starlark code expects the framework path to include the ".framework" directory, which is then
    // stripped to get the actual framework search path.  CcCompilationContext only stores the
    // framework search path, so the best we can do is to append a fake ".framework" directory.
    // This at least preserves the behavior when the field is used for its intended purpose.
    return Depset.of(
        SkylarkType.STRING,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            frameworkInclude().stream()
                .map(x -> x.getChild("fake.framework").getSafePathString())
                .collect(ImmutableList.toImmutableList())));
  }

  public ImmutableList<PathFragment> frameworkInclude() {
    return getCcCompilationContext().getFrameworkIncludeDirs();
  }

  @Override
  public Depset /*<Artifact>*/ forceLoadLibrary() {
    return Depset.of(Artifact.TYPE, get(FORCE_LOAD_LIBRARY));
  }

  @Override
  public Depset /*<Artifact>*/ headerForStarlark() {
    return Depset.of(Artifact.TYPE, header());
  }

  public NestedSet<Artifact> header() {
    return getCcCompilationContext().getDeclaredIncludeSrcs();
  }

  @Override
  public Sequence<Artifact> directHeaders() {
    return getDirect(HEADER);
  }

  @Override
  public Depset /*<Artifact>*/ importedLibrary() {
    return Depset.of(Artifact.TYPE, get(IMPORTED_LIBRARY));
  }

  @Override
  public Depset /*<String>*/ includeForStarlark() {
    return Depset.of(
        SkylarkType.STRING,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            include().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  public ImmutableList<PathFragment> include() {
    ImmutableList.Builder<PathFragment> listBuilder = ImmutableList.builder();
    return listBuilder
        .addAll(strictDependencyIncludes)
        .addAll(getCcCompilationContext().getIncludeDirs())
        .build();
  }

  @Override
  public Depset /*<String>*/ strictIncludeForStarlark() {
    return Depset.of(
        SkylarkType.STRING,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getStrictDependencyIncludes().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @Override
  public Depset systemIncludeForStarlark() {
    return getCcCompilationContext().getSkylarkSystemIncludeDirs();
  }

  public ImmutableList<PathFragment> systemInclude() {
    return getCcCompilationContext().getSystemIncludeDirs();
  }

  @Override
  public Depset quoteIncludeForStarlark() {
    return getCcCompilationContext().getSkylarkQuoteIncludeDirs();
  }

  public ImmutableList<PathFragment> quoteInclude() {
    return getCcCompilationContext().getQuoteIncludeDirs();
  }

  @Override
  public Depset /*<Artifact>*/ j2objcLibrary() {
    return Depset.of(Artifact.TYPE, get(J2OBJC_LIBRARY));
  }

  @Override
  public Depset /*<Artifact>*/ jreLibrary() {
    return Depset.of(Artifact.TYPE, get(JRE_LIBRARY));
  }

  @Override
  public Depset /*<Artifact>*/ library() {
    return Depset.of(Artifact.TYPE, get(LIBRARY));
  }

  @Override
  public Depset /*<Artifact>*/ linkInputs() {
    return Depset.of(Artifact.TYPE, get(LINK_INPUTS));
  }

  @Override
  public Depset /*<Artifact>*/ linkedBinary() {
    return Depset.of(Artifact.TYPE, get(LINKED_BINARY));
  }

  @Override
  public Depset /*<Artifact>*/ linkmapFile() {
    return Depset.of(Artifact.TYPE, get(LINKMAP_FILE));
  }

  @Override
  public Depset /*<String>*/ linkopt() {
    return Depset.of(SkylarkType.STRING, get(LINKOPT));
  }

  @Override
  public Depset /*<Artifact>*/ mergeZip() {
    return Depset.of(Artifact.TYPE, get(MERGE_ZIP));
  }

  @Override
  public Depset /*<Artifact>*/ moduleMap() {
    return Depset.of(Artifact.TYPE, get(MODULE_MAP));
  }

  @Override
  public Sequence<Artifact> directModuleMaps() {
    return getDirect(MODULE_MAP);
  }

  @Override
  public Depset /*<Artifact>*/ multiArchDynamicLibraries() {
    return Depset.of(Artifact.TYPE, get(MULTI_ARCH_DYNAMIC_LIBRARIES));
  }

  @Override
  public Depset /*<Artifact>*/ multiArchLinkedArchives() {
    return Depset.of(Artifact.TYPE, get(MULTI_ARCH_LINKED_ARCHIVES));
  }

  @Override
  public Depset /*<Artifact>*/ multiArchLinkedBinaries() {
    return Depset.of(Artifact.TYPE, get(MULTI_ARCH_LINKED_BINARIES));
  }

  @Override
  public Depset /*<String>*/ sdkDylib() {
    return Depset.of(SkylarkType.STRING, get(SDK_DYLIB));
  }

  @Override
  public Depset sdkFramework() {
    return (Depset)
        ObjcProviderSkylarkConverters.convertToSkylark(SDK_FRAMEWORK, get(SDK_FRAMEWORK));
  }

  @Override
  public Depset /*<Artifact>*/ sourceForStarlark() {
    return Depset.of(Artifact.TYPE, source());
  }

  NestedSet<Artifact> source() {
    return get(SOURCE);
  }

  @Override
  public Sequence<Artifact> directSources() {
    return getDirect(SOURCE);
  }

  @Override
  public Depset /*<Artifact>*/ staticFrameworkFileForStarlark() {
    return Depset.of(Artifact.TYPE, staticFrameworkFile());
  }

  NestedSet<Artifact> staticFrameworkFile() {
    return get(STATIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<Artifact>*/ umbrellaHeader() {
    return Depset.of(Artifact.TYPE, get(UMBRELLA_HEADER));
  }

  @Override
  public Depset weakSdkFramework() {
    return (Depset)
        ObjcProviderSkylarkConverters.convertToSkylark(WEAK_SDK_FRAMEWORK, get(WEAK_SDK_FRAMEWORK));
  }

  @Override
  public CcCompilationContext getCcCompilationContext() {
    return ccCompilationContext;
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
          DYNAMIC_FRAMEWORK_FILE,
          FLAG,
          MERGE_ZIP,
          FRAMEWORK_SEARCH_PATHS,
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

  /** Skylark constructor and identifier for ObjcProvider. */
  public static final BuiltinProvider<ObjcProvider> SKYLARK_CONSTRUCTOR = new Constructor();

  private ObjcProvider(
      StarlarkSemantics semantics,
      ImmutableMap<Key<?>, NestedSet<?>> items,
      ImmutableList<PathFragment> strictDependencyIncludes,
      ImmutableListMultimap<Key<?>, ?> directItems,
      CcCompilationContext ccCompilationContext) {
    this.semantics = semantics;
    this.items = Preconditions.checkNotNull(items);
    this.strictDependencyIncludes = Preconditions.checkNotNull(strictDependencyIncludes);
    this.directItems = Preconditions.checkNotNull(directItems);
    this.ccCompilationContext = ccCompilationContext;
  }

  @Override
  public BuiltinProvider<ObjcProvider> getProvider() {
    return SKYLARK_CONSTRUCTOR;
  }

  /**
   * All artifacts, bundleable files, etc. of the type specified by {@code key}.
   */
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
    return get(FLAG).toList().contains(flag);
  }

  /** Returns the list of .a files required for linking that arise from objc libraries. */
  ImmutableList<Artifact> getObjcLibraries() {
    // JRE libraries must be ordered after all regular objc libraries.
    NestedSet<Artifact> jreLibs = get(JRE_LIBRARY);
    return ImmutableList.<Artifact>builder()
        .addAll(
            Iterables.filter(get(LIBRARY).toList(), Predicates.not(Predicates.in(jreLibs.toSet()))))
        .addAll(jreLibs.toList())
        .build();
  }

  /** Returns the list of .a files required for linking that arise from cc libraries. */
  List<Artifact> getCcLibraries() {
    CcLinkingContext ccLinkingContext =
        CcLinkingContext.builder().addLibraries(get(CC_LIBRARY).toList()).build();
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
      for (Artifact libraryToAvoid : avoidProvider.getPropagable(LIBRARY).toList()) {
        avoidLibrariesSet.add(libraryToAvoid.getRunfilesPath());
      }
    }
    ObjcProvider.NativeBuilder objcProviderBuilder = new ObjcProvider.NativeBuilder(semantics);
    for (Key<?> key : items.keySet()) {
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
    objcProviderBuilder.addStrictDependencyIncludes(strictDependencyIncludes);
    objcProviderBuilder.setCcCompilationContext(ccCompilationContext);
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
    if (propagableItems != null) {
      objcProviderBuilder.addAll(key,
          Iterables.filter(propagableItems.toList(), filterPredicate));
    }
  }

  private void addTransitiveAndAvoidArtifacts(ObjcProvider.Builder objcProviderBuilder,
      Key<Artifact> key, Iterable<ObjcProvider> avoidProviders) {
    // Artifacts to avoid may be in a different configuration and thus a different
    // root directory, hence only the path fragment after the root directory is compared.
    HashSet<PathFragment> avoidPathsSet = new HashSet<>();
    for (ObjcProvider avoidProvider : avoidProviders) {
      for (Artifact artifact : avoidProvider.getPropagable(key).toList()) {
        avoidPathsSet.add(artifact.getRunfilesPath());
      }
    }
    addTransitiveAndFilter(objcProviderBuilder, key, notContainedIn(avoidPathsSet));
  }

  private <T> void addTransitiveAndAvoid(
      ObjcProvider.Builder objcProviderBuilder, Key<T> key, Iterable<ObjcProvider> avoidProviders) {
    HashSet<T> avoidItemsSet = new HashSet<T>();
    for (ObjcProvider avoidProvider : avoidProviders) {
      avoidItemsSet.addAll(avoidProvider.getPropagable(key).toList());
    }
    addTransitiveAndFilter(objcProviderBuilder, key, Predicates.not(Predicates.in(avoidItemsSet)));
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
    for (Artifact file : get(key).toList()) {
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
    for (Artifact file : get(key).toList()) {
      PathFragment frameworkDir = file.getExecPath().getParentDirectory();
      checkIsFrameworkDirectory(frameworkDir);
      paths.add(getFrameworkPath(frameworkDir));
    }
    return paths.build();
  }

  @Override
  public Depset /*<String>*/ dynamicFrameworkNamesForStarlark() {
    return Depset.of(SkylarkType.STRING, dynamicFrameworkNames());
  }

  NestedSet<String> dynamicFrameworkNames() {
    return getFrameworkNames(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<String>*/ dynamicFrameworkPathsForStarlark() {
    return Depset.of(SkylarkType.STRING, dynamicFrameworkPaths());
  }

  NestedSet<String> dynamicFrameworkPaths() {
    return getFrameworkPaths(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<String>*/ staticFrameworkNamesForStarlark() {
    return Depset.of(SkylarkType.STRING, staticFrameworkNames());
  }

  NestedSet<String> staticFrameworkNames() {
    return getFrameworkNames(STATIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<String>*/ staticFrameworkPathsForStarlark() {
    return Depset.of(SkylarkType.STRING, staticFrameworkPaths());
  }

  NestedSet<String> staticFrameworkPaths() {
    return getFrameworkPaths(STATIC_FRAMEWORK_FILE);
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public abstract static class Builder {

    private final StarlarkSemantics starlarkSemantics;
    private final Map<Key<?>, NestedSetBuilder<?>> items = new HashMap<>();
    private final ImmutableList.Builder<PathFragment> strictDependencyIncludes =
        ImmutableList.builder();

    // Only includes items or lists added directly, never flattens any NestedSets.
    private final ImmutableListMultimap.Builder<Key<?>, ?> directItems =
        new ImmutableListMultimap.Builder<>();

    public Builder(StarlarkSemantics semantics) {
      this.starlarkSemantics = semantics;
    }

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
        uncheckedAddTransitive(typeEntry.getKey(), typeEntry.getValue());
      }
      return this;
    }

    /**
     * Add all elements from a single key of the given provider, and propagate them to any
     * (transitive) dependers on this ObjcProvider.
     */
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
    public <E> Builder addTransitiveAndPropagate(Key<E> key, NestedSet<E> items) {
      uncheckedAddTransitive(key, items);
      return this;
    }

    /** Return an EvalException for having a bad key in the direct dependency provider. */
    private static <E> EvalException badDirectDependencyKeyError(Key<E> key) {
      return new EvalException(
          null,
          String.format(
              AppleSkylarkCommon.BAD_DIRECT_DEPENDENCY_KEY_ERROR, key.getSkylarkKeyName()));
    }

    /**
     * Propagate keys and values from the given provider to direct dependers of this ObjcProvider.
     * We no longer support this generically -- the only remaining use case we support is for
     * includes.
     */
    public Builder addAsDirectDeps(ObjcProvider provider) throws EvalException {
      CcCompilationContext providerCcCompilationContext = provider.getCcCompilationContext();

      strictDependencyIncludes.addAll(providerCcCompilationContext.getIncludeDirs());

      // Emit an error if we find any other information in the provider.
      for (Key<?> key : provider.items.keySet()) {
        throw badDirectDependencyKeyError(key);
      }
      if (!provider.define().isEmpty()) {
        throw badDirectDependencyKeyError(DEFINE);
      }
      if (!provider.header().isEmpty()) {
        throw badDirectDependencyKeyError(HEADER);
      }
      if (!providerCcCompilationContext.getFrameworkIncludeDirs().isEmpty()) {
        throw badDirectDependencyKeyError(FRAMEWORK_SEARCH_PATHS);
      }
      if (!providerCcCompilationContext.getSystemIncludeDirs().isEmpty()) {
        throw badDirectDependencyKeyError(INCLUDE_SYSTEM);
      }
      if (!providerCcCompilationContext.getQuoteIncludeDirs().isEmpty()) {
        throw badDirectDependencyKeyError(IQUOTE);
      }
      return this;
    }

    /**
     * Add element, and propagate it to any (transitive) dependers on this ObjcProvider.
     */
    public <E> Builder add(Key<E> key, E toAdd) {
      uncheckedAddAll(key, ImmutableList.of(toAdd));
      return this;
    }

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
    public <E> Builder addAll(Key<E> key, Iterable<? extends E> toAdd) {
      uncheckedAddAll(key, toAdd);
      return this;
    }

    public <E> Builder addAllDirect(Key<E> key, Iterable<? extends E> toAdd) {
      Preconditions.checkState(KEYS_FOR_DIRECT.contains(key));
      uncheckedAddAllDirect(key, toAdd);
      return this;
    }

    protected Builder addStrictDependencyIncludes(Iterable<PathFragment> includes) {
      strictDependencyIncludes.addAll(includes);
      return this;
    }

    abstract ObjcProvider build();

    protected ObjcProvider build(CcCompilationContext ccCompilationContext) {
      ImmutableMap.Builder<Key<?>, NestedSet<?>> propagatedBuilder = new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : items.entrySet()) {
        propagatedBuilder.put(typeEntry.getKey(), typeEntry.getValue().build());
      }
      return new ObjcProvider(
          starlarkSemantics,
          propagatedBuilder.build(),
          strictDependencyIncludes.build(),
          directItems.build(),
          ccCompilationContext);
    }
  }

  /** A builder for this context, specialized for native use. */
  public static final class NativeBuilder extends Builder {
    private CcCompilationContext ccCompilationContext = CcCompilationContext.EMPTY;

    public NativeBuilder(StarlarkSemantics semantics) {
      super(semantics);
    }

    Builder setCcCompilationContext(CcCompilationContext ccCompilationContext) {
      Preconditions.checkState(this.ccCompilationContext == CcCompilationContext.EMPTY);
      Preconditions.checkNotNull(ccCompilationContext);
      this.ccCompilationContext = ccCompilationContext;
      return this;
    }

    @Override
    public ObjcProvider build() {
      return build(ccCompilationContext);
    }
  }

  /** A builder for this context, specialized for Starlark use. */
  public static final class StarlarkBuilder extends Builder {
    private final CcCompilationContext.Builder ccCompilationContextBuilder =
        CcCompilationContext.builder(null, null, null);

    public StarlarkBuilder(StarlarkSemantics semantics) {
      super(semantics);
    }

    /**
     * Add elements in toAdd with the given key from skylark. An error is thrown if toAdd is not an
     * appropriate Depset.
     */
    void addElementsFromSkylark(Key<?> key, Object skylarkToAdd) throws EvalException {
      NestedSet<?> toAdd = ObjcProviderSkylarkConverters.convertToJava(key, skylarkToAdd);
      if (KEYS_FOR_COMPILE_INFO.contains(key)) {
        String keyName = key.getSkylarkKeyName();

        if (key == DEFINE) {
          ccCompilationContextBuilder.addDefines(
              Depset.getSetFromNoneableParam(skylarkToAdd, String.class, keyName));
        } else if (key == FRAMEWORK_SEARCH_PATHS) {
          // Due to legacy reasons, There is a mismatch between the starlark interface for the
          // framework search path, and the internal representation.  The interface specifies that
          // framework_search_paths include the framework directories, but internally we only store
          // their parents.  We will eventually clean up the interface, but for now we need to do
          // this ugly conversion.

          ImmutableList<PathFragment> frameworks =
              Depset.getSetFromNoneableParam(skylarkToAdd, String.class, keyName).toList().stream()
                  .map(x -> PathFragment.create(x))
                  .collect(ImmutableList.toImmutableList());

          ImmutableList.Builder<PathFragment> frameworkSearchPaths = ImmutableList.builder();
          for (PathFragment framework : frameworks) {
            if (!framework.getSafePathString().endsWith(FRAMEWORK_SUFFIX)) {
              throw new EvalException(
                  null, String.format(AppleSkylarkCommon.BAD_FRAMEWORK_PATH_ERROR, framework));
            }
            frameworkSearchPaths.add(framework.getParentDirectory());
          }
          ccCompilationContextBuilder.addFrameworkIncludeDirs(frameworkSearchPaths.build());
        } else if (key == HEADER) {
          ImmutableList<Artifact> hdrs =
              Depset.getSetFromNoneableParam(skylarkToAdd, Artifact.class, keyName).toList();
          ccCompilationContextBuilder.addDeclaredIncludeSrcs(hdrs);
          ccCompilationContextBuilder.addTextualHdrs(hdrs);
        } else if (key == INCLUDE) {
          ccCompilationContextBuilder.addIncludeDirs(
              Depset.getSetFromNoneableParam(skylarkToAdd, String.class, keyName).toList().stream()
                  .map(x -> PathFragment.create(x))
                  .collect(ImmutableList.toImmutableList()));
        } else if (key == INCLUDE_SYSTEM) {
          ccCompilationContextBuilder.addSystemIncludeDirs(
              Depset.getSetFromNoneableParam(skylarkToAdd, String.class, keyName).toList().stream()
                  .map(x -> PathFragment.create(x))
                  .collect(ImmutableList.toImmutableList()));
        } else if (key == IQUOTE) {
          ccCompilationContextBuilder.addQuoteIncludeDirs(
              Depset.getSetFromNoneableParam(skylarkToAdd, String.class, keyName).toList().stream()
                  .map(x -> PathFragment.create(x))
                  .collect(ImmutableList.toImmutableList()));
        }
      } else {
        uncheckedAddTransitive(key, toAdd);
      }

      if (KEYS_FOR_DIRECT.contains(key)) {
        uncheckedAddAllDirect(key, toAdd.toList());
      }
    }

    /**
     * Adds the given providers from skylark. An error is thrown if toAdd is not an iterable of
     * ObjcProvider instances.
     */
    @SuppressWarnings("unchecked")
    void addProvidersFromSkylark(Object toAdd) throws EvalException {
      if (!(toAdd instanceof Iterable)) {
        throw new EvalException(
            null,
            String.format(
                AppleSkylarkCommon.BAD_PROVIDERS_ITER_ERROR, EvalUtils.getDataTypeName(toAdd)));
      } else {
        Iterable<Object> toAddIterable = (Iterable<Object>) toAdd;
        for (Object toAddObject : toAddIterable) {
          if (!(toAddObject instanceof ObjcProvider)) {
            throw new EvalException(
                null,
                String.format(
                    AppleSkylarkCommon.BAD_PROVIDERS_ELEM_ERROR,
                    EvalUtils.getDataTypeName(toAddObject)));
          } else {
            ObjcProvider objcProvider = (ObjcProvider) toAddObject;
            this.addTransitiveAndPropagate(objcProvider);
            ccCompilationContextBuilder.mergeDependentCcCompilationContext(
                objcProvider.getCcCompilationContext());
          }
        }
      }
    }

    /**
     * Adds the given providers from skylark, but propagate any normally-propagated items only to
     * direct dependers. An error is thrown if toAdd is not an iterable of ObjcProvider instances.
     */
    @SuppressWarnings("unchecked")
    void addDirectDepProvidersFromSkylark(Object toAdd) throws EvalException {
      if (!(toAdd instanceof Iterable)) {
        throw new EvalException(
            null,
            String.format(
                AppleSkylarkCommon.BAD_PROVIDERS_ITER_ERROR, EvalUtils.getDataTypeName(toAdd)));
      } else {
        Iterable<Object> toAddIterable = (Iterable<Object>) toAdd;
        for (Object toAddObject : toAddIterable) {
          if (!(toAddObject instanceof ObjcProvider)) {
            throw new EvalException(
                null,
                String.format(
                    AppleSkylarkCommon.BAD_PROVIDERS_ELEM_ERROR,
                    EvalUtils.getDataTypeName(toAddObject)));
          } else {
            this.addAsDirectDeps((ObjcProvider) toAddObject);
          }
        }
      }
    }

    /**
     * Adds the given strict include paths from skylark. An error is thrown if skylarkToAdd is not
     * an appropriate Depset.
     */
    @SuppressWarnings("unchecked")
    void addStrictIncludeFromSkylark(Object skylarkToAdd) throws EvalException {
      NestedSet<PathFragment> toAdd =
          (NestedSet<PathFragment>)
              ObjcProviderSkylarkConverters.convertToJava(INCLUDE, skylarkToAdd);

      addStrictDependencyIncludes(toAdd.toList());
    }

    @Override
    public ObjcProvider build() {
      return build(ccCompilationContextBuilder.build());
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
