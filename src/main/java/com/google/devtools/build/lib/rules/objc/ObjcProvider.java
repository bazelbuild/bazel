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
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.BuiltinProvider.WithLegacyStarlarkName;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

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
   * Include search paths {@code -I} that are stored specially in their own field, and not
   * propagated transitively.
   */
  public static final Key<PathFragment> STRICT_INCLUDE =
      new Key<>(LINK_ORDER, "strict_include", PathFragment.class);

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

  /** The static library files of user-specified static frameworks. */
  public static final Key<Artifact> STATIC_FRAMEWORK_FILE =
      new Key<>(STABLE_ORDER, "static_framework_file", Artifact.class);

  /** The dynamic library files of user-specified dynamic frameworks. */
  public static final Key<Artifact> DYNAMIC_FRAMEWORK_FILE =
      new Key<>(STABLE_ORDER, "dynamic_framework_file", Artifact.class);

  /** Linking information from cc dependencies. */
  public static final Key<LibraryToLink> CC_LIBRARY =
      new Key<>(LINK_ORDER, "cc_library", LibraryToLink.class);

  /** Linkstamps from cc dependencies. */
  // This key exists only to facilitate passing linkstamp data from ObjcLibrary's input CcInfos to
  // its output CcInfo. Other consumers should look at ObjcLibrary's output CcInfo rather than the
  // data behind this key.
  static final Key<CcLinkingContext.Linkstamp> LINKSTAMP =
      new Key<>(STABLE_ORDER, "linkstamp", CcLinkingContext.Linkstamp.class);

  static final Key<LinkerInput> LINKSTAMP_LINKER_INPUTS =
      new Key<>(STABLE_ORDER, "linkstamp_linker_inputs", LinkerInput.class);

  /**
   * Linking options from dependencies.
   */
  public static final Key<String> LINKOPT = new Key<>(LINK_ORDER, "linkopt", String.class);

  public static final Key<LinkerInput> CC_LINKER_INPUTS =
      new Key<>(LINK_ORDER, "cc_linker_inputs", LinkerInput.class);

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

  /** All keys in ObjcProvider that will be passed in the corresponding Starlark provider. */
  static final ImmutableList<Key<?>> KEYS_FOR_STARLARK =
      ImmutableList.<Key<?>>of(
          DYNAMIC_FRAMEWORK_FILE,
          FORCE_LOAD_LIBRARY,
          HEADER,
          IMPORTED_LIBRARY,
          J2OBJC_LIBRARY,
          JRE_LIBRARY,
          LIBRARY,
          LINK_INPUTS,
          LINKOPT,
          MODULE_MAP,
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

  /** Keys that are only used for direct fields, and nothing else. */
  static final ImmutableSet<Key<?>> KEYS_FOR_DIRECT_ONLY = ImmutableSet.<Key<?>>of(HEADER);

  public ImmutableList<PathFragment> getStrictDependencyIncludes() {
    return strictDependencyIncludes;
  }

  @Override
  public Depset /*<Artifact>*/ dynamicFrameworkFileForStarlark() {
    return Depset.of(Artifact.TYPE, dynamicFrameworkFile());
  }

  NestedSet<Artifact> dynamicFrameworkFile() {
    return get(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<Artifact>*/ forceLoadLibrary() {
    return Depset.of(Artifact.TYPE, get(FORCE_LOAD_LIBRARY));
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
  public Depset /*<String>*/ strictIncludeForStarlark() {
    return Depset.of(
        Depset.ElementType.STRING,
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            getStrictDependencyIncludes().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
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
  public Depset /*<String>*/ linkopt() {
    return Depset.of(Depset.ElementType.STRING, getLinkopt());
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
  public Depset /*<String>*/ sdkDylib() {
    return Depset.of(Depset.ElementType.STRING, get(SDK_DYLIB));
  }

  @Override
  public Depset sdkFramework() {
    return (Depset)
        ObjcProviderStarlarkConverters.convertToStarlark(SDK_FRAMEWORK, getFrameworks());
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
        ObjcProviderStarlarkConverters.convertToStarlark(
            WEAK_SDK_FRAMEWORK, get(WEAK_SDK_FRAMEWORK));
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
          // LibraryToLink not exposed to Starlark.
          CC_LIBRARY,
          // Flag enum is not exposed to Starlark.
          FLAG,
          // Linkstamp is not exposed to Starlark. See commentary at its definition.
          LINKSTAMP,
          // Strict include is handled specially.
          STRICT_INCLUDE,
          // Linker inputs avoid unnecessarily flattening nested sets
          CC_LINKER_INPUTS,
          LINKSTAMP_LINKER_INPUTS);

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
          DYNAMIC_FRAMEWORK_FILE,
          FLAG,
          LINKOPT,
          LINKSTAMP,
          LINK_INPUTS,
          SDK_DYLIB,
          SDK_FRAMEWORK,
          WEAK_SDK_FRAMEWORK);

  /**
   * Returns the Starlark key for the given string, or null if no such key exists or is available to
   * Starlark.
   */
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
      StarlarkSemantics semantics,
      ImmutableMap<Key<?>, NestedSet<?>> items,
      ImmutableList<PathFragment> strictDependencyIncludes,
      ImmutableListMultimap<Key<?>, ?> directItems) {
    this.semantics = semantics;
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

  /**
   * All artifacts, bundleable files, etc. of the type specified by {@code key}.
   */
  @SuppressWarnings("unchecked")
  public <E> NestedSet<E> get(Key<E> key) {
    Preconditions.checkNotNull(key);
    Preconditions.checkArgument(!key.equals(CC_LIBRARY));
    Preconditions.checkArgument(!key.equals(LINKOPT));
    Preconditions.checkArgument(!key.equals(LINKSTAMP));
    Preconditions.checkArgument(!key.equals(SDK_FRAMEWORK));
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

  @StarlarkMethod(name = "linker_inputs", documented = false, useStarlarkThread = true)
  public Depset linkerInputsForStarlark(StarlarkThread thread) throws EvalException {
    if (!isBuiltIn(thread)) {
      throw Starlark.errorf("Cannot use builtin API");
    }
    return Depset.of(LinkerInput.TYPE, get(ObjcProvider.CC_LINKER_INPUTS));
  }

  @StarlarkMethod(name = "linkstamp_linker_inputs", documented = false, useStarlarkThread = true)
  public Depset linkstampLinkerInputsForStarlark(StarlarkThread thread) throws EvalException {
    if (!isBuiltIn(thread)) {
      throw Starlark.errorf("Cannot use builtin API");
    }
    return Depset.of(LinkerInput.TYPE, get(ObjcProvider.LINKSTAMP_LINKER_INPUTS));
  }

  private static boolean isBuiltIn(StarlarkThread thread) {
    Label label =
        ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
            .label();
    return label.getPackageIdentifier().getRepository().toString().equals("@_builtins");
  }

  @SuppressWarnings("unchecked")
  public NestedSet<LibraryToLink> getTransitiveCcLibraries() {
    NestedSetBuilder<LibraryToLink> builder = NestedSetBuilder.linkOrder();
    if (items.containsKey(CC_LINKER_INPUTS)) {
      for (LinkerInput linkerInput :
          ((NestedSet<LinkerInput>) items.get(CC_LINKER_INPUTS)).toList()) {
        builder.addAll(linkerInput.getLibraries());
      }
      return builder.build();
    } else {
      return NestedSetBuilder.emptySet(LINK_ORDER);
    }
  }

  @SuppressWarnings("unchecked")
  public NestedSet<String> getLinkopt() {
    NestedSetBuilder<String> builder = NestedSetBuilder.linkOrder();
    if (items.containsKey(LINKOPT)) {
      builder.addTransitive((NestedSet<String>) items.get(LINKOPT));
    }

    if (items.containsKey(CC_LINKER_INPUTS)) {
      for (LinkerInput linkerInput :
          ((NestedSet<LinkerInput>) items.get(CC_LINKER_INPUTS)).toList()) {
        for (LinkOptions linkOptions : linkerInput.getUserLinkFlags()) {
          ImmutableList<String> linkOpts = linkOptions.get();
          ImmutableList.Builder<String> nonFrameworkLinkOpts = ImmutableList.builder();
          for (UnmodifiableIterator<String> iterator = linkOpts.iterator(); iterator.hasNext(); ) {
            String arg = iterator.next();
            if (arg.equals("-framework") && iterator.hasNext()) {
              iterator.next();
            } else {
              nonFrameworkLinkOpts.add(arg);
            }
          }
          builder.addAll(nonFrameworkLinkOpts.build());
        }
      }
    }
    return builder.build();
  }

  @SuppressWarnings("unchecked")
  public NestedSet<SdkFramework> getFrameworks() {
    NestedSetBuilder<SdkFramework> builder = NestedSetBuilder.linkOrder();
    if (items.containsKey(SDK_FRAMEWORK)) {
      builder.addTransitive((NestedSet<SdkFramework>) items.get(SDK_FRAMEWORK));
    }
    if (items.containsKey(CC_LINKER_INPUTS)) {
      for (LinkerInput linkerInput :
          ((NestedSet<LinkerInput>) items.get(CC_LINKER_INPUTS)).toList()) {
        for (LinkOptions linkOptions : linkerInput.getUserLinkFlags()) {
          ImmutableList<String> linkOpts = linkOptions.get();
          ImmutableList.Builder<SdkFramework> frameworkLinkOpts = ImmutableList.builder();
          for (UnmodifiableIterator<String> iterator = linkOpts.iterator(); iterator.hasNext(); ) {
            String arg = iterator.next();
            if (arg.equals("-framework") && iterator.hasNext()) {
              String framework = iterator.next();
              frameworkLinkOpts.add(new SdkFramework(framework));
            }
          }
          builder.addAll(frameworkLinkOpts.build());
        }
      }
    }

    return builder.build();
  }

  @SuppressWarnings("unchecked")
  public NestedSet<Linkstamp> getLinkstamps() {
    NestedSetBuilder<Linkstamp> builder = NestedSetBuilder.linkOrder();
    if (items.containsKey(LINKSTAMP_LINKER_INPUTS)) {
      for (LinkerInput linkerInput :
          ((NestedSet<LinkerInput>) items.get(LINKSTAMP_LINKER_INPUTS)).toList()) {
        builder.addAll(linkerInput.getLinkstamps());
      }
    }
    return builder.build();
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
        CcLinkingContext.builder().addLibraries(getTransitiveCcLibraries().toList()).build();
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
    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder(semantics);
    for (Key<?> key : items.keySet()) {
      if (key == CC_LINKER_INPUTS) {
        addTransitiveAndFilter(
            objcProviderBuilder, CC_LINKER_INPUTS, ccLibraryNotYetLinked(avoidLibrariesSet));
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
  private static Predicate<LinkerInput> ccLibraryNotYetLinked(
      final HashSet<PathFragment> runfilesPaths) {
    return linkerInput -> !checkIfLibraryIsInPaths(linkerInput, runfilesPaths);
  }

  private static boolean checkIfLibraryIsInPaths(
      LinkerInput linkerInput, HashSet<PathFragment> runfilesPaths) {
    ImmutableList.Builder<PathFragment> libraryRunfilesPaths = ImmutableList.builder();
    for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
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
        libraryRunfilesPaths.add(
            libraryToLink.getResolvedSymlinkDynamicLibrary().getRunfilesPath());
      }
      if (libraryToLink.getInterfaceLibrary() != null) {
        libraryRunfilesPaths.add(libraryToLink.getInterfaceLibrary().getRunfilesPath());
      }
      if (libraryToLink.getResolvedSymlinkInterfaceLibrary() != null) {
        libraryRunfilesPaths.add(
            libraryToLink.getResolvedSymlinkInterfaceLibrary().getRunfilesPath());
      }
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
  public Depset /*<LibraryToLink>*/ ccLibrariesForStarlark() {
    return Depset.of(Artifact.TYPE, get(ObjcProvider.CC_LIBRARY));
  }

  @Override
  public Depset /*<Linkstamp>*/ linkstampForstarlark() {
    return Depset.of(Linkstamp.TYPE, get(ObjcProvider.LINKSTAMP));
  }

  @Override
  public Depset /*<String>*/ dynamicFrameworkNamesForStarlark() {
    return Depset.of(Depset.ElementType.STRING, dynamicFrameworkNames());
  }

  NestedSet<String> dynamicFrameworkNames() {
    return getFrameworkNames(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<String>*/ dynamicFrameworkPathsForStarlark() {
    return Depset.of(Depset.ElementType.STRING, dynamicFrameworkPaths());
  }

  NestedSet<String> dynamicFrameworkPaths() {
    return getFrameworkPaths(DYNAMIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<String>*/ staticFrameworkNamesForStarlark() {
    return Depset.of(Depset.ElementType.STRING, staticFrameworkNames());
  }

  NestedSet<String> staticFrameworkNames() {
    return getFrameworkNames(STATIC_FRAMEWORK_FILE);
  }

  @Override
  public Depset /*<String>*/ staticFrameworkPathsForStarlark() {
    return Depset.of(Depset.ElementType.STRING, staticFrameworkPaths());
  }

  NestedSet<String> staticFrameworkPaths() {
    return getFrameworkPaths(STATIC_FRAMEWORK_FILE);
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public static class Builder {

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

    ObjcProvider build() {
      ImmutableMap.Builder<Key<?>, NestedSet<?>> propagatedBuilder = new ImmutableMap.Builder<>();
      for (Map.Entry<Key<?>, NestedSetBuilder<?>> typeEntry : items.entrySet()) {
        propagatedBuilder.put(typeEntry.getKey(), typeEntry.getValue().build());
      }
      return new ObjcProvider(
          starlarkSemantics,
          propagatedBuilder.build(),
          strictDependencyIncludes.build(),
          directItems.build());
    }
  }

  /** A builder for this context, specialized for Starlark use. */
  public static final class StarlarkBuilder extends Builder {
    public StarlarkBuilder(StarlarkSemantics semantics) {
      super(semantics);
    }

    /**
     * Add elements in toAdd with the given key from Starlark. An error is thrown if toAdd is not an
     * appropriate Depset.
     */
    void addElementsFromStarlark(Key<?> key, Object starlarkToAdd) throws EvalException {
      NestedSet<?> toAdd = ObjcProviderStarlarkConverters.convertToJava(key, starlarkToAdd);
      if (!KEYS_FOR_DIRECT_ONLY.contains(key)) {
        uncheckedAddTransitive(key, toAdd);
      }

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
          if (!(toAddObject instanceof ObjcProvider)) {
            throw Starlark.errorf(
                AppleStarlarkCommon.BAD_PROVIDERS_ELEM_ERROR, Starlark.type(toAddObject));
          } else {
            ObjcProvider objcProvider = (ObjcProvider) toAddObject;
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
