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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsFunction.BuiltinsFailedException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.RecordingSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A Skyframe function to look up and load a single .bzl module.
 *
 * <p>Given a {@link Label} referencing a .bzl file, attempts to locate the file and load it. The
 * Label must be absolute, and must not reference the special {@code external} package. If loading
 * is successful, returns a {@link BzlLoadValue} that encapsulates the loaded {@link Module} and its
 * transitive digest information. If loading is unsuccessful, throws a {@link
 * BzlLoadFunctionException} that encapsulates the cause of the failure.
 *
 * <p>This Skyframe function supports a special "inlining" mode in which all (indirectly) recursive
 * calls to {@code BzlLoadFunction} are made in the same thread rather than through Skyframe. The
 * inlining mode's entry point is {@link #computeInline}; see that method for more details. Note
 * that it may only be called on an instance of this Skyfunction created by {@link
 * #createForInlining}.
 */
public class BzlLoadFunction implements SkyFunction {

  // We need the RuleClassProvider to 1) create the BazelStarlarkContext for Starlark evaluation,
  // and 2) to pass it to ASTFileLookupFunction's inlining code path.
  // TODO(#11437): The second use can probably go away by refactoring ASTFileLookupFunction to
  // instead accept the set of predeclared bindings. Simplify the code path and then this comment.
  private final RuleClassProvider ruleClassProvider;

  // Used for StarlarkBuiltinsFunction's inlining code path.
  private final PackageFactory packageFactory;

  // Used for BUILD .bzls if injection is disabled.
  // TODO(#11437): Remove once injection is on unconditionally.
  private final StarlarkBuiltinsValue uninjectedStarlarkBuiltins;

  private final ImmutableMap<String, Object> predeclaredForWorkspaceBzl;

  private final ImmutableMap<String, Object> predeclaredForBuiltinsBzl;

  // Handles retrieving ASTFileLookupValues, either by calling Skyframe or by inlining
  // ASTFileLookupFunction; the latter is not to be confused with inlining of BzlLoadFunction. See
  // comment in create() for rationale.
  private final ASTManager astManager;

  // Handles inlining of BzlLoadFunction calls.
  @Nullable private final CachedBzlLoadDataManager cachedBzlLoadDataManager;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private BzlLoadFunction(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      ASTManager astManager,
      @Nullable CachedBzlLoadDataManager cachedBzlLoadDataManager) {
    this.ruleClassProvider = ruleClassProvider;
    this.packageFactory = packageFactory;
    this.uninjectedStarlarkBuiltins =
        StarlarkBuiltinsFunction.createStarlarkBuiltinsValueWithoutInjection(
            ruleClassProvider, packageFactory);
    this.predeclaredForWorkspaceBzl =
        StarlarkBuiltinsFunction.createPredeclaredForWorkspaceBzl(
            ruleClassProvider, packageFactory);
    this.predeclaredForBuiltinsBzl =
        StarlarkBuiltinsFunction.createPredeclaredForBuiltinsBzl(ruleClassProvider);
    this.astManager = astManager;
    this.cachedBzlLoadDataManager = cachedBzlLoadDataManager;
  }

  public static BzlLoadFunction create(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      DigestHashFunction digestHashFunction,
      Cache<Label, ASTFileLookupValue> astFileLookupValueCache) {
    return new BzlLoadFunction(
        ruleClassProvider,
        packageFactory,
        // When we are not inlining BzlLoadValue nodes, there is no need to have separate
        // ASTFileLookupValue nodes for bzl files. Instead we inline ASTFileLookupFunction for a
        // strict memory win, at a small code complexity cost.
        //
        // Detailed explanation:
        // (1) The ASTFileLookupValue node for a bzl file is used only for the computation of
        // that file's BzlLoadValue node. So there's no concern about duplicate work that would
        // otherwise get deduped by Skyframe.
        // (2) ASTFileLookupValue doesn't have an interesting equality relation, so we have no
        // hope of getting any interesting change-pruning of ASTFileLookupValue nodes. If we
        // had an interesting equality relation that was e.g. able to ignore benign
        // whitespace, then there would be a hypothetical benefit to having separate
        // ASTFileLookupValue nodes (e.g. on incremental builds we'd be able to not re-execute
        // top-level code in bzl files if the file were reparsed to an equivalent AST).
        // (3) A ASTFileLookupValue node lets us avoid redoing work on a BzlLoadFunction Skyframe
        // restart, but we can also achieve that result ourselves with a cache that persists between
        // Skyframe restarts.
        //
        // Therefore, ASTFileLookupValue nodes are wasteful from two perspectives:
        // (a) ASTFileLookupValue contains a StarlarkFile, and that business object is really
        // just a temporary thing for bzl execution. Retaining it forever is pure waste.
        // (b) The memory overhead of the extra Skyframe node and edge per bzl file is pure
        // waste.
        new InliningAndCachingASTManager(
            ruleClassProvider, digestHashFunction, astFileLookupValueCache),
        /*cachedBzlLoadDataManager=*/ null);
  }

  public static BzlLoadFunction createForInlining(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      int bzlLoadValueCacheSize) {
    return new BzlLoadFunction(
        ruleClassProvider,
        packageFactory,
        // When we are inlining BzlLoadValue nodes, then we want to have explicit ASTFileLookupValue
        // nodes, since now (1) in the comment above doesn't hold. This way we read and parse each
        // needed bzl file at most once total globally, rather than once per need (in the worst-case
        // of a BzlLoadValue inlining cache miss). This is important in the situation where a bzl
        // file is loaded by a lot of other bzl files or BUILD files.
        RegularSkyframeASTManager.INSTANCE,
        new CachedBzlLoadDataManager(bzlLoadValueCacheSize));
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    BzlLoadValue.Key key = (BzlLoadValue.Key) skyKey.argument();
    try {
      return computeInternal(key, env, /*inliningState=*/ null);
    } catch (InconsistentFilesystemException e) {
      throw new BzlLoadFunctionException(e, Transience.PERSISTENT);
    } catch (BzlLoadFailedException e) {
      throw new BzlLoadFunctionException(e);
    }
  }

  /**
   * Entry point for computing "inline", without any direct or indirect Skyframe calls back into
   * {@link BzlLoadFunction}. (Other Skyframe calls are permitted.)
   *
   * <p><b>USAGE NOTE:</b> This function is intended to be called from {@link PackageFunction} and
   * {@link StarlarkBuiltinsFunction} and probably shouldn't be used anywhere else. If you think you
   * need inline Starlark computation, consult with the Core subteam and check out cl/305127325 for
   * an example of correcting a misuse.
   *
   * <p>Under bzl inlining, there is some calling context that wants to obtain a set of {@link
   * BzlLoadValue}s without Skyframe evaluation. For example, a calling context can be a BUILD file
   * trying to resolve its top-level {@code load()} statements. Although this work proceeds in a
   * single thread, multiple calling contexts may evaluate .bzls in parallel. To avoid redundant
   * work, they share a single (global to this Skyfunction instance) cache in lieu of the regular
   * Skyframe cache. Unlike the regular Skyframe cache, this cache stores only successes.
   *
   * <p>If two calling contexts race to compute the same .bzl, each one will see a different copy of
   * it, and only one will end up in the shared cache. This presents a hazard: Suppose A and B both
   * need foo.bzl, and A needs it twice due to a diamond dependency. If A and B race to compute
   * foo.bzl, but B's computation populates the cache, then when A comes back to resolve it the
   * second time it will observe a different {@code BzlLoadValue}. This leads to incorrect Starlark
   * evaluation since Starlark values may rely on Java object identity (see b/138598337). Even if we
   * weren't concerned about racing, A may also reevaluate previously computed items due to cache
   * evictions.
   *
   * <p>To solve this, we keep a second cache, {@link InliningState#successfulLoads}, that is local
   * to the current calling context, and which never evicts entries. Like the global cache discussed
   * above, this cache stores only successes. This cache is always checked in preference to the
   * shared one; it may deviate from the shared one in some of its entries, but the calling context
   * won't know the difference. (Since bzl inlining is only used for the loading phase, we don't
   * need to worry about Starlark values from different packages interacting.) The cache is stored
   * as part of the {@code inliningState} passed in by the caller; the caller can obtain this object
   * using {@link InliningState#create}.
   *
   * <p>As an aside, note that we can't avoid having {@link InliningState#successfulLoads} by simply
   * naively blocking evaluation of .bzls on retrievals from the shared cache. This is because two
   * contexts could deadlock while trying to evaluate an illegal {@code load()} cycle from opposite
   * ends. It would be possible to construct a waits-for graph and perform cycle detection, or to
   * monitor slow threads and do detection lazily, but these do not address the cache eviction
   * issue. Alternatively, we could make Starlark tolerant of reloading, but that would be
   * tantamount to implementing full Starlark serialization.
   *
   * <p>Since our local {@link InliningState#successfulLoads} stores only successes, a separate
   * concern is that we don't want to unsuccessfully visit the same .bzl more than once in the same
   * context. (A visitation is unsuccessful if it fails due to an error or if it cannot complete
   * because of a missing Skyframe dep.) To address this concern we maintain a separate {@link
   * InliningState#unsuccessfulLoads} set, and use this set to return null instead of duplicating an
   * unsuccessful visitation.
   *
   * @return the requested {@code BzlLoadValue}, or null if there was a missing Skyframe dep, an
   *     unspecified exception in a Skyframe dep request, or if this was a duplicate unsuccessful
   *     visitation
   */
  // TODO(brandjon): Pick one of the nouns "load" and "bzl" and use that term consistently.
  @Nullable
  BzlLoadValue computeInline(BzlLoadValue.Key key, Environment env, InliningState inliningState)
      throws InconsistentFilesystemException, BzlLoadFailedException, InterruptedException {
    // Note to refactorors: No Skyframe calls may be made before the RecordingSkyFunctionEnvironment
    // is set up below in computeInlineForCacheMiss.
    Preconditions.checkNotNull(cachedBzlLoadDataManager);
    CachedBzlLoadData cachedData = computeInlineCachedData(key, env, inliningState);
    return cachedData != null ? cachedData.getValue() : null;
  }

  /**
   * Retrieves or creates the requested {@link CachedBzlLoadData} object for the given bzl, entering
   * it into the local and shared caches. This is the entry point for recursive calls to the inline
   * code path.
   *
   * <p>Skyframe calls made underneath this function will be logged in the resulting {@code
   * CachedBzlLoadData) (or its transitive dependencies). The given Skyframe environment must not
   * be a {@link RecordingSkyFunctionEnvironment}, since that would imply that calls are being
   * logged in both the returned value and the parent value.
   *
   * @return null if there was a missing Skyframe dep, an unspecified exception in a Skyframe dep
   *     request, or if this was a duplicate unsuccessful visitation
   */
  @Nullable
  private CachedBzlLoadData computeInlineCachedData(
      BzlLoadValue.Key key, Environment env, InliningState inliningState)
      throws InconsistentFilesystemException, BzlLoadFailedException, InterruptedException {
    // Note to refactorors: No Skyframe calls may be made before the RecordingSkyFunctionEnvironment
    // is set up below in computeInlineForCacheMiss.

    // Try the caches of successful loads. We must try the thread-local cache before the shared, for
    // consistency purposes (see the javadoc of #computeInline).
    CachedBzlLoadData cachedData = inliningState.successfulLoads.get(key);
    if (cachedData == null) {
      cachedData = cachedBzlLoadDataManager.cache.getIfPresent(key);
      if (cachedData != null) {
        // Found a cache hit from another thread's computation; register the recorded deps as if our
        // thread required them for the current key. Incorporate into visitedBzls any transitive
        // cache hits it does not already contain.
        cachedData.traverse(env::registerDependencies, inliningState.successfulLoads);
      }
    }

    // See if we've already unsuccessfully visited the bzl.
    if (inliningState.unsuccessfulLoads.contains(key)) {
      return null;
    }

    // If we're here, the bzl must have never been visited before in this calling context. Compute
    // it ourselves, updating the other data structures as appropriate.
    if (cachedData == null) {
      try {
        cachedData = computeInlineForCacheMiss(key, env, inliningState);
      } finally {
        if (cachedData != null) {
          inliningState.successfulLoads.put(key, cachedData);
          cachedBzlLoadDataManager.cache.put(key, cachedData);
        } else {
          inliningState.unsuccessfulLoads.add(key);
          // Either propagate an exception or fall through for null return.
        }
      }
    }

    // On success (from cache hit or from scratch), notify the parent CachedBzlLoadData of its new
    // child.
    if (cachedData != null) {
      inliningState.childCachedDataHandler.accept(cachedData);
    }

    return cachedData;
  }

  @Nullable
  private CachedBzlLoadData computeInlineForCacheMiss(
      BzlLoadValue.Key key, Environment env, InliningState inliningState)
      throws InconsistentFilesystemException, BzlLoadFailedException, InterruptedException {
    // We use an instrumented Skyframe env to capture Skyframe deps in the CachedBzlLoadData. This
    // generally includes transitive Skyframe deps, but specifically excludes deps underneath
    // recursively loaded .bzls. We unwrap the instrumented env right before recursively calling
    // back into computeInlineCachedData.
    CachedBzlLoadData.Builder cachedDataBuilder = cachedBzlLoadDataManager.cachedDataBuilder();
    Preconditions.checkState(
        !(env instanceof RecordingSkyFunctionEnvironment),
        "Found nested RecordingSkyFunctionEnvironment but it should have been stripped: %s",
        env);
    RecordingSkyFunctionEnvironment recordingEnv =
        new RecordingSkyFunctionEnvironment(
            env,
            cachedDataBuilder::addDep,
            cachedDataBuilder::addDeps,
            cachedDataBuilder::noteException);

    inliningState.beginLoad(key); // track for cyclic load() detection
    BzlLoadValue value;
    try {
      value =
          computeInternal(
              key,
              recordingEnv,
              inliningState.createChildState(
                  /*childCachedDataHandler=*/ cachedDataBuilder::addTransitiveDeps));
    } finally {
      inliningState.finishLoad(key);
    }
    if (value == null) {
      return null;
    }

    cachedDataBuilder.setValue(value);
    cachedDataBuilder.setKey(key);
    return cachedDataBuilder.build();
  }

  public void resetInliningCache() {
    cachedBzlLoadDataManager.reset();
  }

  /**
   * Retrieves {@link StarlarkBuiltinsValue}, or uses a dummy value if either 1) the builtins
   * mechanism is disabled or 2) the current .bzl is not being loaded for a BUILD file
   *
   * <p>Returns null if a Skyframe restart/error occurred.
   */
  @Nullable
  private StarlarkBuiltinsValue getStarlarkBuiltinsValue(
      BzlLoadValue.Key key,
      Environment env,
      StarlarkSemantics starlarkSemantics,
      @Nullable InliningState inliningState)
      throws BuiltinsFailedException, InconsistentFilesystemException, InterruptedException {
    // Don't request @builtins if we're in workspace evaluation, or already in @builtins evaluation.
    if (!(key instanceof BzlLoadValue.KeyForBuild)
        // TODO(#11437): Remove ability to disable injection by setting flag to empty string.
        || starlarkSemantics.experimentalBuiltinsBzlPath().equals("")) {
      return uninjectedStarlarkBuiltins;
    }
    // Result may be null.
    return inliningState == null
        ? (StarlarkBuiltinsValue)
            env.getValueOrThrow(StarlarkBuiltinsValue.key(), BuiltinsFailedException.class)
        : StarlarkBuiltinsFunction.computeInline(
            StarlarkBuiltinsValue.key(),
            env,
            inliningState,
            /*bzlLoadFunction=*/ this,
            ruleClassProvider,
            packageFactory);
  }

  @Nullable
  private static ContainingPackageLookupValue getContainingPackageLookupValue(
      Environment env, Label label)
      throws InconsistentFilesystemException, BzlLoadFailedException, InterruptedException {
    PathFragment dir = Label.getContainingDirectory(label);
    PackageIdentifier dirId =
        PackageIdentifier.create(label.getPackageIdentifier().getRepository(), dir);
    ContainingPackageLookupValue containingPackageLookupValue;
    try {
      containingPackageLookupValue =
          (ContainingPackageLookupValue)
              env.getValueOrThrow(
                  ContainingPackageLookupValue.key(dirId),
                  BuildFileNotFoundException.class,
                  InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException e) {
      throw BzlLoadFailedException.errorReadingFile(
          label.toPathFragment(), new ErrorReadingStarlarkExtensionException(e));
    }
    if (containingPackageLookupValue == null) {
      return null;
    }
    // Ensure the label doesn't cross package boundaries.
    if (!containingPackageLookupValue.hasContainingPackage()) {
      throw BzlLoadFailedException.noBuildFile(
          label, containingPackageLookupValue.getReasonForNoContainingPackage());
    }
    if (!containingPackageLookupValue
        .getContainingPackageName()
        .equals(label.getPackageIdentifier())) {
      throw BzlLoadFailedException.labelCrossesPackageBoundary(label, containingPackageLookupValue);
    }
    return containingPackageLookupValue;
  }

  /**
   * An opaque object that holds state for the inlining computation initiated by {@link
   * #computeInline}.
   *
   * <p>An original caller of {@code computeInline} (e.g., {@link PackageFunction}) should obtain
   * one of these objects using {@link InliningState#create}. When the same caller makes several
   * calls to {@code computeInline} (e.g., for multiple top-level loads in the same BUILD file), the
   * same object must be passed to each call.
   *
   * <p>When a Skyfunction that is called by {@code BzlLoadFunction}'s inlining code path in turn
   * calls back into {@code computeInline}, it should forward along the same {@code InliningState}
   * that it received. In particular, {@link StarlarkBuiltinsFunction} forwards the inlining state
   * to ensure that 1) the .bzls that get loaded from the {@code @builtins} pseudo-repository are
   * properly recorded as dependencies of all .bzl files that use builtins injection, and 2) the
   * {@code @builtins} .bzls are not reevaluated.
   */
  static class InliningState {
    /**
     * The set of bzls we're currently in the process of loading but haven't fully visited yet. This
     * is used for cycle detection since we don't have the benefit of Skyframe's internal cycle
     * detection. The set must use insertion order for correct error reporting.
     *
     * <p>This is disjoint with {@link #successfulLoads} and {@link #unsuccessfulLoads}.
     *
     * <p>This is local to current calling context. See {@link #computeInline}.
     */
    // Keyed on the SkyKey, not the label, since label could theoretically be ambiguous, even though
    // in practice keys from BUILD / WORKSPACE / @builtins don't call each other. (Not sure if
    // WORKSPACE chunking can cause duplicate labels to appear, but we're robust regardless.)
    private final LinkedHashSet<BzlLoadValue.Key> loadStack;

    /**
     * Cache of bzls that have been fully visited and successfully loaded to a value.
     *
     * <p>This and {@link #unsuccessfulLoads} partition the set of fully visited bzls.
     *
     * <p>This is local to current calling context. See {@link #computeInline}.
     */
    private final Map<BzlLoadValue.Key, CachedBzlLoadData> successfulLoads;

    /**
     * Set of bzls that have been fully visited, but were not successfully loaded to a value.
     *
     * <p>This and {@link #successfulLoads} partition the set of fully visited bzls, and is disjoint
     * with {@link #loadStack}.
     *
     * <p>This is local to current calling context. See {@link #computeInline}.
     */
    private final HashSet<BzlLoadValue.Key> unsuccessfulLoads;

    /** Called when a transitive {@code CachedBzlLoadData} is produced. */
    private final Consumer<CachedBzlLoadData> childCachedDataHandler;

    private InliningState(
        LinkedHashSet<BzlLoadValue.Key> loadStack,
        Map<BzlLoadValue.Key, CachedBzlLoadData> successfulLoads,
        HashSet<BzlLoadValue.Key> unsuccessfulLoads,
        Consumer<CachedBzlLoadData> childCachedDataHandler) {
      this.loadStack = loadStack;
      this.successfulLoads = successfulLoads;
      this.unsuccessfulLoads = unsuccessfulLoads;
      this.childCachedDataHandler = childCachedDataHandler;
    }

    /**
     * Creates an initial {@code InliningState} with no information about previously loaded files
     * (except the shared cache stored in {@link BzlLoadFunction}).
     */
    static InliningState create() {
      return new InliningState(
          /*loadStack=*/ new LinkedHashSet<>(),
          /*successfulLoads=*/ new HashMap<>(),
          /*unsuccessfulLoads=*/ new HashSet<>(),
          // No parent value to mutate
          /*childCachedDataHandler=*/ x -> {});
    }

    private InliningState createChildState(Consumer<CachedBzlLoadData> childCachedDataHandler) {
      return new InliningState(
          loadStack, successfulLoads, unsuccessfulLoads, childCachedDataHandler);
    }

    /** Records entry to a {@code load()}, throwing an exception if a cycle is detected. */
    private void beginLoad(BzlLoadValue.Key key) throws BzlLoadFailedException {
      if (!loadStack.add(key)) {
        ImmutableList<BzlLoadValue.Key> cycle =
            CycleUtils.splitIntoPathAndChain(Predicates.equalTo(key), loadStack).second;
        throw new BzlLoadFailedException("Starlark load cycle: " + cycle);
      }
    }

    /** Records exit from a {@code load()}. */
    private void finishLoad(BzlLoadValue.Key key) throws BzlLoadFailedException {
      Preconditions.checkState(loadStack.remove(key), key);
    }
  }

  // It is vital that we don't return any value if any call to env#getValue(s)OrThrow throws an
  // exception. We are allowed to wrap the thrown exception and rethrow it for any calling functions
  // to handle though.
  @Nullable
  private BzlLoadValue computeInternal(
      BzlLoadValue.Key key, Environment env, @Nullable InliningState inliningState)
      throws InconsistentFilesystemException, BzlLoadFailedException, InterruptedException {
    Label label = key.getLabel();
    PathFragment filePath = label.toPathFragment();

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    StarlarkBuiltinsValue starlarkBuiltinsValue;
    try {
      starlarkBuiltinsValue = getStarlarkBuiltinsValue(key, env, starlarkSemantics, inliningState);
    } catch (BuiltinsFailedException e) {
      throw BzlLoadFailedException.builtinsFailed(label, e);
    }
    if (starlarkBuiltinsValue == null) {
      return null;
    }

    if (getContainingPackageLookupValue(env, label) == null) {
      return null;
    }

    // Load the AST corresponding to this file.
    ASTFileLookupValue astLookupValue;
    try {
      astLookupValue = astManager.getASTFileLookupValue(label, env);
    } catch (ErrorReadingStarlarkExtensionException e) {
      throw BzlLoadFailedException.errorReadingFile(filePath, e);
    }
    if (astLookupValue == null) {
      return null;
    }

    BzlLoadValue result = null;
    try {
      result =
          computeInternalWithAST(
              key,
              filePath,
              starlarkSemantics,
              starlarkBuiltinsValue,
              astLookupValue,
              env,
              inliningState);
    } catch (InconsistentFilesystemException | BzlLoadFailedException | InterruptedException e) {
      astManager.doneWithASTFileLookupValue(label);
      throw e;
    }
    if (result != null) {
      // Result is final (no Skyframe restart), so no further need for the AST value.
      astManager.doneWithASTFileLookupValue(label);
    }
    return result;
  }

  @Nullable
  private BzlLoadValue computeInternalWithAST(
      BzlLoadValue.Key key,
      PathFragment filePath,
      StarlarkSemantics starlarkSemantics,
      StarlarkBuiltinsValue starlarkBuiltinsValue,
      ASTFileLookupValue astLookupValue,
      Environment env,
      @Nullable InliningState inliningState)
      throws InconsistentFilesystemException, BzlLoadFailedException, InterruptedException {
    Label label = key.getLabel();

    if (!astLookupValue.lookupSuccessful()) {
      // Starlark code must exist.
      throw new BzlLoadFailedException(astLookupValue.getError());
    }
    StarlarkFile file = astLookupValue.getAST();
    if (!file.ok()) {
      throw BzlLoadFailedException.starlarkErrors(filePath);
    }

    // Process load statements in .bzl file (recursive .bzl -> .bzl loads),
    // resolving labels relative to the current repo mapping.
    ImmutableMap<RepositoryName, RepositoryName> repoMapping = getRepositoryMapping(key, env);
    if (repoMapping == null) {
      return null;
    }
    List<Pair<String, Label>> loads =
        getLoadLabels(env.getListener(), file, label.getPackageIdentifier(), repoMapping);
    if (loads == null) {
      // malformed load statements
      throw BzlLoadFailedException.starlarkErrors(filePath);
    }

    // Compute Skyframe key for each label in 'loads'.
    List<BzlLoadValue.Key> loadKeys = Lists.newArrayListWithExpectedSize(loads.size());
    for (Pair<String, Label> load : loads) {
      loadKeys.add(key.getKeyForLoad(load.second));
    }

    // Load .bzl modules in parallel.
    // TODO(bazel-team): In case of a failed load(), we should report the location of the load()
    // statement in the requesting file, e.g. using
    // file.getLoadStatements().get(...).getStartLocation(). We should also probably catch and
    // rethrow InconsistentFilesystemException with location info in the non-inlining code path so
    // the error message is the same in both code paths.
    List<BzlLoadValue> bzlLoads =
        inliningState == null
            ? computeBzlLoadsWithSkyframe(env, loadKeys, file)
            : computeBzlLoadsWithInlining(env, loadKeys, file, inliningState);
    if (bzlLoads == null) {
      return null; // Skyframe deps unavailable
    }

    // Process the loaded modules.
    //
    // Compute a digest of the file itself plus the transitive hashes of the modules it directly
    // loads. Loop iteration order matches the source order of load statements.
    Fingerprint fp = new Fingerprint();
    fp.addBytes(astLookupValue.getDigest());
    Map<String, Module> loadedModules = Maps.newLinkedHashMapWithExpectedSize(loads.size());
    for (int i = 0; i < loads.size(); i++) {
      String loadString = loads.get(i).first;
      BzlLoadValue v = bzlLoads.get(i);
      loadedModules.put(loadString, v.getModule()); // dups ok
      fp.addBytes(v.getTransitiveDigest());
    }
    byte[] transitiveDigest = fp.digestAndReset();

    Module module =
        Module.withPredeclared(
            starlarkSemantics, getPredeclaredEnvironment(key, starlarkBuiltinsValue));

    // Record the module's filename, label, digest, and the set of modules it loads,
    // forming a complete representation of the load DAG.
    module.setClientData(
        BazelModuleContext.create(
            label,
            file.getStartLocation().file(),
            ImmutableMap.copyOf(loadedModules),
            transitiveDigest));

    // executeBzlFile may post events to the Environment's handler, but events do not matter when
    // caching BzlLoadValues. Note that executing the module mutates it.
    executeBzlFile(
        file,
        key.getLabel(),
        module,
        loadedModules,
        starlarkSemantics,
        env.getListener(),
        repoMapping);
    return new BzlLoadValue(module, transitiveDigest);
  }

  private static ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping(
      BzlLoadValue.Key key, Environment env) throws InterruptedException {
    Label enclosingFileLabel = key.getLabel();

    ImmutableMap<RepositoryName, RepositoryName> repositoryMapping;
    if (key instanceof BzlLoadValue.KeyForWorkspace) {
      // Still during workspace file evaluation
      BzlLoadValue.KeyForWorkspace keyForWorkspace = (BzlLoadValue.KeyForWorkspace) key;
      if (keyForWorkspace.getWorkspaceChunk() == 0) {
        // There is no previous workspace chunk
        repositoryMapping = ImmutableMap.of();
      } else {
        SkyKey workspaceFileKey =
            WorkspaceFileValue.key(
                keyForWorkspace.getWorkspacePath(), keyForWorkspace.getWorkspaceChunk() - 1);
        WorkspaceFileValue workspaceFileValue = (WorkspaceFileValue) env.getValue(workspaceFileKey);
        // Note: we know for sure that the requested WorkspaceFileValue is fully computed so we do
        // not need to check if it is null
        repositoryMapping =
            workspaceFileValue
                .getRepositoryMapping()
                .getOrDefault(
                    enclosingFileLabel.getPackageIdentifier().getRepository(), ImmutableMap.of());
      }
    } else {
      // We are fully done with workspace evaluation so we should get the mappings from the
      // final RepositoryMappingValue
      PackageIdentifier packageIdentifier = enclosingFileLabel.getPackageIdentifier();
      RepositoryMappingValue repositoryMappingValue =
          (RepositoryMappingValue)
              env.getValue(RepositoryMappingValue.key(packageIdentifier.getRepository()));
      if (repositoryMappingValue == null) {
        return null;
      }
      repositoryMapping = repositoryMappingValue.getRepositoryMapping();
    }
    return repositoryMapping;
  }

  /**
   * Returns a list of pairs mapping each load string in the BUILD or .bzl file to the Label it
   * resolves to. Labels are resolved relative to {@code base}, the file's package. If any load
   * statement is malformed, the function reports one or more errors to the handler and returns
   * null. Order matches the source.
   */
  @Nullable
  static List<Pair<String, Label>> getLoadLabels(
      EventHandler handler,
      StarlarkFile file,
      PackageIdentifier base,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping) {
    Preconditions.checkArgument(!base.getRepository().isDefault());

    // It's redundant that getRelativeWithRemapping needs a Label;
    // a PackageIdentifier should suffice. Make one here.
    Label buildLabel = getBUILDLabel(base);

    boolean ok = true;
    List<Pair<String, Label>> loads = Lists.newArrayList();
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        LoadStatement load = (LoadStatement) stmt;
        String module = load.getImport().getValue();

        // Parse the load statement's module string as a label.
        // It must end in .bzl and not be in package "//external".
        try {
          Label label = buildLabel.getRelativeWithRemapping(module, repoMapping);
          if (!label.getName().endsWith(".bzl")) {
            throw new LabelSyntaxException("The label must reference a file with extension '.bzl'");
          }
          if (label.getPackageIdentifier().equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
            throw new LabelSyntaxException(
                "Starlark files may not be loaded from the //external package");
          }
          loads.add(Pair.of(module, label));
        } catch (LabelSyntaxException ex) {
          handler.handle(
              Event.error(
                  load.getImport().getStartLocation(), "in load statement: " + ex.getMessage()));
          ok = false;
        }
      }
    }
    return ok ? loads : null;
  }

  private static Label getBUILDLabel(PackageIdentifier pkgid) {
    try {
      return Label.create(pkgid, "BUILD");
    } catch (LabelSyntaxException e) {
      // Shouldn't happen; the Label is well-formed by construction.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Computes the BzlLoadValue for all given keys using vanilla Skyframe evaluation, returning
   * {@code null} if Skyframe deps were missing and have been requested.
   */
  @Nullable
  private static List<BzlLoadValue> computeBzlLoadsWithSkyframe(
      Environment env, List<BzlLoadValue.Key> keys, StarlarkFile requestingFile)
      throws BzlLoadFailedException, InterruptedException {
    List<BzlLoadValue> bzlLoads = Lists.newArrayListWithExpectedSize(keys.size());
    Map<SkyKey, ValueOrException<BzlLoadFailedException>> values =
        env.getValuesOrThrow(keys, BzlLoadFailedException.class);
    // Uses same order as load()s in the file. Order matters since we report the first error.
    for (BzlLoadValue.Key key : keys) {
      try {
        bzlLoads.add((BzlLoadValue) values.get(key).get());
      } catch (BzlLoadFailedException exn) {
        throw BzlLoadFailedException.whileLoadingDep(requestingFile.getStartLocation().file(), exn);
      }
    }
    return env.valuesMissing() ? null : bzlLoads;
  }

  /**
   * Computes the BzlLoadValue for all given keys by reusing this instance of the BzlLoadFunction,
   * bypassing traditional Skyframe evaluation.
   *
   * @return null if there was a missing Skyframe dep, an unspecified exception in a Skyframe dep
   *     request, or if this was a duplicate unsuccessful visitation
   */
  @Nullable
  private List<BzlLoadValue> computeBzlLoadsWithInlining(
      Environment env,
      List<BzlLoadValue.Key> keys,
      StarlarkFile requestingFile,
      InliningState inliningState)
      throws InterruptedException, BzlLoadFailedException, InconsistentFilesystemException {
    String filePathForErrors = requestingFile.getStartLocation().file();
    Preconditions.checkState(
        env instanceof RecordingSkyFunctionEnvironment,
        "Expected to be recording dep requests when inlining BzlLoadFunction: %s",
        filePathForErrors);
    Environment strippedEnv = ((RecordingSkyFunctionEnvironment) env).getDelegate();

    List<BzlLoadValue> bzlLoads = Lists.newArrayListWithExpectedSize(keys.size());
    // For determinism's sake while inlining, preserve the first exception and continue to run
    // subsequently listed loads to completion/exception, loading all transitive deps anyway.
    // TODO(brandjon, adgar): What's the deal with determinism w.r.t. InterruptedException, and
    // don't null returns mean the first exception seen isn't deterministic? See also cl/263656490
    // and discussion therein.
    Exception deferredException = null;
    boolean valuesMissing = false;
    // NOTE: Iterating over loads in the order listed in the file.
    for (BzlLoadValue.Key key : keys) {
      CachedBzlLoadData cachedData;
      try {
        cachedData = computeInlineCachedData(key, strippedEnv, inliningState);
      } catch (BzlLoadFailedException e) {
        e = BzlLoadFailedException.whileLoadingDep(filePathForErrors, e);
        deferredException = MoreObjects.firstNonNull(deferredException, e);
        continue;
      } catch (InconsistentFilesystemException e) {
        deferredException = MoreObjects.firstNonNull(deferredException, e);
        continue;
      }
      if (cachedData == null) {
        // A null value for `cachedData` can occur when it (or its transitive loads) has a Skyframe
        // dep that is missing or in error. It can also occur if there's a transitive load on a bzl
        // that was already seen by inliningState and which returned null (note: in this case, it's
        // not necessarily true that there are missing Skyframe deps because this bzl could have
        // already been visited unsuccessfully). In both these cases, we want to continue making our
        // inline calls, so as to maximize the number of dependent (non-inlined) SkyFunctions that
        // are requested and avoid a quadratic number of restarts.
        valuesMissing = true;
      } else {
        bzlLoads.add(cachedData.getValue());
      }
    }
    if (deferredException != null) {
      Throwables.throwIfInstanceOf(deferredException, BzlLoadFailedException.class);
      Throwables.throwIfInstanceOf(deferredException, InconsistentFilesystemException.class);
      throw new IllegalStateException(
          "caught a checked exception of unexpected type", deferredException);
    }
    return valuesMissing ? null : bzlLoads;
  }

  /**
   * Obtains the predeclared environment for a .bzl file, based on the type of file and (if
   * applicable) the injected builtins.
   */
  private ImmutableMap<String, Object> getPredeclaredEnvironment(
      BzlLoadValue.Key key, StarlarkBuiltinsValue starlarkBuiltinsValue) {
    if (key instanceof BzlLoadValue.KeyForBuild) {
      return starlarkBuiltinsValue.predeclaredForBuildBzl;
    } else if (key instanceof BzlLoadValue.KeyForWorkspace) {
      return predeclaredForWorkspaceBzl;
    } else if (key instanceof BzlLoadValue.KeyForBuiltins) {
      return predeclaredForBuiltinsBzl;
    } else {
      throw new AssertionError("Unknown key type: " + key.getClass());
    }
  }

  /** Executes the .bzl file defining the module to be loaded. */
  private void executeBzlFile(
      StarlarkFile file,
      Label label,
      Module module,
      Map<String, Module> loadedModules,
      StarlarkSemantics starlarkSemantics,
      ExtendedEventHandler skyframeEventHandler,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws BzlLoadFailedException, InterruptedException {
    try (Mutability mu = Mutability.create("loading", label)) {
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      thread.setLoader(loadedModules::get);
      StoredEventHandler starlarkEventHandler = new StoredEventHandler();
      thread.setPrintHandler(Event.makeDebugPrintHandler(starlarkEventHandler));
      ruleClassProvider.setStarlarkThreadContext(thread, label, repositoryMapping);
      execAndExport(file, label, starlarkEventHandler, module, thread);

      Event.replayEventsOn(skyframeEventHandler, starlarkEventHandler.getEvents());
      for (Postable post : starlarkEventHandler.getPosts()) {
        skyframeEventHandler.post(post);
      }
      if (starlarkEventHandler.hasErrors()) {
        throw BzlLoadFailedException.errors(label.toPathFragment());
      }
    }
  }

  // Precondition: file is validated and error-free.
  // Precondition: thread has a valid transitiveDigest.
  // TODO(adonovan): executeBzlFile would make a better public API than this function.
  public static void execAndExport(
      StarlarkFile file, Label label, EventHandler handler, Module module, StarlarkThread thread)
      throws InterruptedException {

    // Intercept execution after every assignment at top level
    // and "export" any newly assigned exportable globals.
    // TODO(adonovan): change the semantics; see b/65374671.
    thread.setPostAssignHook(
        (name, value) -> {
          if (value instanceof StarlarkExportable) {
            StarlarkExportable exp = (StarlarkExportable) value;
            if (!exp.isExported()) {
              try {
                exp.export(label, name);
              } catch (EvalException ex) {
                handler.handle(Event.error(ex.getLocation(), ex.getMessage()));
              }
            }
          }
        });

    try {
      EvalUtils.exec(file, module, thread);
    } catch (EvalException ex) {
      handler.handle(Event.error(ex.getLocation(), ex.getMessage()));
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  static final class BzlLoadFailedException extends Exception implements SaneAnalysisException {
    private final Transience transience;

    private BzlLoadFailedException(String errorMessage) {
      super(errorMessage);
      this.transience = Transience.PERSISTENT;
    }

    private BzlLoadFailedException(String errorMessage, Exception cause, Transience transience) {
      super(errorMessage, cause);
      this.transience = transience;
    }

    Transience getTransience() {
      return transience;
    }

    // TODO(bazel-team): This exception should hold a Location of the requesting file's load
    // statement, and code that catches it should use the location in the Event they create.
    static BzlLoadFailedException whileLoadingDep(
        String requestingFile, BzlLoadFailedException cause) {
      // Don't chain exception cause, just incorporate the message with a prefix.
      return new BzlLoadFailedException("in " + requestingFile + ": " + cause.getMessage());
    }

    static BzlLoadFailedException errors(PathFragment file) {
      return new BzlLoadFailedException(String.format("Extension file '%s' has errors", file));
    }

    static BzlLoadFailedException errorReadingFile(
        PathFragment file, ErrorReadingStarlarkExtensionException cause) {
      return new BzlLoadFailedException(
          String.format(
              "Encountered error while reading extension file '%s': %s", file, cause.getMessage()),
          cause,
          cause.getTransience());
    }

    static BzlLoadFailedException noBuildFile(Label file, @Nullable String reason) {
      if (reason != null) {
        return new BzlLoadFailedException(
            String.format("Unable to find package for %s: %s.", file, reason));
      }
      return new BzlLoadFailedException(
          String.format(
              "Every .bzl file must have a corresponding package, but '%s' does not have one."
                  + " Please create a BUILD file in the same or any parent directory. Note that"
                  + " this BUILD file does not need to do anything except exist.",
              file));
    }

    static BzlLoadFailedException labelCrossesPackageBoundary(
        Label label, ContainingPackageLookupValue containingPackageLookupValue) {
      return new BzlLoadFailedException(
          ContainingPackageLookupValue.getErrorMessageForLabelCrossingPackageBoundary(
              // We don't actually know the proper Root to pass in here (since we don't e.g. know
              // the root of the bzl/BUILD file that is trying to load 'label'). Therefore we just
              // pass in the Root of the containing package in order to still get a useful error
              // message for the user.
              containingPackageLookupValue.getContainingPackageRoot(),
              label,
              containingPackageLookupValue));
    }

    static BzlLoadFailedException starlarkErrors(PathFragment file) {
      return new BzlLoadFailedException(String.format("Extension '%s' has errors", file));
    }

    static BzlLoadFailedException builtinsFailed(Label file, BuiltinsFailedException cause) {
      return new BzlLoadFailedException(
          String.format(
              "Internal error while loading Starlark builtins for %s: %s",
              file, cause.getMessage()),
          cause,
          cause.getTransience());
    }
  }

  /**
   * A manager abstracting over the method for obtaining {@code ASTFileLookupValue}s. See comment in
   * {@link #create}.
   */
  private interface ASTManager {
    @Nullable
    ASTFileLookupValue getASTFileLookupValue(Label label, Environment env)
        throws InconsistentFilesystemException, InterruptedException,
            ErrorReadingStarlarkExtensionException;

    void doneWithASTFileLookupValue(Label label);
  }

  /** A manager that obtains ASTs from Skyframe calls. */
  private static class RegularSkyframeASTManager implements ASTManager {
    private static final RegularSkyframeASTManager INSTANCE = new RegularSkyframeASTManager();

    @Nullable
    @Override
    public ASTFileLookupValue getASTFileLookupValue(Label label, Environment env)
        throws InconsistentFilesystemException, InterruptedException,
            ErrorReadingStarlarkExtensionException {
      return (ASTFileLookupValue)
          env.getValueOrThrow(
              ASTFileLookupValue.key(label),
              ErrorReadingStarlarkExtensionException.class,
              InconsistentFilesystemException.class);
    }

    @Override
    public void doneWithASTFileLookupValue(Label label) {}
  }

  /**
   * A manager that obtains ASTs by inlining {@link ASTFileLookupFunction} (not to be confused with
   * inlining of {@code BzlLoadFunction}). Values are cached within the manager and released
   * explicitly by calling {@link #doneWithASTFileLookupValue}.
   */
  private static class InliningAndCachingASTManager implements ASTManager {
    private final RuleClassProvider ruleClassProvider;
    private final DigestHashFunction digestHashFunction;
    // We keep a cache of ASTFileLookupValues that have been computed but whose corresponding
    // BzlLoadValue has not yet completed. This avoids repeating the ASTFileLookupValue work in case
    // of Skyframe restarts. (If we weren't inlining, Skyframe would cache this for us.)
    private final Cache<Label, ASTFileLookupValue> astFileLookupValueCache;

    private InliningAndCachingASTManager(
        RuleClassProvider ruleClassProvider,
        DigestHashFunction digestHashFunction,
        Cache<Label, ASTFileLookupValue> astFileLookupValueCache) {
      this.ruleClassProvider = ruleClassProvider;
      this.digestHashFunction = digestHashFunction;
      this.astFileLookupValueCache = astFileLookupValueCache;
    }

    @Nullable
    @Override
    public ASTFileLookupValue getASTFileLookupValue(Label label, Environment env)
        throws InconsistentFilesystemException, InterruptedException,
            ErrorReadingStarlarkExtensionException {
      ASTFileLookupValue value = astFileLookupValueCache.getIfPresent(label);
      if (value == null) {
        value =
            ASTFileLookupFunction.computeInline(
                ASTFileLookupValue.key(label), env, ruleClassProvider, digestHashFunction);
        if (value != null) {
          astFileLookupValueCache.put(label, value);
        }
      }
      return value;
    }

    @Override
    public void doneWithASTFileLookupValue(Label label) {
      astFileLookupValueCache.invalidate(label);
    }
  }

  /**
   * Per-instance manager for {@link CachedBzlLoadData}, used when {@code BzlLoadFunction} calls are
   * inlined.
   */
  private static class CachedBzlLoadDataManager {
    private final int cacheSize;
    private Cache<BzlLoadValue.Key, CachedBzlLoadData> cache;
    private CachedBzlLoadDataBuilderFactory cachedDataBuilderFactory =
        new CachedBzlLoadDataBuilderFactory();

    private CachedBzlLoadDataManager(int cacheSize) {
      this.cacheSize = cacheSize;
    }

    private CachedBzlLoadData.Builder cachedDataBuilder() {
      return cachedDataBuilderFactory.newCachedBzlLoadDataBuilder();
    }

    private void reset() {
      if (cache != null) {
        logger.atInfo().log("Starlark inlining cache stats from earlier build: " + cache.stats());
      }
      cachedDataBuilderFactory = new CachedBzlLoadDataBuilderFactory();
      Preconditions.checkState(
          cacheSize >= 0, "Expected positive Starlark cache size if caching. %s", cacheSize);
      cache =
          CacheBuilder.newBuilder()
              .concurrencyLevel(BlazeInterners.concurrencyLevel())
              .maximumSize(cacheSize)
              .recordStats()
              .build();
    }
  }

  private static final class BzlLoadFunctionException extends SkyFunctionException {
    private BzlLoadFunctionException(BzlLoadFailedException cause) {
      super(cause, cause.transience);
    }

    private BzlLoadFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }
  }
}
