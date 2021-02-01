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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
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
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.StarlarkLoading;
import com.google.devtools.build.lib.server.FailureDetails.StarlarkLoading.Code;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsFunction.BuiltinsFailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
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
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.LoadStatement;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;
import net.starlark.java.syntax.StringLiteral;

/**
 * A Skyframe function to look up and load a single .bzl module.
 *
 * <p>Given a {@link Label} referencing a .bzl file, attempts to locate the file and load it. The
 * Label must be absolute, and must not reference the special {@code external} package. If loading
 * is successful, returns a {@link BzlLoadValue} that encapsulates the loaded {@link Module} and its
 * transitive digest information. If loading is unsuccessful, throws a {@link
 * BzlLoadFunctionException} that encapsulates the cause of the failure.
 *
 * <p>This Skyframe function supports a special bzl "inlining" mode in which all (indirectly)
 * recursive calls to {@code BzlLoadFunction} are made in the same thread rather than through
 * Skyframe. This inlining mode's entry point is {@link #computeInline}; see that method for more
 * details. Note that it may only be called on an instance of this Skyfunction created by {@link
 * #createForInlining}. Bzl inlining is not to be confused with the separate inlining of {@code
 * BzlCompileFunction}
 */
public class BzlLoadFunction implements SkyFunction {

  // Used for: 1) obtaining a RuleClassProvider to create the BazelStarlarkContext for Starlark
  // evaluation; 2) providing predeclared environments to other Skyfunctions
  // (StarlarkBuiltinsFunction, BzlCompileFunction) when they are inlined and called via a static
  // computeInline() entry point.
  private final PackageFactory packageFactory;

  // Used for determining paths to builtins bzls that live in the workspace.
  private final BlazeDirectories directories;

  // Handles retrieving BzlCompileValues, either by calling Skyframe or by inlining
  // BzlCompileFunction; the latter is not to be confused with inlining of BzlLoadFunction. See
  // comment in create() for rationale.
  private final ValueGetter getter;

  // Handles inlining of BzlLoadFunction calls.
  @Nullable private final CachedBzlLoadDataManager cachedBzlLoadDataManager;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private BzlLoadFunction(
      PackageFactory packageFactory,
      BlazeDirectories directories,
      ValueGetter getter,
      @Nullable CachedBzlLoadDataManager cachedBzlLoadDataManager) {
    this.packageFactory = packageFactory;
    this.directories = directories;
    this.getter = getter;
    this.cachedBzlLoadDataManager = cachedBzlLoadDataManager;
  }

  public static BzlLoadFunction create(
      PackageFactory packageFactory,
      BlazeDirectories directories,
      HashFunction hashFunction,
      Cache<BzlCompileValue.Key, BzlCompileValue> bzlCompileCache) {
    return new BzlLoadFunction(
        packageFactory,
        directories,
        // When we are not inlining BzlLoadValue nodes, there is no need to have separate
        // BzlCompileValue nodes for bzl files. Instead we inline BzlCompileFunction for a
        // strict memory win, at a small code complexity cost.
        //
        // Detailed explanation:
        // (1) The BzlCompileValue node for a bzl file is used only for the computation of
        // that file's BzlLoadValue node. So there's no concern about duplicate work that would
        // otherwise get deduped by Skyframe.
        // (2) BzlCompileValue doesn't have an interesting equality relation, so we have no
        // hope of getting any interesting change-pruning of BzlCompileValue nodes. If we
        // had an interesting equality relation that was e.g. able to ignore benign
        // whitespace, then there would be a hypothetical benefit to having separate
        // BzlCompileValue nodes (e.g. on incremental builds we'd be able to not re-execute
        // top-level code in bzl files if the file were reparsed to an equivalent tree).
        // TODO(adonovan): this will change once it truly compiles the code (soon).
        // (3) A BzlCompileValue node lets us avoid redoing work on a BzlLoadFunction Skyframe
        // restart, but we can also achieve that result ourselves with a cache that persists between
        // Skyframe restarts.
        //
        // Therefore, BzlCompileValue nodes are wasteful from two perspectives:
        // (a) BzlCompileValue contains syntax trees, and that business object is really
        // just a temporary thing for bzl execution. Retaining it forever is pure waste.
        // (b) The memory overhead of the extra Skyframe node and edge per bzl file is pure
        // waste.
        new InliningAndCachingGetter(packageFactory, hashFunction, bzlCompileCache),
        /*cachedBzlLoadDataManager=*/ null);
  }

  public static BzlLoadFunction createForInlining(
      PackageFactory packageFactory, BlazeDirectories directories, int bzlLoadValueCacheSize) {
    return new BzlLoadFunction(
        packageFactory,
        directories,
        // When we are inlining BzlLoadValue nodes, then we want to have explicit BzlCompileValue
        // nodes, since now (1) in the comment above doesn't hold. This way we read and parse each
        // needed bzl file at most once total globally, rather than once per need (in the worst-case
        // of a BzlLoadValue inlining cache miss). This is important in the situation where a bzl
        // file is loaded by a lot of other bzl files or BUILD files.
        RegularSkyframeGetter.INSTANCE,
        new CachedBzlLoadDataManager(bzlLoadValueCacheSize));
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    BzlLoadValue.Key key = (BzlLoadValue.Key) skyKey.argument();
    try {
      return computeInternal(key, env, /*inliningState=*/ null);
    } catch (BzlLoadFailedException e) {
      throw new BzlLoadFunctionException(e);
    }
  }

  /**
   * Entry point for computing "inline", without any direct or indirect Skyframe calls back into
   * {@link BzlLoadFunction}. (Other Skyframe calls are permitted.)
   *
   * <p><b>USAGE NOTES:</b>
   *
   * <ul>
   *   <li>This method is intended to be called from {@link PackageFunction} and {@link
   *       StarlarkBuiltinsFunction} and probably shouldn't be used anywhere else. If you think you
   *       need inline Starlark computation, consult with the Core subteam and check out
   *       cl/305127325 for an example of correcting a misuse.
   *   <li>If this method is used with --keep_going and if Skyframe evaluation will never be
   *       interrupted, then this function ensures that the evaluation graph and any error reported
   *       are deterministic.
   * </ul>
   *
   * <p>Under bzl inlining, there is some calling context that wants to obtain a set of {@link
   * BzlLoadValue}s without Skyframe evaluation. For example, a calling context can be a BUILD file
   * trying to resolve its top-level {@code load} statements. Although this work proceeds in a
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
  BzlLoadValue computeInline(BzlLoadValue.Key key, InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    Preconditions.checkNotNull(cachedBzlLoadDataManager);
    CachedBzlLoadData cachedData = computeInlineCachedData(key, inliningState);
    return cachedData != null ? cachedData.getValue() : null;
  }

  /**
   * Retrieves or creates the requested {@link CachedBzlLoadData} object for the given bzl, entering
   * it into the local and shared caches. This is the entry point for recursive calls to the inline
   * code path.
   *
   * @return null if there was a missing Skyframe dep, an unspecified exception in a Skyframe dep
   *     request, or if this was a duplicate unsuccessful visitation
   */
  @Nullable
  private CachedBzlLoadData computeInlineCachedData(
      BzlLoadValue.Key key, InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    // Try the caches of successful loads. We must try the thread-local cache before the shared, for
    // consistency purposes (see the javadoc of #computeInline).
    CachedBzlLoadData cachedData = inliningState.successfulLoads.get(key);
    if (cachedData == null) {
      cachedData = cachedBzlLoadDataManager.cache.getIfPresent(key);
      if (cachedData != null) {
        // Found a cache hit from another thread's computation. Register the cache hit's recorded
        // deps as if we had requested them directly in the unwrapped environment. We do this for
        // the unwrapped environment, not the recording environment, because there's no need to
        // embed one CachedBzlLoadData's metadata inside another; the dependency relationship will
        // still be accurately reflected in the cache by the call to addTransitiveDeps() via
        // childCachedDataHandler at the bottom of this function.
        //
        // Also incorporate into successfulLoads any transitive cache hits that it does not already
        // contains.
        cachedData.traverse(
            inliningState.recordingEnv.getDelegate()::registerDependencies,
            inliningState.successfulLoads);
      }
    }

    // See if we've already unsuccessfully visited the bzl. "Unsuccessfully" includes getting null
    // for a missing Skyframe dep; the top-level caller will use a fresh InliningState when it does
    // its Skyframe restart.
    if (inliningState.unsuccessfulLoads.contains(key)) {
      return null;
    }

    // If we're here, the bzl must have never been visited before in this calling context. Compute
    // it ourselves, updating the other data structures as appropriate.
    if (cachedData == null) {
      try {
        cachedData = computeInlineForCacheMiss(key, inliningState);
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
      BzlLoadValue.Key key, InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    // We use an instrumented Skyframe env to capture Skyframe deps in the CachedBzlLoadData (see
    // InliningState#recordingEnv). This generally includes transitive Skyframe deps, but
    // specifically excludes deps underneath recursively loaded .bzls. In this way, the
    // CachedBzlLoadData objects form a DAG that mirrors the bzl load graph: Each node still reaches
    // *all* the transitive skyframe deps needed for its computation, but the bzl-level granularity
    // allows for sharing of cached results for portions of the bzl load graph.
    //
    // Here we are at the boundary between one CachedBzlLoadData and the next. createChildState()
    // unwraps the old recording env and starts a new one for a new node.

    InliningState childState = inliningState.createChildState(cachedBzlLoadDataManager);
    childState.beginLoad(key); // track for cyclic load() detection
    BzlLoadValue value;
    try {
      value = computeInternal(key, childState.recordingEnv, childState);
    } finally {
      childState.finishLoad(key);
    }
    if (value == null) {
      return null;
    }

    return childState.buildCachedData(key, value);
  }

  public void resetInliningCache() {
    cachedBzlLoadDataManager.reset();
  }

  /**
   * An opaque object that holds state for the bzl inlining computation initiated by {@link
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
   * to ensure that 1) the .bzls that get loaded from the {@code @_builtins} pseudo-repository are
   * properly recorded as dependencies of all .bzl files that use builtins injection, and 2) the
   * builtins .bzls are not reevaluated.
   */
  // TODO(brandjon): Consider making this even more opaque and encapsulating more of the details of
  // inlining. E.g., merge beginLoad/finishLoad with child state tracking, and encapsulate
  // management of [un]successfulLoads.
  static class InliningState {

    /**
     * The Skyframe environment, instrumented to record dependencies inside CachedBzlLoadData
     * objects. A new CachedBzlLoadData, and therefore a new recording environment, is started in
     * each call to computeInlineForCacheMiss(). The initial InliningState's recording environment
     * doesn't instrument anything since it represents the piece of the work that will not be saved
     * in any CachedBzlLoadData.
     */
    private final RecordingSkyFunctionEnvironment recordingEnv;

    /**
     * The builder of the CachedBzlLoadData node that we are currently working on, if any. Null iff
     * we're the initial InliningState, where recordingEnv doesn't instrument anything.
     */
    private final CachedBzlLoadData.Builder cachedDataBuilder;

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
    // in practice keys from BUILD / WORKSPACE / builtins don't call each other. (Not sure if
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
        RecordingSkyFunctionEnvironment recordingEnv,
        CachedBzlLoadData.Builder cachedDataBuilder,
        LinkedHashSet<BzlLoadValue.Key> loadStack,
        Map<BzlLoadValue.Key, CachedBzlLoadData> successfulLoads,
        HashSet<BzlLoadValue.Key> unsuccessfulLoads,
        Consumer<CachedBzlLoadData> childCachedDataHandler) {
      this.recordingEnv = recordingEnv;
      this.cachedDataBuilder = cachedDataBuilder;
      this.loadStack = loadStack;
      this.successfulLoads = successfulLoads;
      this.unsuccessfulLoads = unsuccessfulLoads;
      this.childCachedDataHandler = childCachedDataHandler;
    }

    /**
     * Creates an initial {@code InliningState} with no information about previously loaded files
     * (except the shared cache stored in {@link BzlLoadFunction}).
     */
    static InliningState create(Environment env) {
      return new InliningState(
          new RecordingSkyFunctionEnvironment(env, x -> {}, x -> {}, x -> {}),
          /*cachedDataBuilder=*/ null,
          /*loadStack=*/ new LinkedHashSet<>(),
          /*successfulLoads=*/ new HashMap<>(),
          /*unsuccessfulLoads=*/ new HashSet<>(),
          // No parent value to mutate
          /*childCachedDataHandler=*/ x -> {});
    }

    /**
     * Creates another InliningState from this one, but with the recording Skyframe environment set
     * up to log dependency metadata into a CachedBzlLoadData node that is a child of this
     * InliningState's node.
     */
    private InliningState createChildState(CachedBzlLoadDataManager cachedBzlLoadDataManager) {
      CachedBzlLoadData.Builder newBuilder = cachedBzlLoadDataManager.cachedDataBuilder();
      RecordingSkyFunctionEnvironment newRecordingEnv =
          new RecordingSkyFunctionEnvironment(
              recordingEnv.getDelegate(),
              newBuilder::addDep,
              newBuilder::addDeps,
              newBuilder::noteException);
      return new InliningState(
          newRecordingEnv,
          newBuilder,
          loadStack,
          successfulLoads,
          unsuccessfulLoads,
          newBuilder::addTransitiveDeps);
    }

    /**
     * Finishes construction of the current CachedBzlLoadData node. This InliningState object should
     * not be used after calling this method.
     */
    private CachedBzlLoadData buildCachedData(BzlLoadValue.Key key, BzlLoadValue value) {
      cachedDataBuilder.setValue(value);
      cachedDataBuilder.setKey(key);
      return cachedDataBuilder.build();
    }

    /** Records entry to a {@code load()}, throwing an exception if a cycle is detected. */
    private void beginLoad(BzlLoadValue.Key key) throws BzlLoadFailedException {
      if (!loadStack.add(key)) {
        ImmutableList<BzlLoadValue.Key> cycle =
            CycleUtils.splitIntoPathAndChain(Predicates.equalTo(key), loadStack).second;
        throw new BzlLoadFailedException("Starlark load cycle: " + cycle, Code.CYCLE);
      }
    }

    /** Records exit from a {@code load()}. */
    private void finishLoad(BzlLoadValue.Key key) throws BzlLoadFailedException {
      Preconditions.checkState(loadStack.remove(key), key);
    }

    /** Retrieves the Skyframe environment to use to do work under this InliningState. */
    Environment getEnvironment() {
      return recordingEnv;
    }
  }

  /**
   * Entry point for compute logic that's common to both (bzl) inlining and non-inlining code paths.
   */
  // It is vital that we don't return any value if any call to env#getValue(s)OrThrow throws an
  // exception. We are allowed to wrap the thrown exception and rethrow it for any calling functions
  // to handle though.
  @Nullable
  private BzlLoadValue computeInternal(
      BzlLoadValue.Key key, Environment env, @Nullable InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    Label label = key.getLabel();
    PathFragment filePath = label.toPathFragment();

    StarlarkBuiltinsValue builtins = getBuiltins(key, env, inliningState);
    if (builtins == null) {
      return null;
    }

    BzlCompileValue.Key compileKey =
        validatePackageAndGetCompileKey(
            key,
            env,
            builtins.starlarkSemantics.get(BuildLanguageOptions.EXPERIMENTAL_BUILTINS_BZL_PATH));
    if (compileKey == null) {
      return null;
    }
    BzlCompileValue compileValue;
    try {
      compileValue = getter.getBzlCompileValue(compileKey, env);
    } catch (BzlCompileFunction.FailedIOException e) {
      throw BzlLoadFailedException.errorReadingBzl(filePath, e);
    }
    if (compileValue == null) {
      return null;
    }

    BzlLoadValue result = null;
    // Release the compiled bzl iff the value gets completely evaluated (to either error or non-null
    // result).
    boolean completed = true;
    try {
      result = computeInternalWithCompiledBzl(key, compileValue, builtins, env, inliningState);
      completed = result != null;
    } finally {
      if (completed) { // only false on unexceptional null result
        getter.doneWithBzlCompileValue(compileKey);
      }
    }
    return result;
  }

  /**
   * Obtain a suitable StarlarkBuiltinsValue.
   *
   * <p>For BUILD-loaded .bzl files, this is a real builtins value, obtained using either Skyframe
   * or inlining of StarlarkBuiltinsFunction (depending on whether {@code inliningState} is
   * non-null). The returned value includes the StarlarkSemantics.
   *
   * <p>For other .bzl files, the builtins computation is not needed and would create a Skyframe
   * cycle if requested, so we instead return an empty builtins value that just wraps the
   * StarlarkSemantics. (NB: In the case of WORKSPACE-loaded .bzl files, the cycle goes through the
   * repository remapping value. It's possible this could be avoided if we ever wanted to make this
   * kind of .bzl file use builtins injection.)
   */
  @Nullable
  private StarlarkBuiltinsValue getBuiltins(
      BzlLoadValue.Key key, Environment env, @Nullable InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    if (!(key instanceof BzlLoadValue.KeyForBuild)) {
      StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
      if (starlarkSemantics == null) {
        return null;
      }
      return StarlarkBuiltinsValue.createEmpty(starlarkSemantics);
    }
    try {
      if (inliningState == null) {
        return (StarlarkBuiltinsValue)
            env.getValueOrThrow(StarlarkBuiltinsValue.key(), BuiltinsFailedException.class);
      } else {
        return StarlarkBuiltinsFunction.computeInline(
            StarlarkBuiltinsValue.key(), inliningState, packageFactory, /*bzlLoadFunction=*/ this);
      }
    } catch (BuiltinsFailedException e) {
      throw BzlLoadFailedException.builtinsFailed(key.getLabel(), e);
    }
  }

  /**
   * Given a bzl key, validates that the corresponding package exists (if required) and returns the
   * associated compile key based on the package's root. Returns null for a missing Skyframe dep or
   * unspecified exception.
   *
   * <p>.bzl files are not necessarily targets, because they can be loaded by BUILD and other .bzl
   * files without ever being declared in a BUILD file. However, .bzl files are still identified by
   * a label in the same way that file targets are. In particular, it is illegal to refer to a .bzl
   * file using a label whose package part is not the .bzl file's innermost containing package. For
   * example, if pkg and pkg/subpkg have BUILD files but not pkg/subdir, then {@code
   * pkg/subdir:foo.bzl} and {@code pkg:subpkg/foo.bzl} are disallowed.
   *
   * <p>In the case of builtins .bzl files, all labels are written as if the pseudo-repo constitutes
   * one big package, e.g {@code @builtins//:some/path/foo.bzl}, but no BUILD file need exist. The
   * compile key's root is determined by {@code --experimental_builtins_bzl_path} (passed as {@code
   * builtinsBzlPath}) instead of by package lookup.
   */
  @Nullable
  private BzlCompileValue.Key validatePackageAndGetCompileKey(
      BzlLoadValue.Key key, Environment env, String builtinsBzlPath)
      throws BzlLoadFailedException, InterruptedException {
    Label label = key.getLabel();

    // Bypass package lookup entirely if builtins.
    if (key.isBuiltins()) {
      if (!label.getPackageName().isEmpty()) {
        throw BzlLoadFailedException.noBuildFile(label, "@_builtins cannot have subpackages");
      }
      return key.getCompileKey(getBuiltinsRoot(builtinsBzlPath));
    }

    // Do package lookup.
    PathFragment dir = Label.getContainingDirectory(label);
    PackageIdentifier dirId = PackageIdentifier.create(label.getRepository(), dir);
    ContainingPackageLookupValue packageLookup;
    try {
      packageLookup =
          (ContainingPackageLookupValue)
              env.getValueOrThrow(
                  ContainingPackageLookupValue.key(dirId),
                  BuildFileNotFoundException.class,
                  InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException | InconsistentFilesystemException e) {
      throw BzlLoadFailedException.errorFindingContainingPackage(label.toPathFragment(), e);
    }
    if (packageLookup == null) {
      return null;
    }

    // Resolve to compile key or error.
    BzlCompileValue.Key compileKey;
    boolean packageOk =
        packageLookup.hasContainingPackage()
            && packageLookup.getContainingPackageName().equals(label.getPackageIdentifier());
    if (key.isBuildPrelude() && !packageOk) {
      // Ignore the prelude, its package doesn't exist.
      compileKey = BzlCompileValue.EMPTY_PRELUDE_KEY;
    } else {
      if (packageOk) {
        compileKey = key.getCompileKey(packageLookup.getContainingPackageRoot());
      } else {
        if (!packageLookup.hasContainingPackage()) {
          throw BzlLoadFailedException.noBuildFile(
              label, packageLookup.getReasonForNoContainingPackage());
        } else {
          throw BzlLoadFailedException.labelCrossesPackageBoundary(label, packageLookup);
        }
      }
    }
    return compileKey;
  }

  private Root getBuiltinsRoot(String builtinsBzlPath) {
    // TODO(#11437): Remove once injection can't be disabled.
    if (builtinsBzlPath.isEmpty()) {
      throw new IllegalStateException("Requested builtins root, but injection is disabled");
    }

    Path path;
    if (builtinsBzlPath.equals("%bundled%")) {
      // May be null in tests, but in that case the builtins path shouldn't be set to %bundled%.
      path =
          Preconditions.checkNotNull(
              packageFactory.getRuleClassProvider().getBuiltinsBzlRoot(),
              "rule class provider does not specify a builtins root; either call"
                  + " setBuiltinsBzlZipResource() or else set --experimental_builtins_bzl_path to"
                  + " a root");
    } else if (builtinsBzlPath.equals("%workspace%")) {
      String packagePath =
          packageFactory.getRuleClassProvider().getBuiltinsBzlPackagePathInSource();
      path = directories.getWorkspace().getRelative(packagePath);
    } else {
      path = directories.getWorkspace().getRelative(builtinsBzlPath);
    }
    // TODO(#11437): Should we consider interning these roots?
    return Root.fromPath(path);
  }

  /**
   * Compute logic for once the compiled .bzl has been fetched and confirmed to exist (though it may
   * have Starlark errors).
   */
  @Nullable
  private BzlLoadValue computeInternalWithCompiledBzl(
      BzlLoadValue.Key key,
      BzlCompileValue compileValue,
      StarlarkBuiltinsValue builtins,
      Environment env,
      @Nullable InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    // Ensure the .bzl exists and passes static checks (parsing, resolving).
    // (A missing prelude file still returns a valid but empty BzlCompileValue.)
    if (!compileValue.lookupSuccessful()) {
      throw new BzlLoadFailedException(compileValue.getError(), Code.COMPILE_ERROR);
    }
    Program prog = compileValue.getProgram();

    // Determine dependency BzlLoadValue keys for the load statements in this bzl.
    // Labels are resolved relative to the current repo mapping.
    ImmutableMap<RepositoryName, RepositoryName> repoMapping = getRepositoryMapping(key, env);
    if (repoMapping == null) {
      return null;
    }
    ImmutableList<Pair<String, Location>> programLoads = getLoadsFromProgram(prog);
    ImmutableList<Label> loadLabels =
        getLoadLabels(
            env.getListener(), programLoads, key.getLabel().getPackageIdentifier(), repoMapping);
    if (loadLabels == null) {
      throw new BzlLoadFailedException(
          String.format(
              "module '%s'%s has invalid load statements",
              key.getLabel().toPathFragment(),
              StarlarkBuiltinsValue.isBuiltinsRepo(key.getLabel().getRepository())
                  ? " (internal)"
                  : ""),
          Code.PARSE_ERROR);
    }
    ImmutableList.Builder<BzlLoadValue.Key> loadKeys =
        ImmutableList.builderWithExpectedSize(loadLabels.size());
    for (Label loadLabel : loadLabels) {
      loadKeys.add(key.getKeyForLoad(loadLabel));
    }

    // Load .bzl modules.
    // When not using bzl inlining, this is done in parallel for all loads.
    List<BzlLoadValue> loadValues =
        inliningState == null
            ? computeBzlLoadsWithSkyframe(env, loadKeys.build(), programLoads)
            : computeBzlLoadsWithInlining(env, loadKeys.build(), programLoads, inliningState);
    if (loadValues == null) {
      return null; // Skyframe deps unavailable
    }

    // Accumulate a transitive digest of the bzl file, the digests of its direct loads, and the
    // digest of the @_builtins pseudo-repository (if applicable).
    Fingerprint fp = new Fingerprint();
    fp.addBytes(compileValue.getDigest());

    // Populate the load map and add transitive digests to the fingerprint.
    Map<String, Module> loadMap = Maps.newLinkedHashMapWithExpectedSize(programLoads.size());
    int i = 0;
    for (Pair<String, Location> load : programLoads) {
      BzlLoadValue v = loadValues.get(i++);
      loadMap.put(load.first, v.getModule()); // dups ok
      fp.addBytes(v.getTransitiveDigest());
    }

    // Retrieve predeclared symbols and complete the digest computation.
    ImmutableMap<String, Object> predeclared =
        getAndDigestPredeclaredEnvironment(key, builtins, fp);
    if (predeclared == null) {
      return null;
    }
    byte[] transitiveDigest = fp.digestAndReset();

    // Construct the initial Starlark module, then execute the code and return the result.
    // The additional information in BazelModuleContext reifies the load DAG.
    // The module must match the environment used to compile the .bzl file.
    Module module = Module.withPredeclared(builtins.starlarkSemantics, predeclared);
    module.setClientData(
        BazelModuleContext.create(
            key.getLabel(), prog.getFilename(), ImmutableMap.copyOf(loadMap), transitiveDigest));

    // executeBzlFile may post events to the Environment's handler, but events do not matter when
    // caching BzlLoadValues. Note that executing the code mutates the module.
    executeBzlFile(
        prog,
        key.getLabel(),
        module,
        loadMap,
        builtins.starlarkSemantics,
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
                .getOrDefault(enclosingFileLabel.getRepository(), ImmutableMap.of());
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
   * Given a list of {@code load("module")} strings and their locations, in source order, returns a
   * corresponding list of Labels they each resolve to. Labels are resolved relative to {@code
   * base}, the file's package. If any label is malformed, the function reports one or more errors
   * to the handler and returns null.
   */
  @Nullable
  static ImmutableList<Label> getLoadLabels(
      EventHandler handler,
      ImmutableList<Pair<String, Location>> loads,
      PackageIdentifier base,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping) {
    Preconditions.checkArgument(!base.getRepository().isDefault());

    // It's redundant that getRelativeWithRemapping needs a Label;
    // a PackageIdentifier should suffice. Make one here.
    Label buildLabel = getBUILDLabel(base);

    boolean ok = true;

    ImmutableList.Builder<Label> loadLabels = ImmutableList.builderWithExpectedSize(loads.size());
    for (Pair<String, Location> load : loads) {
      // Parse the load statement's module string as a label.
      // It must end in .bzl and not be in package "//external".
      try {
        Label label = buildLabel.getRelativeWithRemapping(load.first, repoMapping);
        if (!label.getName().endsWith(".bzl")) {
          throw new LabelSyntaxException("The label must reference a file with extension '.bzl'");
        }
        if (label.getPackageIdentifier().equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
          throw new LabelSyntaxException(
              "Starlark files may not be loaded from the //external package");
        }
        if (StarlarkBuiltinsValue.isBuiltinsRepo(base.getRepository())
            && !StarlarkBuiltinsValue.isBuiltinsRepo(label.getRepository())) {
          throw new LabelSyntaxException(
              ".bzl files in @_builtins cannot load from outside of @_builtins");
        }
        loadLabels.add(label);
      } catch (LabelSyntaxException ex) {
        handler.handle(Event.error(load.second, "in load statement: " + ex.getMessage()));
        ok = false;
      }
    }
    return ok ? loadLabels.build() : null;
  }

  /** Extracts load statements from compiled program (see {@link #getLoadLabels}). */
  static ImmutableList<Pair<String, Location>> getLoadsFromProgram(Program prog) {
    int n = prog.getLoads().size();
    ImmutableList.Builder<Pair<String, Location>> loads = ImmutableList.builderWithExpectedSize(n);
    for (int i = 0; i < n; i++) {
      loads.add(Pair.of(prog.getLoads().get(i), prog.getLoadLocation(i)));
    }
    return loads.build();
  }

  /** Extracts load statements from file syntax (see {@link #getLoadLabels}). */
  static ImmutableList<Pair<String, Location>> getLoadsFromStarlarkFile(StarlarkFile file) {
    ImmutableList.Builder<Pair<String, Location>> loads = ImmutableList.builder();
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        StringLiteral module = ((LoadStatement) stmt).getImport();
        loads.add(Pair.of(module.getValue(), module.getStartLocation()));
      }
    }
    return loads.build();
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
   * Computes the BzlLoadValue for all given .bzl load keys using ordinary Skyframe evaluation,
   * returning {@code null} if Skyframe deps were missing and have been requested. {@code
   * programLoads} provides the locations of the load statements in source order, for error
   * reporting.
   */
  @Nullable
  private static List<BzlLoadValue> computeBzlLoadsWithSkyframe(
      Environment env, List<BzlLoadValue.Key> keys, List<Pair<String, Location>> programLoads)
      throws BzlLoadFailedException, InterruptedException {
    List<BzlLoadValue> bzlLoads = Lists.newArrayListWithExpectedSize(keys.size());
    Map<SkyKey, ValueOrException<BzlLoadFailedException>> values =
        env.getValuesOrThrow(keys, BzlLoadFailedException.class);
    // Process loads (and report first error) in source order.
    for (int i = 0; i < keys.size(); i++) {
      try {
        bzlLoads.add((BzlLoadValue) values.get(keys.get(i)).get());
      } catch (BzlLoadFailedException ex) {
        throw BzlLoadFailedException.whileLoadingDep(programLoads.get(i).second, ex);
      }
    }
    return env.valuesMissing() ? null : bzlLoads;
  }

  /**
   * Computes the BzlLoadValue for all given keys by reusing this instance of the BzlLoadFunction,
   * bypassing traditional Skyframe evaluation. {@code programLoads} provides the locations of the
   * load statements in source order, for error reporting.
   *
   * @return null if there was a missing Skyframe dep, an unspecified exception in a Skyframe dep
   *     request, or if this was a duplicate unsuccessful visitation
   */
  @Nullable
  private List<BzlLoadValue> computeBzlLoadsWithInlining(
      Environment env,
      List<BzlLoadValue.Key> keys,
      List<Pair<String, Location>> programLoads,
      InliningState inliningState)
      throws BzlLoadFailedException, InterruptedException {
    Preconditions.checkState(env == inliningState.recordingEnv);

    List<BzlLoadValue> bzlLoads = Lists.newArrayListWithExpectedSize(keys.size());
    // For the sake of ensuring the graph structure is deterministic, we need to request all of our
    // deps, even if some of them yield errors. The first exception that is seen gets deferred, to
    // be raised after the loop. All other exceptions are swallowed.
    //
    // To see how immediately returning the first error leads to non-determinism, consider the case
    // of two dependencies A and B, where A is in error and appears in a load statement above B.
    // If A has completed at the time we request it, and if we were to immediately propagate that
    // error, we never request B. On the other hand, if A is missing (null return), we do request B
    // in the meantime for the sake of parallelism.
    //
    // This approach assumes --keep_going; determinism is not guaranteed otherwise. It also assumes
    // InterruptedException does not occur, since we don't catch and defer it.
    BzlLoadFailedException deferredException = null;
    boolean valuesMissing = false;
    for (int i = 0; i < keys.size(); i++) {
      CachedBzlLoadData cachedData;
      try {
        cachedData = computeInlineCachedData(keys.get(i), inliningState);
      } catch (BzlLoadFailedException e) {
        if (deferredException == null) {
          deferredException = BzlLoadFailedException.whileLoadingDep(programLoads.get(i).second, e);
        }
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
      throw deferredException;
    }
    return valuesMissing ? null : bzlLoads;
  }

  /**
   * Obtains the predeclared environment for a .bzl file, based on the type of .bzl and (if
   * applicable) the injected builtins.
   *
   * <p>Returns null if there was a missing Skyframe dep or unspecified exception.
   *
   * <p>In the case that injected builtins are used, updates the given fingerprint with the digest
   * of the {@code @_builtins} pseudo-repository.
   */
  @Nullable
  private ImmutableMap<String, Object> getAndDigestPredeclaredEnvironment(
      BzlLoadValue.Key key, StarlarkBuiltinsValue builtins, Fingerprint fp)
      throws BzlLoadFailedException, InterruptedException {
    BazelStarlarkEnvironment starlarkEnv = packageFactory.getBazelStarlarkEnvironment();
    if (key instanceof BzlLoadValue.KeyForBuild) {
      // TODO(#11437): Remove ability to disable injection by setting flag to empty string.
      if (builtins
          .starlarkSemantics
          .get(BuildLanguageOptions.EXPERIMENTAL_BUILTINS_BZL_PATH)
          .isEmpty()) {
        return starlarkEnv.getUninjectedBuildBzlEnv();
      }
      fp.addBytes(builtins.transitiveDigest);
      return builtins.predeclaredForBuildBzl;
    } else if (key instanceof BzlLoadValue.KeyForWorkspace) {
      return starlarkEnv.getWorkspaceBzlEnv();
    } else if (key instanceof BzlLoadValue.KeyForBuiltins) {
      return starlarkEnv.getBuiltinsBzlEnv();
    } else {
      throw new AssertionError("Unknown key type: " + key.getClass());
    }
  }

  /** Executes the compiled .bzl file defining the module to be loaded. */
  private void executeBzlFile(
      Program prog,
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
      packageFactory
          .getRuleClassProvider()
          .setStarlarkThreadContext(thread, label, repositoryMapping);
      execAndExport(prog, label, starlarkEventHandler, module, thread);

      Event.replayEventsOn(skyframeEventHandler, starlarkEventHandler.getEvents());
      for (Postable post : starlarkEventHandler.getPosts()) {
        skyframeEventHandler.post(post);
      }
      if (starlarkEventHandler.hasErrors()) {
        throw BzlLoadFailedException.executionFailed(label);
      }
    }
  }

  // Precondition: thread has a valid transitiveDigest.
  // TODO(adonovan): executeBzlFile would make a better public API than this function.
  public static void execAndExport(
      Program prog, Label label, EventHandler handler, Module module, StarlarkThread thread)
      throws InterruptedException {

    // Intercept execution after every assignment at top level
    // and "export" any newly assigned exportable globals.
    // TODO(adonovan): change the semantics; see b/65374671.
    thread.setPostAssignHook(
        (name, value) -> {
          if (value instanceof StarlarkExportable) {
            StarlarkExportable exp = (StarlarkExportable) value;
            if (!exp.isExported()) {
              exp.export(handler, label, name);
            }
          }
        });

    try {
      Starlark.execFileProgram(prog, module, thread);
    } catch (EvalException ex) {
      handler.handle(Event.error(null, ex.getMessageWithStack()));
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * A manager abstracting over the method for obtaining {@code BzlCompileValue}s. See comment in
   * {@link #create}.
   */
  private interface ValueGetter {
    @Nullable
    BzlCompileValue getBzlCompileValue(BzlCompileValue.Key key, Environment env)
        throws BzlCompileFunction.FailedIOException, InterruptedException;

    void doneWithBzlCompileValue(BzlCompileValue.Key key);
  }

  /** A manager that obtains compiled .bzl files from Skyframe calls. */
  private static class RegularSkyframeGetter implements ValueGetter {
    private static final RegularSkyframeGetter INSTANCE = new RegularSkyframeGetter();

    @Nullable
    @Override
    public BzlCompileValue getBzlCompileValue(BzlCompileValue.Key key, Environment env)
        throws BzlCompileFunction.FailedIOException, InterruptedException {
      return (BzlCompileValue) env.getValueOrThrow(key, BzlCompileFunction.FailedIOException.class);
    }

    @Override
    public void doneWithBzlCompileValue(BzlCompileValue.Key key) {}
  }

  /**
   * A manager that obtains compiled .bzls by inlining {@link BzlCompileFunction} (not to be
   * confused with inlining of {@code BzlLoadFunction}). Values are cached within the manager and
   * released explicitly by calling {@link #doneWithBzlCompileValue}.
   */
  private static class InliningAndCachingGetter implements ValueGetter {
    private final PackageFactory packageFactory;
    private final HashFunction hashFunction;
    // We keep a cache of BzlCompileValues that have been computed but whose corresponding
    // BzlLoadValue has not yet completed. This avoids repeating the BzlCompileValue work in case
    // of Skyframe restarts. (If we weren't inlining, Skyframe would cache this for us.)
    private final Cache<BzlCompileValue.Key, BzlCompileValue> bzlCompileCache;

    private InliningAndCachingGetter(
        PackageFactory packageFactory,
        HashFunction hashFunction,
        Cache<BzlCompileValue.Key, BzlCompileValue> bzlCompileCache) {
      this.packageFactory = packageFactory;
      this.hashFunction = hashFunction;
      this.bzlCompileCache = bzlCompileCache;
    }

    @Nullable
    @Override
    public BzlCompileValue getBzlCompileValue(BzlCompileValue.Key key, Environment env)
        throws BzlCompileFunction.FailedIOException, InterruptedException {
      BzlCompileValue value = bzlCompileCache.getIfPresent(key);
      if (value == null) {
        value = BzlCompileFunction.computeInline(key, env, packageFactory, hashFunction);
        if (value != null) {
          bzlCompileCache.put(key, value);
        }
      }
      return value;
    }

    @Override
    public void doneWithBzlCompileValue(BzlCompileValue.Key key) {
      bzlCompileCache.invalidate(key);
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

  static final class BzlLoadFailedException extends Exception implements SaneAnalysisException {
    private final Transience transience;
    private final DetailedExitCode detailedExitCode;

    private BzlLoadFailedException(
        String errorMessage, DetailedExitCode detailedExitCode, Transience transience) {
      super(errorMessage);
      this.transience = transience;
      this.detailedExitCode = detailedExitCode;
    }

    private BzlLoadFailedException(String errorMessage, DetailedExitCode detailedExitCode) {
      this(errorMessage, detailedExitCode, Transience.PERSISTENT);
    }

    private BzlLoadFailedException(
        String errorMessage,
        DetailedExitCode detailedExitCode,
        Exception cause,
        Transience transience) {
      super(errorMessage, cause);
      this.transience = transience;
      this.detailedExitCode = detailedExitCode;
    }

    private BzlLoadFailedException(String errorMessage, Code code) {
      this(errorMessage, createDetailedExitCode(errorMessage, code), Transience.PERSISTENT);
    }

    private BzlLoadFailedException(
        String errorMessage, Code code, Exception cause, Transience transience) {
      this(errorMessage, createDetailedExitCode(errorMessage, code), cause, transience);
    }

    Transience getTransience() {
      return transience;
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return detailedExitCode;
    }

    private static DetailedExitCode createDetailedExitCode(String message, Code code) {
      return DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage(message)
              .setStarlarkLoading(StarlarkLoading.newBuilder().setCode(code))
              .build());
    }

    private static BzlLoadFailedException whileLoadingDep(
        Location loc, BzlLoadFailedException cause) {
      // Don't chain exception cause, just incorporate the message with a prefix.
      // TODO(bazel-team): This exception should hold a Location of the requesting file's load
      // statement, and code that catches it should use the location in the Event they create.
      return new BzlLoadFailedException(
          "at " + loc + ": " + cause.getMessage(), cause.getDetailedExitCode());
    }

    static BzlLoadFailedException executionFailed(Label label) {
      return new BzlLoadFailedException(
          String.format(
              "initialization of module '%s'%s failed",
              label.toPathFragment(),
              StarlarkBuiltinsValue.isBuiltinsRepo(label.getRepository()) ? " (internal)" : ""),
          Code.EVAL_ERROR);
    }

    static BzlLoadFailedException errorFindingContainingPackage(
        PathFragment file, Exception cause) {
      String errorMessage =
          String.format(
              "Encountered error while reading extension file '%s': %s", file, cause.getMessage());
      DetailedExitCode detailedExitCode =
          cause instanceof DetailedException
              ? ((DetailedException) cause).getDetailedExitCode()
              : createDetailedExitCode(errorMessage, Code.CONTAINING_PACKAGE_NOT_FOUND);
      return new BzlLoadFailedException(
          errorMessage, detailedExitCode, cause, Transience.PERSISTENT);
    }

    static BzlLoadFailedException errorReadingBzl(
        PathFragment file, BzlCompileFunction.FailedIOException cause) {
      String errorMessage =
          String.format(
              "Encountered error while reading extension file '%s': %s", file, cause.getMessage());
      return new BzlLoadFailedException(errorMessage, Code.IO_ERROR, cause, cause.getTransience());
    }

    static BzlLoadFailedException noBuildFile(Label file, @Nullable String reason) {
      if (reason != null) {
        return new BzlLoadFailedException(
            String.format("Unable to find package for %s: %s.", file, reason),
            Code.PACKAGE_NOT_FOUND);
      }
      return new BzlLoadFailedException(
          String.format(
              "Every .bzl file must have a corresponding package, but '%s' does not have one."
                  + " Please create a BUILD file in the same or any parent directory. Note that"
                  + " this BUILD file does not need to do anything except exist.",
              file),
          Code.PACKAGE_NOT_FOUND);
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
              containingPackageLookupValue),
          Code.LABEL_CROSSES_PACKAGE_BOUNDARY);
    }

    static BzlLoadFailedException builtinsFailed(Label file, BuiltinsFailedException cause) {
      return new BzlLoadFailedException(
          String.format(
              "Internal error while loading Starlark builtins for %s: %s",
              file, cause.getMessage()),
          Code.BUILTINS_ERROR,
          cause,
          cause.getTransience());
    }
  }

  private static final class BzlLoadFunctionException extends SkyFunctionException {
    private BzlLoadFunctionException(BzlLoadFailedException cause) {
      super(cause, cause.transience);
    }
  }
}
