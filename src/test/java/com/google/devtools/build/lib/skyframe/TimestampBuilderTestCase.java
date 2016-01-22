// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.actions.util.ActionCacheTestHelper.AMNESIAC_CACHE;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildtool.SkyframeBuilder;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.junit.Before;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * The common code that's shared between various builder tests.
 */
public abstract class TimestampBuilderTestCase extends FoundationTestCase {

  private static final SkyKey OWNER_KEY = new SkyKey(SkyFunctions.ACTION_LOOKUP, "OWNER");
  protected static final ActionLookupValue.ActionLookupKey ALL_OWNER =
      new SingletonActionLookupKey();
  protected static final Predicate<Action> ALWAYS_EXECUTE_FILTER = Predicates.alwaysTrue();
  protected static final String CYCLE_MSG = "Yarrrr, there be a cycle up in here";

  protected Clock clock = BlazeClock.instance();
  protected TimestampGranularityMonitor tsgm;
  protected RecordingDifferencer differencer = new RecordingDifferencer();
  private Set<Action> actions;

  protected AtomicReference<EventBus> eventBusRef = new AtomicReference<>();

  @Before
  public final void initialize() throws Exception  {
    inMemoryCache = new InMemoryActionCache();
    tsgm = new TimestampGranularityMonitor(clock);
    ResourceManager.instance().setAvailableResources(ResourceSet.createWithRamCpuIo(100, 1, 1));
    actions = new HashSet<>();
  }

  protected void clearActions() {
    actions.clear();
  }

  protected <T extends Action> T registerAction(T action) {
    actions.add(action);
    return action;
  }

  protected Builder createBuilder(ActionCache actionCache) {
    return createBuilder(actionCache, 1, /*keepGoing=*/ false);
  }

  /**
   * Create a ParallelBuilder with a DatabaseDependencyChecker using the
   * specified ActionCache.
   */
  protected Builder createBuilder(
      final ActionCache actionCache, final int threadCount, final boolean keepGoing) {
    return createBuilder(actionCache, threadCount, keepGoing, null);
  }

  protected Builder createBuilder(
      final ActionCache actionCache,
      final int threadCount,
      final boolean keepGoing,
      @Nullable EvaluationProgressReceiver evaluationProgressReceiver) {
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)));
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(pkgLocator, false);
    differencer = new RecordingDifferencer();

    ActionExecutionStatusReporter statusReporter =
        ActionExecutionStatusReporter.create(new StoredEventHandler());
    final SkyframeActionExecutor skyframeActionExecutor =
        new SkyframeActionExecutor(
            ResourceManager.instance(), eventBusRef, new AtomicReference<>(statusReporter));

    skyframeActionExecutor.setActionLogBufferPathGenerator(
        new ActionLogBufferPathGenerator(actionOutputBase));

    final InMemoryMemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(SkyFunctions.FILE_STATE, new FileStateFunction(tsgm, externalFilesHelper))
                .put(SkyFunctions.FILE, new FileFunction(pkgLocator))
                .put(SkyFunctions.ARTIFACT,
                    new ArtifactFunction(Predicates.<PathFragment>alwaysFalse()))
                .put(SkyFunctions.ACTION_EXECUTION,
                    new ActionExecutionFunction(skyframeActionExecutor, tsgm))
                .put(SkyFunctions.PACKAGE,
                    new PackageFunction(null, null, null, null, null, null, null))
                .put(SkyFunctions.PACKAGE_LOOKUP, new PackageLookupFunction(null))
                .put(SkyFunctions.WORKSPACE_AST,
                    new WorkspaceASTFunction(TestRuleClassProvider.getRuleClassProvider()))
                .put(SkyFunctions.WORKSPACE_FILE,
                    new WorkspaceFileFunction(TestRuleClassProvider.getRuleClassProvider(),
                        new PackageFactory(TestRuleClassProvider.getRuleClassProvider()),
                        new BlazeDirectories(rootDirectory, outputBase, rootDirectory)))
                .build(),
            differencer,
            evaluationProgressReceiver);
    final SequentialBuildDriver driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());

    return new Builder() {
      private void setGeneratingActions() {
        if (evaluator.getExistingValueForTesting(OWNER_KEY) == null) {
          differencer.inject(ImmutableMap.of(OWNER_KEY, new ActionLookupValue(actions)));
        }
      }

      @Override
      public void buildArtifacts(
          Reporter reporter,
          Set<Artifact> artifacts,
          Set<ConfiguredTarget> parallelTests,
          Set<ConfiguredTarget> exclusiveTests,
          Collection<ConfiguredTarget> targetsToBuild,
          Collection<AspectValue> aspects,
          Executor executor,
          Set<ConfiguredTarget> builtTargets,
          boolean explain,
          Range<Long> lastExecutionTimeRange)
          throws BuildFailedException, AbruptExitException, InterruptedException,
              TestExecException {
        skyframeActionExecutor.prepareForExecution(
            reporter,
            executor,
            keepGoing, /*explain=*/
            false,
            new ActionCacheChecker(actionCache, null, ALWAYS_EXECUTE_FILTER, false), null);

        List<SkyKey> keys = new ArrayList<>();
        for (Artifact artifact : artifacts) {
          keys.add(ArtifactValue.key(artifact, true));
        }

        setGeneratingActions();
        EvaluationResult<SkyValue> result = driver.evaluate(keys, keepGoing, threadCount, reporter);

        if (result.hasError()) {
          boolean hasCycles = false;
          for (Map.Entry<SkyKey, ErrorInfo> entry : result.errorMap().entrySet()) {
            Iterable<CycleInfo> cycles = entry.getValue().getCycleInfo();
            hasCycles |= !Iterables.isEmpty(cycles);
          }
          if (hasCycles) {
            throw new BuildFailedException(CYCLE_MSG);
          } else if (result.errorMap().isEmpty() || keepGoing) {
            throw new BuildFailedException();
          } else {
            SkyframeBuilder.rethrow(Preconditions.checkNotNull(result.getError().getException()));
          }
        }
      }
    };
  }

  /** A non-persistent cache. */
  protected InMemoryActionCache inMemoryCache;

  /** A class that records an event. */
  protected static class Button implements Runnable {
    protected boolean pressed = false;

    @Override
    public void run() {
      pressed = true;
    }
  }

  /** A class that counts occurrences of an event. */
  static class Counter implements Runnable {
    int count = 0;

    @Override
    public void run() {
      count++;
    }
  }

  Artifact createSourceArtifact(String name) {
    return createSourceArtifact(scratch.getFileSystem(), name);
  }

  Artifact createSourceArtifact(FileSystem fs, String name) {
    Path root = fs.getPath(TestUtils.tmpDir());
    return new Artifact(new PathFragment(name), Root.asSourceRoot(root));
  }

  protected Artifact createDerivedArtifact(String name) {
    return createDerivedArtifact(scratch.getFileSystem(), name);
  }

  Artifact createDerivedArtifact(FileSystem fs, String name) {
    Path execRoot = fs.getPath(TestUtils.tmpDir());
    PathFragment execPath = new PathFragment("out").getRelative(name);
    Path path = execRoot.getRelative(execPath);
    return new Artifact(
        path, Root.asDerivedRoot(execRoot, execRoot.getRelative("out")), execPath, ALL_OWNER);
  }

  /**
   * Creates and returns a new "amnesiac" builder based on the amnesiac cache.
   */
  protected Builder amnesiacBuilder() {
    return createBuilder(AMNESIAC_CACHE);
  }

  /**
   * Creates and returns a new caching builder based on the inMemoryCache.
   */
  protected Builder cachingBuilder() {
    return createBuilder(inMemoryCache);
  }

  /**
   * Creates a TestAction from 'inputs' to 'outputs', and a new button, such
   * that executing the action causes the button to be pressed.  The button is
   * returned.
   */
  protected Button createActionButton(Collection<Artifact> inputs, Collection<Artifact> outputs) {
    Button button = new Button();
    registerAction(new TestAction(button, inputs, outputs));
    return button;
  }

  /**
   * Creates a TestAction from 'inputs' to 'outputs', and a new counter, such
   * that executing the action causes the counter to be incremented.  The
   * counter is returned.
   */
  protected Counter createActionCounter(Collection<Artifact> inputs, Collection<Artifact> outputs) {
    Counter counter = new Counter();
    registerAction(new TestAction(counter, inputs, outputs));
    return counter;
  }

  protected static Set<Artifact> emptySet = Collections.emptySet();

  protected void buildArtifacts(Builder builder, Artifact... artifacts)
      throws BuildFailedException, AbruptExitException, InterruptedException, TestExecException {

    tsgm.setCommandStartTime();
    Set<Artifact> artifactsToBuild = Sets.newHashSet(artifacts);
    Set<ConfiguredTarget> builtArtifacts = new HashSet<>();
    try {
      builder.buildArtifacts(
          reporter,
          artifactsToBuild,
          null,
          null,
          null,
          null,
          new DummyExecutor(rootDirectory),
          builtArtifacts, /*explain=*/
          false,
          null);
    } finally {
      tsgm.waitForTimestampGranularity(reporter.getOutErr());
    }
  }

  protected static class InMemoryActionCache implements ActionCache {

    private final Map<String, Entry> actionCache = new HashMap<>();

    @Override
    public synchronized void put(String key, ActionCache.Entry entry) {
      actionCache.put(key, entry);
    }

    @Override
    public synchronized Entry get(String key) {
      return actionCache.get(key);
    }

    @Override
    public synchronized void remove(String key) {
      actionCache.remove(key);
    }

    @Override
    public Entry createEntry(String key, boolean discoversInputs) {
      return new ActionCache.Entry(key, discoversInputs);
    }

    public synchronized void reset() {
      actionCache.clear();
    }

    @Override
    public long save() {
      // safe to ignore
      return 0;
    }

    @Override
    public void dump(PrintStream out) {
      out.println("In-memory action cache has " + actionCache.size() + " records");
    }
  }

  private static class SingletonActionLookupKey extends ActionLookupValue.ActionLookupKey {
    @Override
    SkyKey getSkyKey() {
      return OWNER_KEY;
    }

    @Override
    SkyFunctionName getType() {
      throw new UnsupportedOperationException();
    }
  }
}
