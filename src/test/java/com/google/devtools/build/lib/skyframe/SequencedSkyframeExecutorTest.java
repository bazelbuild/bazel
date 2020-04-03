// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionCacheTestHelper.AMNESIAC_CACHE;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEventRegex;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertEventCount;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNotContainsEventRegex;
import static com.google.devtools.build.lib.util.subjects.DetailedExitCodeSubjectFactory.assertThatDetailedExitCode;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.hash.HashCode;
import com.google.common.testing.GcFinalization;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.actions.util.InjectedActionLookupKey;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.SkyframeBuilder;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.BasicFilesystemDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ProgressSupplier;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.DeterministicHelper;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.NotifyingHelper;
import com.google.devtools.build.skyframe.NotifyingHelper.EventType;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.TaggedEvents;
import com.google.devtools.build.skyframe.TrackingAwaiter;
import com.google.devtools.build.skyframe.ValueWithMetadata;
import com.google.devtools.common.options.OptionsParser;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SequencedSkyframeExecutor}. */
@RunWith(JUnit4.class)
public final class SequencedSkyframeExecutorTest extends BuildViewTestCase {
  private TransitivePackageLoader visitor;
  private OptionsParser options;

  @Before
  public final void createSkyframeExecutorAndVisitor() throws Exception {
    skyframeExecutor = getSkyframeExecutor();
    skyframeExecutor.setRemoteOutputsMode(RemoteOutputsMode.ALL);
    visitor = skyframeExecutor.pkgLoader();
    options =
        OptionsParser.builder()
            .optionsClasses(
                ImmutableList.of(
                    KeepGoingOption.class, BuildRequestOptions.class, AnalysisOptions.class))
            .build();
    options.parse("--jobs=20");
  }

  @Test
  public void testChangeFile() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, Root.fromPath(rootDirectory));

    String pathString = rootDirectory + "/python/hello/BUILD";
    scratch.file(pathString, "py_binary(name = 'hello', srcs = ['hello.py'])");

    // A dummy file that is never changed.
    scratch.file(rootDirectory + "/misc/BUILD", "sh_binary(name = 'misc', srcs = ['hello.sh'])");

    sync("//python/hello:hello", "//misc:misc");

    // No changes yet.
    assertThat(dirtyValues()).isEmpty();

    // Make a change.
    scratch.overwriteFile(pathString, "py_binary(name = 'hello', srcs = ['something_else.py'])");
    assertThat(dirtyValues())
        .containsExactly(
            FileStateValue.key(
                RootedPath.toRootedPath(
                    Root.fromPath(rootDirectory), PathFragment.create("python/hello/BUILD"))));

    // The method will continue returning the value until we invalidate it and re-evaluate.
    assertThat(dirtyValues()).hasSize(1);
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter,
        ModifiedFileSet.builder().modify(PathFragment.create("python/hello/BUILD")).build(),
        Root.fromPath(rootDirectory));
    sync("//python/hello:hello");
    assertThat(dirtyValues()).isEmpty();
  }

  // Regression for b/13328517. clearAnalysisCache() method is call when --discard_analysis_cache
  // is used. This saves about 10% of the memory during execution.
  @Test
  public void testClearAnalysisCache() throws Exception {
    scratch.file(rootDirectory + "/discard/BUILD",
        "genrule(name='x', srcs=['input'], outs=['out'], cmd='false')");
    scratch.file(rootDirectory + "/discard/input", "foo");

    ConfiguredTarget ct =
        skyframeExecutor.getConfiguredTargetForTesting(
            reporter,
            Label.parseAbsolute("@//discard:x", ImmutableMap.of()),
            getTargetConfiguration());
    assertThat(ct).isNotNull();
    WeakReference<ConfiguredTarget> ref = new WeakReference<>(ct);
    ct = null;
    // Allow all values to be cleared by passing in empty set of top-level values, since we're not
    // actually building.
    skyframeExecutor.clearAnalysisCache(
        ImmutableSet.<ConfiguredTarget>of(), ImmutableSet.<AspectValue>of());
    GcFinalization.awaitClear(ref);
  }

  @Test
  public void testChangeDirectory() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, Root.fromPath(rootDirectory));

    scratch.file("python/hello/BUILD",
        "py_binary(name = 'hello', srcs = ['hello.py'], data = glob(['*.txt']))");
    scratch.file("python/hello/foo.txt", "foo");

    // A dummy directory that is not changed.
    scratch.file("misc/BUILD",
        "py_binary(name = 'misc', srcs = ['other.py'], data = glob(['*.txt']))");

    sync("//python/hello:hello", "//misc:misc");

    // No changes yet.
    assertThat(dirtyValues()).isEmpty();

    // Make a change.
    scratch.file("python/hello/bar.txt", "bar");
    assertThat(dirtyValues())
        .containsExactly(
            DirectoryListingStateValue.key(
                RootedPath.toRootedPath(
                    Root.fromPath(rootDirectory), PathFragment.create("python/hello"))));

    // The method will continue returning the value until we invalidate it and re-evaluate.
    assertThat(dirtyValues()).hasSize(1);
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter,
        ModifiedFileSet.builder().modify(PathFragment.create("python/hello/bar.txt")).build(),
        Root.fromPath(rootDirectory));
    sync("//python/hello:hello");
    assertThat(dirtyValues()).isEmpty();
  }

  @Test
  public void testSetDeletedPackages() throws Exception {
    ExtendedEventHandler eventHandler = NullEventHandler.INSTANCE;
    scratch.file("foo/bar/BUILD", "cc_library(name = 'bar', hdrs = ['bar.h'])");
    scratch.file("foo/baz/BUILD", "cc_library(name = 'baz', hdrs = ['baz.h'])");

    assertThat(
            skyframeExecutor
                .getPackageManager()
                .isPackage(eventHandler, PackageIdentifier.createInMainRepo("foo/bar")))
        .isTrue();
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .getBuildFileForPackage(PackageIdentifier.createInMainRepo("foo/bar")))
        .isNotNull();
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .isPackage(eventHandler, PackageIdentifier.createInMainRepo("foo/baz")))
        .isTrue();
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .getBuildFileForPackage(PackageIdentifier.createInMainRepo("foo/baz")))
        .isNotNull();
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .isPackage(eventHandler, PackageIdentifier.createInMainRepo("not/a/package")))
        .isFalse();
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .getBuildFileForPackage(PackageIdentifier.createInMainRepo("not/a/package")))
        .isNull();

    skyframeExecutor.getPackageManager().getPackage(
        eventHandler, PackageIdentifier.createInMainRepo("foo/bar"));
    skyframeExecutor.getPackageManager().getPackage(
        eventHandler, PackageIdentifier.createInMainRepo("foo/baz"));

    assertThrows(
        "non-existent package was incorrectly thought to exist",
        NoSuchPackageException.class,
        () ->
            skyframeExecutor
                .getPackageManager()
                .getPackage(eventHandler, PackageIdentifier.createInMainRepo("not/a/package")));

    ImmutableSet<PackageIdentifier> deletedPackages = ImmutableSet.of(
        PackageIdentifier.createInMainRepo("foo/bar"));
    skyframeExecutor.setDeletedPackages(deletedPackages);

    assertThat(
            skyframeExecutor
                .getPackageManager()
                .isPackage(eventHandler, PackageIdentifier.createInMainRepo("foo/bar")))
        .isFalse();
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .getBuildFileForPackage(PackageIdentifier.createInMainRepo("foo/bar")))
        .isNull();
    assertThrows(
        "deleted package was incorrectly thought to exist",
        NoSuchPackageException.class,
        () ->
            skyframeExecutor
                .getPackageManager()
                .getPackage(eventHandler, PackageIdentifier.createInMainRepo("foo/bar")));
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .isPackage(eventHandler, PackageIdentifier.createInMainRepo("foo/baz")))
        .isTrue();
  }

  // Directly tests that PackageFunction adds a dependency on the PackageLookupValue for
  // (potential) subpackages. This is tested indirectly in several places (e.g.
  // LabelVisitorTest#testSubpackageBoundaryAdd and
  // PackageDeletionTest#testUnsuccessfulBuildAfterDeletion) but those tests are also indirectly
  // testing the behavior of TargetFunction when the target has a '/'.
  @Test
  public void testDependencyOnPotentialSubpackages() throws Exception {
    ExtendedEventHandler eventHandler = NullEventHandler.INSTANCE;
    scratch.file("x/BUILD",
        "sh_library(name = 'x', deps = ['//x:y/z'])",
        "sh_library(name = 'y/z')");

    Package pkgBefore = skyframeExecutor.getPackageManager().getPackage(
        eventHandler, PackageIdentifier.createInMainRepo("x"));
    assertThat(pkgBefore.containsErrors()).isFalse();

    scratch.file("x/y/BUILD",
        "sh_library(name = 'z')");
    ModifiedFileSet modifiedFiles = ModifiedFileSet.builder()
        .modify(PathFragment.create("x"))
        .modify(PathFragment.create("x/y"))
        .modify(PathFragment.create("x/y/BUILD"))
        .build();
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, modifiedFiles, Root.fromPath(rootDirectory));

    // The package lookup for "x" should now fail because it's invalid.
    reporter.removeHandler(failFastHandler); // expect errors
    assertThat(
            skyframeExecutor
                .getPackageManager()
                .getPackage(eventHandler, PackageIdentifier.createInMainRepo("x"))
                .containsErrors())
        .isTrue();

    scratch.deleteFile("x/y/BUILD");
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, modifiedFiles, Root.fromPath(rootDirectory));

    // The package lookup for "x" should now succeed again.
    reporter.addHandler(failFastHandler); // no longer expect errors
    Package pkgAfter = skyframeExecutor.getPackageManager().getPackage(
        eventHandler, PackageIdentifier.createInMainRepo("x"));
    assertThat(pkgAfter).isNotSameInstanceAs(pkgBefore);
  }

  @Test
  public void testSkyframePackageManagerGetBuildFileForPackage() throws Exception {
    PackageManager skyframePackageManager = skyframeExecutor.getPackageManager();

    scratch.file("nobuildfile/foo.txt");
    scratch.file("deletedpackage/BUILD");
    skyframeExecutor.setDeletedPackages(ImmutableList.of(
        PackageIdentifier.createInMainRepo("deletedpackage")));
    scratch.file("invalidpackagename.42/BUILD");
    Path everythingGoodBuildFilePath = scratch.file("everythinggood/BUILD");

    assertThat(
            skyframePackageManager.getBuildFileForPackage(
                PackageIdentifier.createInMainRepo("nobuildfile")))
        .isNull();
    assertThat(
            skyframePackageManager.getBuildFileForPackage(
                PackageIdentifier.createInMainRepo("deletedpackage")))
        .isNull();
    assertThat(
            skyframePackageManager.getBuildFileForPackage(
                PackageIdentifier.createInMainRepo("everythinggood")))
        .isEqualTo(everythingGoodBuildFilePath);
  }

  /**
   * Indirect regression test for b/12543229: "The Skyframe error propagation model is
   * problematic".
   */
  @Test
  public void testPackageFunctionHandlesExceptionFromDependencies() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path badDirPath = scratch.dir("bad/dir");
    // This will cause an IOException when trying to compute the glob, which is required to load
    // the package.
    badDirPath.setReadable(false);
    scratch.file("bad/BUILD",
        "filegroup(name='fg', srcs=glob(['**']))");
    assertThrows(
        NoSuchPackageException.class,
        () ->
            skyframeExecutor
                .getPackageManager()
                .getPackage(reporter, PackageIdentifier.createInMainRepo("bad")));
  }

  private Collection<SkyKey> dirtyValues() throws InterruptedException {
    Diff diff =
        new FilesystemValueChecker(new TimestampGranularityMonitor(BlazeClock.instance()), null)
            .getDirtyKeys(
                skyframeExecutor.getEvaluatorForTesting().getValues(),
                new BasicFilesystemDirtinessChecker());
    return ImmutableList.<SkyKey>builder()
        .addAll(diff.changedKeysWithoutNewValues())
        .addAll(diff.changedKeysWithNewValues().keySet())
        .build();
  }

  private void sync(String... labelStrings) throws Exception {
    Set<Label> labels = new HashSet<>();
    for (String labelString : labelStrings) {
      labels.add(Label.parseAbsolute(labelString, ImmutableMap.of()));
    }
    visitor.sync(reporter, labels, /*keepGoing=*/ false, /*parallelThreads=*/ 200);
  }

  @Test
  public void testInterruptLoadedTarget() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
    scratch.file("python/hello/BUILD",
        "py_binary(name = 'hello', srcs = ['hello.py'], data = glob(['*.txt']))");
    Thread.currentThread().interrupt();
    LoadedPackageProvider packageProvider =
        new LoadedPackageProvider(skyframeExecutor.getPackageManager(), reporter);
    assertThrows(
        InterruptedException.class,
        () ->
            packageProvider.getLoadedTarget(Label.parseAbsoluteUnchecked("//python/hello:hello")));
    Target target = packageProvider.getLoadedTarget(
        Label.parseAbsoluteUnchecked("//python/hello:hello"));
    assertThat(target).isNotNull();
  }

  /**
   * Generating the same output from two targets is ok if we build them on successive builds
   * and invalidate the first target before we build the second target. This test is basically
   * copied here from {@code AnalysisCachingTest} because here we can control the number of Skyframe
   * update calls that we make. This prevents an intermediate update call from clearing the action
   * and hiding the bug.
   */
  @Test
  public void testNoActionConflictWithInvalidatedTarget() throws Exception {
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    ConfiguredTargetAndData conflict =
        skyframeExecutor.getConfiguredTargetAndDataForTesting(
            reporter,
            Label.parseAbsolute("@//conflict:x", ImmutableMap.of()),
            getTargetConfiguration());
    assertThat(conflict).isNotNull();
    ArtifactRoot root =
        getTargetConfiguration()
            .getBinDirectory(
                conflict.getConfiguredTarget().getLabel().getPackageIdentifier().getRepository());

    Action oldAction =
        getGeneratingAction(
            getDerivedArtifact(
                PathFragment.create("conflict/_objs/x/foo.o"),
                root,
                ConfiguredTargetKey.of(
                    conflict.getConfiguredTarget(), conflict.getConfiguration())));
    assertThat(oldAction.getOwner().getLabel().toString()).isEqualTo("//conflict:x");
    skyframeExecutor.handleAnalysisInvalidatingChange();
    ConfiguredTargetAndData objsConflict =
        skyframeExecutor.getConfiguredTargetAndDataForTesting(
            reporter,
            Label.parseAbsolute("@//conflict:_objs/x/foo.o", ImmutableMap.of()),
            getTargetConfiguration());
    assertThat(objsConflict).isNotNull();
    Action newAction =
        getGeneratingAction(
            getDerivedArtifact(
                PathFragment.create("conflict/_objs/x/foo.o"),
                root,
                ConfiguredTargetKey.of(
                    objsConflict.getConfiguredTarget(), objsConflict.getConfiguration())));
    assertThat(newAction.getOwner().getLabel().toString()).isEqualTo("//conflict:_objs/x/foo.o");
  }

  @Test
  public void testGetPackageUsesListener() throws Exception {
    scratch.file("pkg/BUILD", "thisisanerror");
    EventCollector customEventCollector = new EventCollector(EventKind.ERRORS);
    Package pkg = skyframeExecutor.getPackageManager().getPackage(
        new Reporter(new EventBus(), customEventCollector),
        PackageIdentifier.createInMainRepo("pkg"));
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(customEventCollector, "name 'thisisanerror' is not defined");
  }

  /** Dummy action that does not create its lone output file. */
  private static class MissingOutputAction extends DummyAction {
    MissingOutputAction(NestedSet<Artifact> inputs, Artifact output, MiddlemanType type) {
      super(inputs, output, type);
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      ActionResult actionResult = super.execute(actionExecutionContext);
      try {
        getPrimaryOutput().getPath().deleteTree();
      } catch (IOException e) {
        throw new AssertionError(e);
      }
      return actionResult;
    }
  }

  private static final ActionCacheChecker NULL_CHECKER =
      new ActionCacheChecker(
          AMNESIAC_CACHE,
          new ArtifactResolver() {
            @Override
            public Artifact getSourceArtifact(
                PathFragment execPath, Root root, ArtifactOwner owner) {
              throw new UnsupportedOperationException();
            }

            @Override
            public Artifact getSourceArtifact(PathFragment execPath, Root root) {
              throw new UnsupportedOperationException();
            }

            @Override
            public Artifact resolveSourceArtifact(
                PathFragment execPath, RepositoryName repositoryName) {
              throw new UnsupportedOperationException();
            }

            @Override
            public Map<PathFragment, Artifact> resolveSourceArtifacts(
                Iterable<PathFragment> execPaths, PackageRootResolver resolver) {
              throw new UnsupportedOperationException();
            }

            @Override
            public Path getPathFromSourceExecPath(Path execRoot, PathFragment execPath) {
              throw new UnsupportedOperationException();
            }
          },
          new ActionKeyContext(),
          Predicates.<Action>alwaysTrue(),
          null);

  private static final ProgressSupplier EMPTY_PROGRESS_SUPPLIER = new ProgressSupplier() {
    @Override
    public String getProgressString() {
      return "";
    }
  };

  private static final ActionCompletedReceiver EMPTY_COMPLETION_RECEIVER =
      new ActionCompletedReceiver() {
        @Override
        public void actionCompleted(ActionLookupData actionLookupData) {}

        @Override
        public void noteActionEvaluationStarted(ActionLookupData actionLookupData, Action action) {}
      };

  private EvaluationResult<FileArtifactValue> evaluate(Iterable<? extends SkyKey> roots)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SequencedSkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHander(reporter)
            .build();
    return skyframeExecutor.getDriver().evaluate(roots, evaluationContext);
  }

  /**
   * Make sure that if a shared action fails to create an output file, the other action doesn't
   * complain about it too.
   */
  @Test
  public void testSharedActionsNoOutputs() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("missing");
    // We create two "configured targets" and two copies of the same artifact, each generated by
    // an action from its respective configured target.
    ActionLookupValue.ActionLookupKey lc1 = new InjectedActionLookupKey("lc1");
    Artifact output1 =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lc1);
    Action action1 =
        new MissingOutputAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), output1, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctValue1 = createConfiguredTargetValue(action1, lc1);
    ActionLookupValue.ActionLookupKey lc2 = new InjectedActionLookupKey("lc2");
    Artifact output2 =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lc2);
    Action action2 =
        new MissingOutputAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), output2, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctValue2 = createConfiguredTargetValue(action2, lc2);
    // Inject the "configured targets" into the graph.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lc1, ctValue1, lc2, ctValue2));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER, ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.<Artifact>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<AspectValue>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    reporter.removeHandler(failFastHandler); // Expect errors.
    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    EvaluationResult<FileArtifactValue> result = evaluate(ImmutableList.of(output1, output2));
    assertWithMessage(result.toString()).that(result.keyNames()).isEmpty();
    assertThat(result.hasError()).isTrue();
    MoreAsserts.assertContainsEvent(eventCollector,
        "output '" + output1.prettyPrint() + "' was not created");
    MoreAsserts.assertContainsEvent(eventCollector, "not all outputs were created or valid");
    assertEventCount(2, eventCollector);
  }

  /** Shared actions can race and both check the action cache and try to execute. */
  @Test
  public void testSharedActionsRacing() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("file");
    Path sourcePath = rootDirectory.getRelative("foo/src");
    FileSystemUtils.createDirectoryAndParents(sourcePath.getParentDirectory());
    FileSystemUtils.createEmptyFile(sourcePath);

    // We create two "configured targets" and two copies of the same artifact, each generated by
    // an action from its respective configured target. Both actions will consume the input file
    // "out/input" so we can synchronize their execution.
    ActionLookupValue.ActionLookupKey inputKey = new InjectedActionLookupKey("input");
    Artifact input =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            PathFragment.create("out").getRelative("input"),
            inputKey);
    Action baseAction =
        new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), input, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctBase = createConfiguredTargetValue(baseAction, inputKey);
    ActionLookupValue.ActionLookupKey lc1 = new InjectedActionLookupKey("lc1");
    Artifact output1 =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lc1);
    Action action1 =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, input), output1, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctValue1 = createConfiguredTargetValue(action1, lc1);
    ActionLookupValue.ActionLookupKey lc2 = new InjectedActionLookupKey("lc2");
    Artifact output2 =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lc2);
    Action action2 =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, input), output2, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctValue2 = createConfiguredTargetValue(action2, lc2);

    // Stall both actions during the "checking inputs" phase so that neither will enter
    // SkyframeActionExecutor before both have asked SkyframeActionExecutor if another shared action
    // is running. This way, both actions will check the action cache beforehand and try to update
    // the action cache post-build.
    final CountDownLatch inputsRequested = new CountDownLatch(2);
    skyframeExecutor
        .getEvaluatorForTesting()
        .injectGraphTransformerForTesting(
            NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  if (type == EventType.GET_VALUE_WITH_METADATA
                      && key.functionName().equals(Artifact.ARTIFACT)
                      && input.equals(key)) {
                    inputsRequested.countDown();
                    try {
                      assertThat(
                              inputsRequested.await(
                                  TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                          .isTrue();
                    } catch (InterruptedException e) {
                      throw new IllegalStateException(e);
                    }
                  }
                }));

    // Inject the "configured targets" and artifact into the graph.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lc1, ctValue1, lc2, ctValue2, inputKey, ctBase));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER, ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.<Artifact>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<AspectValue>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    EvaluationResult<FileArtifactValue> result =
        evaluate(Artifact.keys(ImmutableList.of(output1, output2)));
    assertThat(result.hasError()).isFalse();
    TrackingAwaiter.INSTANCE.assertNoErrors();
  }

  /**
   * Tests a subtle situation when three shared actions race and are interrupted. Action A starts
   * executing. Actions B and C start executing. Action B notices action A is already executing and
   * sets completionFuture. It then exits, returning control to
   * AbstractParallelEvaluator$Evaluate#run code. The build is interrupted. When B's code tries to
   * register the future with AbstractQueueVisitor, the future is canceled (or if the interrupt
   * races with B registering the future, shortly thereafter). Action C then starts running. It too
   * notices Action A is already executing. The future's state should be consistent. A cannot finish
   * until C runs, since otherwise C would see that A was done.
   */
  @Test
  public void testThreeSharedActionsRacing() throws Exception {
    Path root = getExecRoot();
    PathFragment out = PathFragment.create("out");
    PathFragment execPath = out.getRelative("file");
    // We create three "configured targets" and three copies of the same artifact, each generated by
    // an action from its respective configured target. The actions wouldn't actually do the same
    // thing if they executed, but they look the same to our execution engine.
    ActionLookupValue.ActionLookupKey lcA = new InjectedActionLookupKey("lcA");
    Artifact outputA =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lcA);
    CountDownLatch actionAStartedSoOthersCanProceed = new CountDownLatch(1);
    CountDownLatch actionCFinishedSoACanFinish = new CountDownLatch(1);
    Action actionA =
        new TestAction(
            (Serializable & Callable<Void>)
                () -> {
                  actionAStartedSoOthersCanProceed.countDown();
                  try {
                    Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                  } catch (InterruptedException e) {
                    TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                        actionCFinishedSoACanFinish, "third didn't finish");
                    throw e;
                  }
                  throw new IllegalStateException("Should have been interrupted");
                },
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(outputA));
    ConfiguredTargetValue ctA = createConfiguredTargetValue(actionA, lcA);

    // Shared actions: they look the same from the point of view of Blaze data.
    ActionLookupValue.ActionLookupKey lcB = new InjectedActionLookupKey("lcB");
    Artifact outputB =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lcB);
    Action actionB =
        new DummyAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), outputB, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctB = createConfiguredTargetValue(actionB, lcB);
    ActionLookupValue.ActionLookupKey lcC = new InjectedActionLookupKey("lcC");
    Artifact outputC =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lcC);
    Action actionC =
        new DummyAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), outputC, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctC = createConfiguredTargetValue(actionC, lcC);

    // Both shared actions wait for A to start executing. We do that by stalling their dep requests
    // on their configured targets. We then let B proceed. Once B finishes its SkyFunction run, it
    // interrupts the main thread. C just waits until it has been interrupted, and then another
    // little bit, to give B time to attempt to add the future and try to cancel it. It not waiting
    // long enough can lead to a flaky pass.

    Thread mainThread = Thread.currentThread();
    CountDownLatch cStarted = new CountDownLatch(1);
    skyframeExecutor
        .getEvaluatorForTesting()
        .injectGraphTransformerForTesting(
            NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  if (type == EventType.GET_VALUE_WITH_METADATA
                      && (key.equals(lcB) || key.equals(lcC))) {
                    // One of the shared actions is requesting its configured target dep.
                    TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                        actionAStartedSoOthersCanProceed, "primary didn't start");
                    if (key.equals(lcC)) {
                      cStarted.countDown();
                      // Wait until interrupted.
                      try {
                        Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                        throw new IllegalStateException("Should have been interrupted");
                      } catch (InterruptedException e) {
                        // Because ActionExecutionFunction doesn't check for interrupts, this
                        // interrupted state will persist until the ADD_REVERSE_DEP code below. If
                        // it does not, this test will start to fail, which is good, since it would
                        // be strange to check for interrupts in that stretch of hot code.
                        Thread.currentThread().interrupt();
                      }
                      // Wait for B thread to cancel its future. It's hard to know exactly when that
                      // will be, so give it time. No flakes in 2k runs with this sleep.
                      Uninterruptibles.sleepUninterruptibly(100, TimeUnit.MILLISECONDS);
                    }
                  } else if (type == EventType.ADD_REVERSE_DEP
                      && key.equals(lcB)
                      && order == NotifyingHelper.Order.BEFORE
                      && context != null) {
                    TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(cStarted, "c missing");
                    // B thread has finished its run. Interrupt build!
                    mainThread.interrupt();
                  } else if (type == EventType.ADD_REVERSE_DEP
                      && key.equals(lcC)
                      && order == NotifyingHelper.Order.BEFORE
                      && context != null) {
                    // Test is almost over: let action A finish now that C observed future.
                    actionCFinishedSoACanFinish.countDown();
                  }
                }));

    // Inject the "configured targets" and artifacts into the graph.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lcA, ctA, lcB, ctB, lcC, ctC));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    reporter.removeHandler(failFastHandler);
    try {
      evaluate(Artifact.keys(ImmutableList.of(outputA, outputB, outputC)));
      fail();
    } catch (InterruptedException e) {
      // Expected.
    }
    TrackingAwaiter.INSTANCE.assertNoErrors();
  }

  /** Dummy codec for serialization. Doesn't actually serialize {@link CountDownLatch}! */
  @SuppressWarnings("unused")
  private static class CountDownLatchCodec implements ObjectCodec<CountDownLatch> {
    private static final CountDownLatch RETURNED = new CountDownLatch(0);

    @Override
    public Class<? extends CountDownLatch> getEncodedClass() {
      return CountDownLatch.class;
    }

    @Override
    public void serialize(
        SerializationContext context, CountDownLatch obj, CodedOutputStream codedOut) {}

    @Override
    public CountDownLatch deserialize(DeserializationContext context, CodedInputStream codedIn) {
      return RETURNED;
    }
  }

  /** Regression test for ##5396: successfully build shared actions with tree artifacts. */
  @Test
  public void sharedActionsWithTree() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("trees");
    // We create two "configured targets" and two copies of the same artifact, each generated by
    // an action from its respective configured target.
    ActionLookupValue.ActionLookupKey lc1 = new InjectedActionLookupKey("lc1");
    Artifact.SpecialArtifact output1 =
        new Artifact.SpecialArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath,
            lc1,
            Artifact.SpecialArtifactType.TREE);
    ImmutableList<PathFragment> children = ImmutableList.of(PathFragment.create("child"));
    Action action1 =
        new TreeArtifactAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output1, children);
    ConfiguredTargetValue ctValue1 = createConfiguredTargetValue(action1, lc1);
    ActionLookupValue.ActionLookupKey lc2 = new InjectedActionLookupKey("lc2");
    Artifact.SpecialArtifact output2 =
        new Artifact.SpecialArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath,
            lc2,
            Artifact.SpecialArtifactType.TREE);
    Action action2 =
        new TreeArtifactAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output2, children);
    ConfiguredTargetValue ctValue2 = createConfiguredTargetValue(action2, lc2);
    // Inject the "configured targets" into the graph.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lc1, ctValue1, lc2, ctValue2));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.<Artifact>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<AspectValue>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    evaluate(ImmutableList.of(output1, output2));
  }

  /** Dummy action that creates a tree output. */
  // AutoCodec because the superclass has a WrappedRunnable inside it.
  @AutoCodec
  @AutoCodec.VisibleForSerialization
  static class TreeArtifactAction extends TestAction {
    @SuppressWarnings("unused") // Only needed for serialization.
    private final Artifact.SpecialArtifact output;

    @SuppressWarnings("unused") // Only needed for serialization.
    private final Iterable<PathFragment> children;

    TreeArtifactAction(
        NestedSet<Artifact> inputs,
        Artifact.SpecialArtifact output,
        Iterable<PathFragment> children) {
      super(() -> createDirectoryAndFiles(output, children), inputs, ImmutableSet.of(output));
      Preconditions.checkState(output.isTreeArtifact(), output);
      this.output = output;
      this.children = children;
    }

    private static void createDirectoryAndFiles(
        Artifact.SpecialArtifact output, Iterable<PathFragment> children) {
      Path directory = output.getPath();
      try {
        directory.createDirectoryAndParents();
        for (PathFragment child : children) {
          FileSystemUtils.createEmptyFile(directory.getRelative(child));
        }
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  /** Regression test for ##5396: successfully build shared actions with tree artifacts. */
  @Test
  public void sharedActionTemplate() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("trees");
    // We create two "configured targets" and two copies of the same artifact, each generated by
    // an action from its respective configured target.
    ActionLookupValue.ActionLookupKey baseKey = new InjectedActionLookupKey("base");
    Artifact.SpecialArtifact baseOutput =
        new Artifact.SpecialArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath,
            baseKey,
            Artifact.SpecialArtifactType.TREE);
    ImmutableList<PathFragment> children = ImmutableList.of(PathFragment.create("child"));
    Action action1 =
        new TreeArtifactAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), baseOutput, children);
    ConfiguredTargetValue baseCt = createConfiguredTargetValue(action1, baseKey);
    ActionLookupValue.ActionLookupKey shared1 = new InjectedActionLookupKey("shared1");
    PathFragment execPath2 = PathFragment.create("out").getRelative("treesShared");
    Artifact.SpecialArtifact sharedOutput1 =
        new Artifact.SpecialArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath2,
            shared1,
            Artifact.SpecialArtifactType.TREE);
    ActionTemplate<DummyAction> template1 =
        new DummyActionTemplate(baseOutput, sharedOutput1, ActionOwner.SYSTEM_ACTION_OWNER);
    ConfiguredTargetValue shared1Ct = createConfiguredTargetValue(template1, shared1);
    ActionLookupValue.ActionLookupKey shared2 = new InjectedActionLookupKey("shared2");
    Artifact.SpecialArtifact sharedOutput2 =
        new Artifact.SpecialArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath2,
            shared2,
            Artifact.SpecialArtifactType.TREE);
    ActionTemplate<DummyAction> template2 =
        new DummyActionTemplate(baseOutput, sharedOutput2, ActionOwner.SYSTEM_ACTION_OWNER);
    ConfiguredTargetValue shared2Ct = createConfiguredTargetValue(template2, shared2);
    // Inject the "configured targets" into the graph.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(baseKey, baseCt, shared1, shared1Ct, shared2, shared2Ct));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.<Artifact>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<AspectValue>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    evaluate(ImmutableList.of(sharedOutput1, sharedOutput2));
  }

  private static class DummyActionTemplate implements ActionTemplate<DummyAction> {
    private final Artifact.SpecialArtifact inputArtifact;
    private final Artifact.SpecialArtifact outputArtifact;
    private final ActionOwner actionOwner;

    private DummyActionTemplate(
        Artifact.SpecialArtifact inputArtifact,
        Artifact.SpecialArtifact outputArtifact,
        ActionOwner actionOwner) {
      this.inputArtifact = inputArtifact;
      this.outputArtifact = outputArtifact;
      this.actionOwner = actionOwner;
    }

    @Override
    public boolean isShareable() {
      return true;
    }

    @Override
    public Iterable<DummyAction> generateActionForInputArtifacts(
        Iterable<TreeFileArtifact> inputTreeFileArtifacts, ActionLookupKey artifactOwner) {
      return ImmutableList.copyOf(inputTreeFileArtifacts).stream()
          .map(
              input -> {
                Artifact.TreeFileArtifact output =
                    ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
                        outputArtifact, input.getParentRelativePath(), artifactOwner);
                return new DummyAction(NestedSetBuilder.create(Order.STABLE_ORDER, input), output);
              })
          .collect(ImmutableList.toImmutableList());
    }

    @Override
    public String getKey(ActionKeyContext actionKeyContext) {
      Fingerprint fp = new Fingerprint();
      fp.addPath(inputArtifact.getPath());
      fp.addPath(outputArtifact.getPath());
      return fp.hexDigestAndReset();
    }

    @Override
    public Artifact getInputTreeArtifact() {
      return inputArtifact;
    }

    @Override
    public Artifact getOutputTreeArtifact() {
      return outputArtifact;
    }

    @Override
    public ActionOwner getOwner() {
      return actionOwner;
    }

    @Override
    public String getMnemonic() {
      return "DummyTemplate";
    }

    @Override
    public String prettyPrint() {
      return "DummyTemplate";
    }

    @Override
    public NestedSet<Artifact> getTools() {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public NestedSet<Artifact> getInputs() {
      return NestedSetBuilder.create(Order.STABLE_ORDER, inputArtifact);
    }

    @Override
    public Iterable<String> getClientEnvironmentVariables() {
      return ImmutableList.of();
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      return ImmutableSet.of(outputArtifact);
    }

    @Override
    public NestedSet<Artifact> getInputFilesForExtraAction(
        ActionExecutionContext actionExecutionContext) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public ImmutableSet<Artifact> getMandatoryOutputs() {
      return ImmutableSet.of();
    }

    @Override
    public Artifact getPrimaryInput() {
      return inputArtifact;
    }

    @Override
    public Artifact getPrimaryOutput() {
      return outputArtifact;
    }

    @Override
    public NestedSet<Artifact> getMandatoryInputs() {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
      return this != action;
    }

    @Override
    public MiddlemanType getActionType() {
      return MiddlemanType.NORMAL;
    }
  }

  /**
   * b/150153544: demonstration of how shared actions do not work on incremental builds when action
   * cache is disabled. In practice, this test usually throws an exception and deadlocks, because
   * the "top" action notices that its input is missing even before the callable specified here
   * executes and throws an exception, so shared action2 never gets the signal to finish. However,
   * even if "top" is delayed to wait for the shared action2 to run, the assertion that the artifact
   * exists will fail, since action2's "prepare" step deleted it.
   */
  @Ignore
  @Test
  public void incrementalSharedActions() throws Exception {
    Path root = getExecRoot();
    PathFragment relativeOut = PathFragment.create("out");
    PathFragment execPath = relativeOut.getRelative("file");
    Path sourcePath = rootDirectory.getRelative("foo/src");
    FileSystemUtils.createDirectoryAndParents(sourcePath.getParentDirectory());
    FileSystemUtils.createEmptyFile(sourcePath);

    // We create two "configured targets" and two copies of the same artifact, each generated by
    // an action from its respective configured target.
    ActionLookupValue.ActionLookupKey lc1 = new InjectedActionLookupKey("lc1");
    Artifact output1 =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lc1);
    Action action1 =
        new DummyAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), output1, MiddlemanType.NORMAL);
    ConfiguredTargetValue ctValue1 = createConfiguredTargetValue(action1, lc1);
    ActionLookupValue.ActionLookupKey lc2 = new InjectedActionLookupKey("lc2");
    Artifact output2 =
        new Artifact.DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, lc2);
    CountDownLatch action2Running = new CountDownLatch(1);
    CountDownLatch topActionTestedOutput = new CountDownLatch(1);
    Action action2 =
        new TestAction(
            (Callable<Void> & Serializable)
                () -> {
                  action2Running.countDown();
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      topActionTestedOutput, "top ran");
                  return null;
                },
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(output2));
    ConfiguredTargetValue ctValue2 = createConfiguredTargetValue(action2, lc2);

    ActionLookupValue.ActionLookupKey topLc = new InjectedActionLookupKey("top");
    Artifact top =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), relativeOut.getChild("top"), topLc);
    Action topAction =
        new TestAction(
            (Callable<Void> & Serializable)
                () -> {
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      action2Running, "action 2 running");
                  try {
                    assertThat(output1.getPath().exists()).isTrue();
                  } finally {
                    topActionTestedOutput.countDown();
                  }
                  return null;
                },
            NestedSetBuilder.create(Order.STABLE_ORDER, output1),
            ImmutableSet.of(top));
    ConfiguredTargetValue ctTop = createConfiguredTargetValue(topAction, topLc);

    // Inject the "configured targets" and artifact into the graph.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lc1, ctValue1, lc2, ctValue2, topLc, ctTop));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.<Artifact>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<AspectValue>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        ImmutableSet.<ConfiguredTarget>of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    // NULL_CHECKER here means action cache, which would be our savior, is not in play.
    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    EvaluationResult<FileArtifactValue> result = evaluate(Artifact.keys(ImmutableList.of(output1)));
    assertThat(result.hasError()).isFalse();
    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    EvaluationResult<FileArtifactValue> result2 =
        evaluate(Artifact.keys(ImmutableList.of(top, output2)));
    assertThat(result2.hasError()).isFalse();
    TrackingAwaiter.INSTANCE.assertNoErrors();
  }

  /**
   * Tests that events from action lookup keys (i.e., analysis events) are not stored in execution.
   * This test is actually more extreme than Blaze is, since it skips the analysis phase and so
   * <i>never</i> emits the analysis events, while in reality Blaze will always emit the analysis
   * events, during the analysis phase.
   *
   * <p>Also incidentally tests that events coming from action execution are actually not stored at
   * all.
   */
  @Test
  public void analysisEventsNotStoredInExecution() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");
    ActionLookupValue.ActionLookupKey lc1 = new InjectedActionLookupKey("lc1");
    Artifact output =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("foo"), lc1);
    Action action1 = new WarningAction(ImmutableList.of(), output, "action 1");
    SkyValue ctValue1 =
        ValueWithMetadata.normal(
            createConfiguredTargetValue(action1, lc1),
            null,
            NestedSetBuilder.create(
                Order.STABLE_ORDER,
                new TaggedEvents(null, ImmutableList.of(Event.warn("analysis warning 1")))),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    ActionLookupValue.ActionLookupKey lc2 = new InjectedActionLookupKey("lc2");
    Artifact output2 =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("bar"), lc2);
    Action action2 = new WarningAction(ImmutableList.of(output), output2, "action 2");
    SkyValue ctValue2 =
        ValueWithMetadata.normal(
            createConfiguredTargetValue(action2, lc2),
            null,
            NestedSetBuilder.create(
                Order.STABLE_ORDER,
                new TaggedEvents(null, ImmutableList.of(Event.warn("analysis warning 2")))),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lc1, ctValue1, lc2, ctValue2));
    // Do a null build, so that the skyframe executor initializes the action executor properly.
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));
    skyframeExecutor.buildArtifacts(
        reporter,
        ResourceManager.instanceForTestingOnly(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.<ConfiguredTarget>of(),
        options,
        NULL_CHECKER,
        null,
        null,
        null);

    skyframeExecutor.prepareBuildingForTestingOnly(
        reporter, new DummyExecutor(fileSystem, rootDirectory), options, NULL_CHECKER, null);
    evaluate(ImmutableList.of(Artifact.key(output2)));
    assertContainsEvent("action 1");
    assertContainsEvent("action 2");
    assertDoesNotContainEvent("analysis warning 1");
    assertDoesNotContainEvent("analysis warning 2");

    // Action's warnings are not stored, and configured target warnings never seen.
    assertThat(
            ValueWithMetadata.getEvents(
                    skyframeExecutor
                        .getDriver()
                        .getEntryForTesting(ActionLookupData.create(lc1, 0))
                        .getValueMaybeWithMetadata())
                .toList())
        .isEmpty();
    assertThat(
            ValueWithMetadata.getEvents(
                    skyframeExecutor
                        .getDriver()
                        .getEntryForTesting(ActionLookupData.create(lc2, 0))
                        .getValueMaybeWithMetadata())
                .toList())
        .isEmpty();
  }

  private static class WarningAction extends AbstractAction {
    private final String warningText;

    private WarningAction(ImmutableList<Artifact> inputs, Artifact output, String warningText) {
      super(
          NULL_ACTION_OWNER,
          NestedSetBuilder.<Artifact>stableOrder().addAll(inputs).build(),
          ImmutableSet.of(output));
      this.warningText = warningText;
    }

    @Override
    public String getMnemonic() {
      return "warning action";
    }

    @Override
    protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
      fp.addString(warningText);
      fp.addPath(getPrimaryOutput().getExecPath());
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      actionExecutionContext.getEventHandler().handle(Event.warn(warningText));
      try {
        FileSystemUtils.createEmptyFile(actionExecutionContext.getInputPath(getPrimaryOutput()));
      } catch (IOException e) {
        throw new ActionExecutionException(e, this, false);
      }
      return ActionResult.EMPTY;
    }
  }

  /** Dummy action that throws a catastrophic error when it runs. */
  private static class CatastrophicAction extends DummyAction {
    public static final ExitCode expectedExitCode = ExitCode.RESERVED;

    CatastrophicAction(Artifact output) {
      super(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, MiddlemanType.NORMAL);
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      throw new ActionExecutionException(
          "message",
          new Exception("just cause"),
          this,
          /*catastrophe=*/ true,
          DetailedExitCode.justExitCode(expectedExitCode));
    }
  }

  /** Dummy action that flips a boolean when it runs. */
  private static class MarkerAction extends DummyAction {
    private final AtomicBoolean executed;

    MarkerAction(Artifact output, AtomicBoolean executed) {
      super(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, MiddlemanType.NORMAL);
      this.executed = executed;
      assertThat(executed.get()).isFalse();
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      ActionResult actionResult = super.execute(actionExecutionContext);
      assertThat(executed.getAndSet(true)).isFalse();
      return actionResult;
    }
  }

  private BinTools setupEmbeddedArtifacts() throws IOException {
    List<String> embeddedTools = analysisMock.getEmbeddedTools();
    directories.getEmbeddedBinariesRoot().createDirectoryAndParents();
    for (String embeddedToolName : embeddedTools) {
      Path toolPath = directories.getEmbeddedBinariesRoot().getRelative(embeddedToolName);
      FileSystemUtils.touchFile(toolPath);
    }
    return BinTools.forIntegrationTesting(directories, embeddedTools);
  }

  /** Test appropriate behavior when an action halts the build with a catastrophic failure. */
  private void runCatastropheHaltsBuild() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");
    ActionLookupValue.ActionLookupKey lc1 = new InjectedActionLookupKey("lc1");
    Artifact output =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("foo"), lc1);
    Action action1 = new CatastrophicAction(output);
    ConfiguredTargetValue ctValue1 = createConfiguredTargetValue(action1, lc1);
    ActionLookupValue.ActionLookupKey lc2 = new InjectedActionLookupKey("lc2");
    Artifact output2 =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("bar"), lc2);
    AtomicBoolean markerRan = new AtomicBoolean(false);
    Action action2 = new MarkerAction(output2, markerRan);
    ConfiguredTargetValue ctValue2 = createConfiguredTargetValue(action2, lc2);

    // Perform testing-related setup.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(lc1, ctValue1, lc2, ctValue2));
    skyframeExecutor.setEventBus(new EventBus());
    setupEmbeddedArtifacts();
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER, ActionExecutionStatusReporter.create(reporter));

    reporter.removeHandler(failFastHandler); // Expect errors.
    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /* fileCache= */ null,
            ActionInputPrefetcher.NONE);
    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();
    // Note that since ImmutableSet iterates through its elements in the order they are passed in
    // here, we are guaranteed that output will be built before output2, throwing an exception and
    // shutting down the build before output2 is requested.
    Set<Artifact> normalArtifacts = ImmutableSet.of(output, output2);
    BuildFailedException e =
        assertThrows(
            BuildFailedException.class,
            () ->
                builder.buildArtifacts(
                    reporter,
                    normalArtifacts,
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<AspectValue>of(),
                    new DummyExecutor(fileSystem, rootDirectory),
                    builtTargets,
                    builtAspects,
                    options,
                    null,
                    null,
                    /* trustRemoteArtifacts= */ false));
    // The catastrophic exception should be propagated into the BuildFailedException whether or not
    // --keep_going is set.
    assertThatDetailedExitCode(e.getDetailedExitCode())
        .hasExitCode(CatastrophicAction.expectedExitCode);
    assertThat(builtTargets).isEmpty();
    assertThat(markerRan.get()).isFalse();
  }

  private static NonRuleConfiguredTargetValue createConfiguredTargetValue(
      ActionAnalysisMetadata generatingAction, ActionLookupValue.ActionLookupKey actionLookupKey) {
    return new NonRuleConfiguredTargetValue(
        new SerializableConfiguredTarget(),
        GeneratingActions.fromSingleAction(generatingAction, actionLookupKey),
        NestedSetBuilder.<Package>stableOrder().build());
  }

  @Test
  public void testCatastropheInNoKeepGoing() throws Exception {
    options.parse("--nokeep_going", "--jobs=1");
    runCatastropheHaltsBuild();
  }

  @Test
  public void testCatastrophicBuild() throws Exception {
    options.parse("--keep_going", "--jobs=1");
    runCatastropheHaltsBuild();
  }

  /**
   * Test appropriate behavior when an action halts the build with a transitive catastrophic
   * failure.
   */
  @Test
  public void testTransitiveCatastropheHaltsBuild() throws Exception {
    options.parse("--keep_going", "--jobs=5");

    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");
    ActionLookupValue.ActionLookupKey catastropheCTK = new InjectedActionLookupKey("catastrophe");
    Artifact catastropheArtifact =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath.getRelative("zcatas"),
            catastropheCTK);
    CountDownLatch failureHappened = new CountDownLatch(1);
    Action catastrophicAction =
        new CatastrophicAction(catastropheArtifact) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext)
              throws ActionExecutionException {
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                failureHappened, "didn't count failure");
            return super.execute(actionExecutionContext);
          }
        };
    ConfiguredTargetValue catastropheCTV =
        createConfiguredTargetValue(catastrophicAction, catastropheCTK);
    ActionLookupValue.ActionLookupKey failureCTK = new InjectedActionLookupKey("failure");
    Artifact failureArtifact =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("fail"), failureCTK);
    Action failureAction = new FailedExecAction(failureArtifact, ExitCode.RESERVED);
    ConfiguredTargetValue failureCTV = createConfiguredTargetValue(failureAction, failureCTK);
    ActionLookupValue.ActionLookupKey topCTK = new InjectedActionLookupKey("top");
    Artifact topArtifact =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("top"), topCTK);
    Action topAction =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, failureArtifact, catastropheArtifact),
            topArtifact);
    ConfiguredTargetValue topCTV = createConfiguredTargetValue(topAction, topCTK);
    // Perform testing-related setup.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(
            ImmutableMap.of(
                catastropheCTK, catastropheCTV,
                failureCTK, failureCTV,
                topCTK, topCTV));
    skyframeExecutor
        .getDriver()
        .getGraphForTesting()
        .injectGraphTransformerForTesting(
            DeterministicHelper.makeTransformer(
                (key, type, order, context) -> {
                  if (key.equals(Artifact.key(failureArtifact)) && type == EventType.SET_VALUE) {
                    failureHappened.countDown();
                  }
                },
                /*deterministic=*/ true));
    skyframeExecutor.setEventBus(new EventBus());
    setupEmbeddedArtifacts();
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));

    reporter.removeHandler(failFastHandler); // Expect errors.
    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /*fileCache=*/ null,
            ActionInputPrefetcher.NONE);
    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();
    Set<Artifact> normalArtifacts = ImmutableSet.of(topArtifact);
    BuildFailedException e =
        assertThrows(
            BuildFailedException.class,
            () ->
                builder.buildArtifacts(
                    reporter,
                    normalArtifacts,
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<AspectValue>of(),
                    new DummyExecutor(fileSystem, rootDirectory),
                    builtTargets,
                    builtAspects,
                    options,
                    null,
                    null,
                    /* trustRemoteArtifacts= */ false));
    // The catastrophic exception should be propagated into the BuildFailedException whether or not
    // --keep_going is set.
    assertThatDetailedExitCode(e.getDetailedExitCode())
        .hasExitCode(CatastrophicAction.expectedExitCode);
    assertThat(builtTargets).isEmpty();
  }

  /**
   * Test appropriate behavior when an action halts the build with a transitive catastrophic
   * failure.
   */
  @Test
  public void testCatastropheAndNonCatastropheInCompletion() throws Exception {
    options.parse("--keep_going", "--jobs=5");

    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");
    ActionLookupValue.ActionLookupKey configuredTargetKey = new InjectedActionLookupKey("key");
    Artifact catastropheArtifact =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath.getRelative("catas"),
            configuredTargetKey);
    int failedSize = 100;
    CountDownLatch failureHappened = new CountDownLatch(failedSize);
    Action catastrophicAction =
        new CatastrophicAction(catastropheArtifact) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext)
              throws ActionExecutionException {
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                failureHappened, "didn't count failure");
            return super.execute(actionExecutionContext);
          }
        };
    // Because of random map ordering when getting values back in CompletionFunction, we just
    // sprinkle our failure nodes randomly about the alphabet, trusting that at least one will come
    // before "catas".
    List<Action> failedActions = new ArrayList<>(failedSize);
    LinkedHashSet<Artifact> failedArtifacts = new LinkedHashSet<>();
    for (int i = 0; i < failedSize; i++) {
      String failString = HashCode.fromBytes(("fail" + i).getBytes(UTF_8)).toString();
      Artifact failureArtifact =
          new Artifact.DerivedArtifact(
              ArtifactRoot.asDerivedRoot(root, "out"),
              execPath.getRelative(failString),
              configuredTargetKey);
      failedArtifacts.add(failureArtifact);
      failedActions.add(new FailedExecAction(failureArtifact, ExitCode.BUILD_FAILURE));
    }
    NonRuleConfiguredTargetValue nonRuleConfiguredTargetValue =
        new NonRuleConfiguredTargetValue(
            new SerializableConfiguredTarget(),
            Actions.assignOwnersAndFilterSharedActionsAndThrowActionConflict(
                new ActionKeyContext(),
                ImmutableList.<ActionAnalysisMetadata>builder()
                    .add(catastrophicAction)
                    .addAll(failedActions)
                    .build(),
                configuredTargetKey,
                /*outputFiles=*/ null),
            NestedSetBuilder.<Package>stableOrder().build());
    HashSet<ActionLookupData> failedActionKeys = new HashSet<>();
    for (Action failedAction : failedActions) {
      failedActionKeys.add(
          ((Artifact.DerivedArtifact) failedAction.getPrimaryOutput()).getGeneratingActionKey());
    }

    // Perform testing-related setup.
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(ImmutableMap.of(configuredTargetKey, nonRuleConfiguredTargetValue));
    skyframeExecutor
        .getDriver()
        .getGraphForTesting()
        .injectGraphTransformerForTesting(
            DeterministicHelper.makeTransformer(
                (key, type, order, context) -> {
                  if ((key instanceof ActionLookupData)
                      && failedActionKeys.contains(key)
                      && type == EventType.SET_VALUE) {
                    failureHappened.countDown();
                  }
                },
                // Determinism actually doesn't help here because the internal maps are still
                // effectively unordered.
                /*deterministic=*/ true));
    skyframeExecutor.setEventBus(new EventBus());
    setupEmbeddedArtifacts();
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));

    reporter.removeHandler(failFastHandler); // Expect errors.
    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /*fileCache=*/ null,
            ActionInputPrefetcher.NONE);
    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();
    BuildFailedException e =
        assertThrows(
            BuildFailedException.class,
            () ->
                builder.buildArtifacts(
                    reporter,
                    ImmutableSet.<Artifact>builder()
                        .addAll(failedArtifacts)
                        .add(catastropheArtifact)
                        .build(),
                    ImmutableSet.of(),
                    ImmutableSet.of(),
                    ImmutableSet.of(),
                    ImmutableSet.of(),
                    ImmutableSet.of(),
                    new DummyExecutor(fileSystem, rootDirectory),
                    builtTargets,
                    builtAspects,
                    options,
                    null,
                    new TopLevelArtifactContext(
                        /*runTestsExclusively=*/ false,
                        false,
                        OutputGroupInfo.determineOutputGroups(ImmutableList.of(), true)),
                    /* trustRemoteArtifacts= */ false));
    // The catastrophic exception should be propagated into the BuildFailedException whether or not
    // --keep_going is set.
    assertThatDetailedExitCode(e.getDetailedExitCode())
        .hasExitCode(CatastrophicAction.expectedExitCode);
    assertThat(builtTargets).isEmpty();
  }

  @Test
  public void testCatastrophicBuildWithoutEdges() throws Exception {
    options.parse("--keep_going", "--jobs=1", "--discard_analysis_cache");
    skyframeExecutor.setActive(false);
    skyframeExecutor.decideKeepIncrementalState(
        /*batch=*/ true,
        /*keepStateAfterBuild=*/ true,
        /*shouldTrackIncrementalState=*/ true,
        /*discardAnalysisCache=*/ true,
        reporter);
    skyframeExecutor.setActive(true);
    runCatastropheHaltsBuild();
  }

  @Test
  public void testCatastropheReportingWithError() throws Exception {
    options.parse("--keep_going", "--jobs=1");
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");
    // When we have an action that throws a (non-catastrophic) exception when it is executed,
    ActionLookupValue.ActionLookupKey failedKey = new InjectedActionLookupKey("failed");
    Artifact failedOutput =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("failed"), failedKey);
    final AtomicReference<Action> failedActionReference = new AtomicReference<>();
    final Action failedAction =
        new TestAction(
            new Callable<Void>() {
              @Override
              public Void call() throws ActionExecutionException {
                throw new ActionExecutionException(
                    new Exception(), failedActionReference.get(), /*catastrophe=*/ false);
              }
            },
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(failedOutput));
    ConfiguredTargetValue failedTarget = createConfiguredTargetValue(failedAction, failedKey);

    // And an action that throws a catastrophic exception when it is executed,
    ActionLookupValue.ActionLookupKey catastrophicKey = new InjectedActionLookupKey("catastrophic");
    Artifact catastrophicOutput =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath.getRelative("catastrophic"),
            catastrophicKey);
    Action catastrophicAction = new CatastrophicAction(catastrophicOutput);
    ConfiguredTargetValue catastrophicTarget =
        createConfiguredTargetValue(catastrophicAction, catastrophicKey);

    // And the relevant configured targets have been injected into the graph,
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(
            ImmutableMap.of(
                failedKey, failedTarget,
                catastrophicKey, catastrophicTarget));
    skyframeExecutor.setEventBus(new EventBus());
    setupEmbeddedArtifacts();
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));

    // And the two artifacts are requested,
    reporter.removeHandler(failFastHandler); // Expect errors.
    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /* fileCache= */ null,
            ActionInputPrefetcher.NONE);
    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();
    // Note that since ImmutableSet iterates through its elements in the order they are passed in
    // here, we are guaranteed that failedOutput will be built before catastrophicOutput is
    // requested, putting a top-level failure into the build result.
    Set<Artifact> normalArtifacts = ImmutableSet.of(failedOutput, catastrophicOutput);
    BuildFailedException e =
        assertThrows(
            BuildFailedException.class,
            () ->
                builder.buildArtifacts(
                    reporter,
                    normalArtifacts,
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<AspectValue>of(),
                    new DummyExecutor(fileSystem, rootDirectory),
                    builtTargets,
                    builtAspects,
                    options,
                    null,
                    null,
                    /* trustRemoteArtifacts= */ false));
    // The catastrophic exception should be propagated into the BuildFailedException whether or not
    // --keep_going is set.
    assertThatDetailedExitCode(e.getDetailedExitCode())
        .hasExitCode(CatastrophicAction.expectedExitCode);
    assertThat(builtTargets).isEmpty();
  }

  /** Dummy action that throws a ActionExecution error when it runs. */
  private static class FailedExecAction extends DummyAction {
    private final ExitCode exitCode;

    FailedExecAction(Artifact output, ExitCode exitCode) {
      super(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, MiddlemanType.NORMAL);
      this.exitCode = exitCode;
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      throw new ActionExecutionException(
          "foo",
          new Exception("bar"),
          this,
          /*catastrophe=*/ false,
          DetailedExitCode.justExitCode(exitCode));
    }
  }

  private static final ExitCode USER_EXIT_CODE = ExitCode.create(Integer.MAX_VALUE, "user_error");
  private static final ExitCode INFRA_EXIT_CODE =
      ExitCode.createInfrastructureFailure(Integer.MAX_VALUE - 1, "infra_error");

  /**
   * Verify SkyframeBuilder returns correct user error code as global error code when:
   *    1. keepGoing mode is true.
   *    2. user error code exists.
   *    3. no infrastructure error code exists.
   */
  @Test
  public void testKeepGoingExitCodeWithUserError() throws Exception {
    options.parse("--keep_going", "--jobs=1");
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");

    ActionLookupValue.ActionLookupKey succeededKey = new InjectedActionLookupKey("succeeded");
    Artifact succeededOutput =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath.getRelative("succeeded"),
            succeededKey);

    ActionLookupValue.ActionLookupKey failedKey = new InjectedActionLookupKey("failed");
    Artifact failedOutput =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("failed"), failedKey);

    // Create 1 succeeded key and 1 failed key with user error
    Action succeededAction =
        new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), succeededOutput);
    ConfiguredTargetValue succeededTarget =
        createConfiguredTargetValue(succeededAction, succeededKey);
    Action failedAction = new FailedExecAction(failedOutput, USER_EXIT_CODE);
    ConfiguredTargetValue failedTarget = createConfiguredTargetValue(failedAction, failedKey);

    // Inject the targets into the graph,
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(
            ImmutableMap.of(
                succeededKey, succeededTarget,
                failedKey, failedTarget));
    skyframeExecutor.setEventBus(new EventBus());
    setupEmbeddedArtifacts();
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));

    // And the two artifacts are requested,
    reporter.removeHandler(failFastHandler); // Expect errors.
    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /* fileCache= */ null,
            ActionInputPrefetcher.NONE);
    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();
    Set<Artifact> normalArtifacts = ImmutableSet.of(succeededOutput, failedOutput);
    BuildFailedException e =
        assertThrows(
            BuildFailedException.class,
            () ->
                builder.buildArtifacts(
                    reporter,
                    normalArtifacts,
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<AspectValue>of(),
                    new DummyExecutor(fileSystem, rootDirectory),
                    builtTargets,
                    builtAspects,
                    options,
                    null,
                    null,
                    /* trustRemoteArtifacts= */ false));
    // The exit code should be propagated into the BuildFailedException whether or not --keep_going
    // is set.
    assertThatDetailedExitCode(e.getDetailedExitCode()).hasExitCode(USER_EXIT_CODE);
  }

  /**
   * Verify SkyframeBuilder returns correct infrastructure error code as global error code when:
   *    1. keepGoing mode is true.
   *    2. infrastructure error code exists.
   */
  @Test
  public void testKeepGoingExitCodeWithUserAndInfrastructureError() throws Exception {
    options.parse("--keep_going", "--jobs=1");
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");

    ActionLookupValue.ActionLookupKey succeededKey = new InjectedActionLookupKey("succeeded");
    Artifact succeededOutput =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"),
            execPath.getRelative("succeeded"),
            succeededKey);

    ActionLookupValue.ActionLookupKey failedKey1 = new InjectedActionLookupKey("failed1");
    Artifact failedOutput1 =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("failed1"), failedKey1);

    ActionLookupValue.ActionLookupKey failedKey2 = new InjectedActionLookupKey("failed2");
    Artifact failedOutput2 =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("failed2"), failedKey2);

    // Create 1 succeeded key, 1 failed key with infrastructure error and another failed key with
    // user error.

    Action succeededAction =
        new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), succeededOutput);
    ConfiguredTargetValue succeededTarget =
        createConfiguredTargetValue(succeededAction, succeededKey);
    Action failedAction1 = new FailedExecAction(failedOutput1, USER_EXIT_CODE);
    ConfiguredTargetValue failedTarget1 = createConfiguredTargetValue(failedAction1, failedKey1);
    Action failedAction2 = new FailedExecAction(failedOutput2, INFRA_EXIT_CODE);
    ConfiguredTargetValue failedTarget2 = createConfiguredTargetValue(failedAction2, failedKey2);

    // Inject the targets into the graph,
    skyframeExecutor
        .getDifferencerForTesting()
        .inject(
            ImmutableMap.<SkyKey, SkyValue>of(
                succeededKey, succeededTarget,
                failedKey1, failedTarget1,
                failedKey2, failedTarget2));
    skyframeExecutor.setEventBus(new EventBus());
    setupEmbeddedArtifacts();
    skyframeExecutor.setActionOutputRoot(getOutputPath());
    skyframeExecutor.setActionExecutionProgressReportingObjects(
        EMPTY_PROGRESS_SUPPLIER,
        EMPTY_COMPLETION_RECEIVER,
        ActionExecutionStatusReporter.create(reporter));

    // And the two artifacts are requested,
    reporter.removeHandler(failFastHandler); // Expect errors.
    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /* fileCache= */ null,
            ActionInputPrefetcher.NONE);
    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();
    Set<Artifact> normalArtifacts = ImmutableSet.of(failedOutput1, failedOutput2);
    BuildFailedException e =
        assertThrows(
            BuildFailedException.class,
            () ->
                builder.buildArtifacts(
                    reporter,
                    normalArtifacts,
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<ConfiguredTarget>of(),
                    ImmutableSet.<AspectValue>of(),
                    new DummyExecutor(fileSystem, rootDirectory),
                    builtTargets,
                    builtAspects,
                    options,
                    null,
                    null,
                    /* trustRemoteArtifacts= */ false));
    // The exit code should be propagated into the BuildFailedException whether or not --keep_going
    // is set.
    assertThatDetailedExitCode(e.getDetailedExitCode()).hasExitCode(INFRA_EXIT_CODE);
  }

  /**
   * Tests that when an input-discovering action terminates input discovery with missing inputs, its
   * progress message goes away. We create an input-discovering action that declares a new input.
   * When that new input is declared, which comes after the scanning is completed, we trigger a
   * progress message, and assert that the message does not contain the "Scanning" message.
   *
   * <p>To guard against the output format changing, we also trigger a progress message during the
   * scan, and assert that the message there is as expected.
   */
  @Test
  public void inputDiscoveryMessageDoesntLinger() throws Exception {
    Path root = getExecRoot();
    PathFragment execPath = PathFragment.create("out").getRelative("dir");

    ActionLookupValue.ActionLookupKey topKey = new InjectedActionLookupKey("top");
    Artifact topOutput =
        new Artifact.DerivedArtifact(
            ArtifactRoot.asDerivedRoot(root, "out"), execPath.getRelative("top"), topKey);

    Artifact sourceInput =
        new Artifact.SourceArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(rootDirectory)),
            PathFragment.create("source.optional"),
            ArtifactOwner.NullArtifactOwner.INSTANCE);
    FileSystemUtils.createEmptyFile(sourceInput.getPath());

    Action inputDiscoveringAction =
        new DummyAction(NestedSetBuilder.create(Order.STABLE_ORDER, sourceInput), topOutput) {
          @Override
          public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
            skyframeExecutor
                .getActionExecutionStatusReporterForTesting()
                .showCurrentlyExecutingActions("during scanning ");
            return super.discoverInputs(actionExecutionContext);
          }
        };

    ConfiguredTargetValue topTarget = createConfiguredTargetValue(inputDiscoveringAction, topKey);
    skyframeExecutor.getDifferencerForTesting().inject(ImmutableMap.of(topKey, topTarget));
    // Collect all events.
    eventCollector = new EventCollector();
    reporter = new Reporter(eventBus, eventCollector);
    skyframeExecutor.setEventBus(eventBus);
    skyframeExecutor.setActionOutputRoot(getOutputPath());

    Builder builder =
        new SkyframeBuilder(
            skyframeExecutor,
            ResourceManager.instanceForTestingOnly(),
            NULL_CHECKER,
            null,
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /*fileCache=*/ null,
            ActionInputPrefetcher.NONE);
    builder.buildArtifacts(
        reporter,
        ImmutableSet.of(topOutput),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableSet.of(),
        new DummyExecutor(fileSystem, rootDirectory),
        ImmutableSet.of(),
        ImmutableSet.of(),
        options,
        null,
        null,
        /* trustRemoteArtifacts= */ false);
    assertContainsEventRegex(eventCollector, ".*during scanning.*\n.*Scanning.*\n.*Test dir/top.*");
    assertNotContainsEventRegex(
        eventCollector, ".*after scanning.*\n.*Scanning.*\n.*Test dir/top.*");
  }

  private AnalysisProtos.Artifact getArtifact(
      String execPath, ActionGraphContainer actionGraphContainer) {
    for (AnalysisProtos.Artifact artifact : actionGraphContainer.getArtifactsList()) {
      if (execPath.equals(artifact.getExecPath())) {
        return artifact;
      }
    }
    return null;
  }

  private AnalysisProtos.Artifact getArtifactFromBinDir(
      String workspaceRelativePath, ActionGraphContainer actionGraphContainer) {
    return getArtifact(
        getTargetConfiguration()
            .getBinDir()
            .getExecPath()
            .getRelative(workspaceRelativePath)
            .getPathString(),
        actionGraphContainer);
  }

  private AnalysisProtos.Action getGeneratingAction(
      String outputArtifactId, ActionGraphContainer actionGraphContainer) {
    for (AnalysisProtos.Action action : actionGraphContainer.getActionsList()) {
      for (String outputId : action.getOutputIdsList()) {
        if (outputArtifactId.equals(outputId)) {
          return action;
        }
      }
    }
    return null;
  }

  private AnalysisProtos.Target getTarget(String label, ActionGraphContainer actionGraphContainer) {
    for (AnalysisProtos.Target target : actionGraphContainer.getTargetsList()) {
      if (label.equals(target.getLabel())) {
        return target;
      }
    }
    return null;
  }

  private AnalysisProtos.AspectDescriptor getAspectDescriptor(
      String aspectDescriptorId, ActionGraphContainer actionGraphContainer) {
    for (AnalysisProtos.AspectDescriptor aspectDescriptor :
        actionGraphContainer.getAspectDescriptorsList()) {
      if (aspectDescriptorId.equals(aspectDescriptor.getId())) {
        return aspectDescriptor;
      }
    }
    return null;
  }

  private AnalysisProtos.RuleClass getRuleClass(
      String ruleClassId, ActionGraphContainer actionGraphContainer) {
    for (AnalysisProtos.RuleClass ruleClass : actionGraphContainer.getRuleClassesList()) {
      if (ruleClassId.equals(ruleClass.getId())) {
        return ruleClass;
      }
    }
    return null;
  }

  public static final ImmutableList<String> ACTION_GRAPH_DEFAULT_TARGETS = ImmutableList.of("...");

  @Test
  public void testActionGraphDumpWithoutInputArtifacts() throws Exception {
    scratch.file("x/BUILD", "genrule(name='x', srcs=['input'], outs=['out'], cmd='false')");
    scratch.file("x/input", "foo");

    ConfiguredTarget ct =
        skyframeExecutor.getConfiguredTargetForTesting(
            reporter, Label.parseAbsolute("@//x", ImmutableMap.of()), getTargetConfiguration());
    assertThat(ct).isNotNull();
    ActionGraphContainer actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ false);

    assertThat(actionGraphContainer.getActionsList()).isNotEmpty();
    assertThat(actionGraphContainer.getArtifactsList()).isEmpty();
    assertThat(actionGraphContainer.getDepSetOfFilesList()).isEmpty();
    assertThat(actionGraphContainer.getActionsList().get(0).getInputDepSetIdsList()).isEmpty();
    assertThat(actionGraphContainer.getActionsList().get(0).getOutputIdsList()).isEmpty();
  }

  @Test
  public void testActionGraphDumpBrokenAnalysis() throws Exception {
    scratch.file("x/BUILD", "java_library(name='x', exports=[':doesnotexist'])");

    reporter.removeHandler(failFastHandler);
    assertThat(
            skyframeExecutor.getConfiguredTargetForTesting(
                reporter, Label.parseAbsolute("@//x", ImmutableMap.of()), getTargetConfiguration()))
        .isNull();
    assertContainsEvent(
        "in exports attribute of java_library rule //x:x: rule '//x:doesnotexist' does not exist");
    ActionGraphContainer actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ true);
    assertThat(actionGraphContainer).isNotNull();
  }


  @Test
  public void testActionGraphDumpWithTreeArtifact() throws Exception {
    scratch.file(
        "x/def.bzl",
        "def _tree_impl(ctx):",
        "  tree_artifact = ctx.actions.declare_directory(ctx.attr.name + '_dir')",
        "  ctx.actions.run_shell(",
        "      inputs = [ctx.file.dummy],",
        "      outputs = [tree_artifact],",
        "      mnemonic = 'Treemove',",
        "      use_default_shell_env = True,",
        "      command = 'cp $1 $2',",
        "      arguments = [",
        "          ctx.file.dummy.path,",
        "          tree_artifact.path,",
        "      ],",
        "  )",
        "  return [",
        "      DefaultInfo(files=depset([tree_artifact])),",
        "  ]",
        "",
        "tree = rule(",
        "    implementation = _tree_impl,",
        "    attrs = {",
        "        'dummy': attr.label(allow_single_file = True),",
        "    },",
        ")");
    scratch.file(
        "x/BUILD",
        "load('//x:def.bzl', 'tree')",
        "tree(",
        "    name = 'tree',",
        "    dummy = 'foo.txt',",
        ")");
    scratch.file("x/foo.txt", "hello world");

    ConfiguredTarget ct =
        skyframeExecutor.getConfiguredTargetForTesting(
            reporter,
            Label.parseAbsolute("@//x:tree", ImmutableMap.of()),
            getTargetConfiguration());
    assertThat(ct).isNotNull();
    ActionGraphContainer actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ true);

    AnalysisProtos.Artifact inputArtifact = getArtifact("x/foo.txt", actionGraphContainer);
    assertThat(inputArtifact).isNotNull();
    assertThat(inputArtifact.getIsTreeArtifact()).isFalse();
    AnalysisProtos.Artifact outputArtifact =
        getArtifactFromBinDir("x/tree_dir", actionGraphContainer);
    assertThat(outputArtifact).isNotNull();
    assertThat(outputArtifact.getIsTreeArtifact()).isTrue();
    AnalysisProtos.Action action =
        getGeneratingAction(outputArtifact.getId(), actionGraphContainer);
    assertThat(action).isNotNull();
    assertThat(action.getMnemonic()).isEqualTo("Treemove");
  }

  @Test
  public void testActionGraphDumpWithAspect() throws Exception {
    scratch.file(
        "x/def.bzl",
        "Count = provider(",
        "    fields = {",
        "        'count' : 'count',",
        "        'out' : 'outputfile'",
        "    }",
        ")",
        "",
        "def _count_aspect_impl(target, ctx):",
        "    count = int(ctx.attr.default_count)",
        "    for dep in ctx.rule.attr.deps:",
        "        count = count + dep[Count].count",
        "    output = ctx.actions.declare_file('count')",
        "    ctx.actions.write(content = 'count = %s' % (count), output = output)",
        "    return [",
        "        Count(count = count, out = output),",
        "        OutputGroupInfo(all_files = [output]),",
        "    ]",
        "",
        "count_aspect = aspect(implementation = _count_aspect_impl,",
        "    attr_aspects = ['deps'],",
        "    attrs = {",
        "        'default_count' : attr.string(values = ['0', '1', '42']),",
        "    }",
        ")",
        "",
        "def _count_rule_impl(ctx):",
        "  outs = []",
        "  for dep in ctx.attr.deps:",
        "    outs += [dep[Count].out]",
        "  return DefaultInfo(files=depset(outs))",
        "",
        "count_rule = rule(",
        "  implementation = _count_rule_impl,",
        "  attrs = {",
        "      'deps' : attr.label_list(aspects = [count_aspect]),",
        "      'default_count' : attr.string(default = '1'),",
        "  },",
        ")");
    scratch.file(
        "x/BUILD",
        "load('//x:def.bzl', 'count_rule')",
        "",
        "count_rule(",
        "    name = 'bar',",
        ")",
        "",
        "count_rule(",
        "    name = 'foo',",
        "    deps = ['bar'],",
        ")");

    ConfiguredTarget ct =
        skyframeExecutor.getConfiguredTargetForTesting(
            reporter, Label.parseAbsolute("@//x:foo", ImmutableMap.of()), getTargetConfiguration());
    assertThat(ct).isNotNull();
    ActionGraphContainer actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ true);

    AnalysisProtos.Artifact countArtifact = getArtifactFromBinDir("x/count", actionGraphContainer);
    assertThat(countArtifact).isNotNull();
    AnalysisProtos.Target target = getTarget("//x:bar", actionGraphContainer);
    assertThat(target).isNotNull();
    AnalysisProtos.RuleClass ruleClass =
        getRuleClass(target.getRuleClassId(), actionGraphContainer);
    assertThat(ruleClass.getName()).isEqualTo("count_rule");
    AnalysisProtos.Action action = getGeneratingAction(countArtifact.getId(), actionGraphContainer);
    assertThat(action).isNotNull();
    assertThat(action.getTargetId()).isEqualTo(target.getId());
    String aspectDescriptorId = Iterables.getOnlyElement(action.getAspectDescriptorIdsList());
    AnalysisProtos.AspectDescriptor aspectDescriptor =
        getAspectDescriptor(aspectDescriptorId, actionGraphContainer);
    assertThat(aspectDescriptor.getName()).isEqualTo("//x:def.bzl%count_aspect");
    AnalysisProtos.KeyValuePair aspectParameter =
        Iterables.getOnlyElement(aspectDescriptor.getParametersList());
    assertThat(aspectParameter.getKey()).isEqualTo("default_count");
    assertThat(aspectParameter.getValue()).isEqualTo("1");
  }

  @Test
  public void testActionGraphDumpFilter() throws Exception {
    scratch.file(
        "x/BUILD",
        "genrule(name='x', srcs=['input'], outs=['intermediate1'], cmd='false')",
        "genrule(name='y', srcs=['intermediate1'], outs=['intermediate2'], cmd='false')",
        "genrule(name='z', srcs=['intermediate2'], outs=['output'], cmd='false')");
    scratch.file("x/input", "foo");

    ConfiguredTarget ct =
        skyframeExecutor.getConfiguredTargetForTesting(
            reporter, Label.parseAbsolute("@//x:z", ImmutableMap.of()), getTargetConfiguration());
    assertThat(ct).isNotNull();

    // Check unfiltered case first, all three targets should be there.
    ActionGraphContainer actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ true);
    for (String targetString : ImmutableList.of("//x:x", "//x:y", "//x:z")) {
      AnalysisProtos.Target target = getTarget(targetString, actionGraphContainer);
      assertThat(target).isNotNull();
    }

    // Now check filtered case, only the requested target should exist.
    actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ImmutableList.of("//x:y"),
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ true);
    for (String targetString : ImmutableList.of("//x:x", "//x:z")) {
      AnalysisProtos.Target target = getTarget(targetString, actionGraphContainer);
      assertThat(target).isNull();
    }
    AnalysisProtos.Target target = getTarget("//x:y", actionGraphContainer);
    assertThat(target).isNotNull();
    // Make sure that we also don't include actions for other targets.
    AnalysisProtos.Action action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());
    assertThat(action.getTargetId()).isEqualTo(target.getId());
  }

  @Test
  public void testActionGraphCmdLineDump() throws Exception {
    scratch.file(
        "x/def.bzl",
        "def _impl(ctx):",
        "    output = ctx.outputs.out",
        "    input = ctx.file.file",
        "    # The command may only access files declared in inputs.",
        "    ctx.actions.run_shell(",
        "        inputs=[input],",
        "        outputs=[output],",
        "        progress_message='Getting size of %s' % input.short_path,",
        "        command='stat -L -c%%s %s > %s' % (input.path, output.path))",
        "",
        "size = rule(",
        "    implementation=_impl,",
        "    attrs={'file': attr.label(mandatory=True, allow_single_file=True)},",
        "    outputs={'out': '%{name}.size'},",
        ")");
    scratch.file("x/BUILD",
        "load('//x:def.bzl', 'size')",
        "size(name = 'x', file = 'foo.txt')");
    scratch.file("x/foo.txt",
        "foo");

    ConfiguredTarget ct =
        skyframeExecutor.getConfiguredTargetForTesting(
            reporter, Label.parseAbsolute("@//x", ImmutableMap.of()), getTargetConfiguration());
    assertThat(ct).isNotNull();

    // Check case without command line first.
    ActionGraphContainer actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ false,
            /* includeArtifacts= */ true);
    AnalysisProtos.Action action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());
    assertThat(action.getArgumentsCount()).isEqualTo(0);

    // Now check with command line.
    actionGraphContainer =
        skyframeExecutor.getActionGraphContainer(
            ACTION_GRAPH_DEFAULT_TARGETS,
            /* includeActionCmdLine= */ true,
            /* includeArtifacts= */ true);
    action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());

    List<String> args = action.getArgumentsList();
    assertThat(args).hasSize(3);
    assertThat(args.get(0)).matches("^.*(/bash|/bash.exe)$");
    assertThat(args.get(1)).isEqualTo("-c");
    assertThat(args.get(2)).startsWith("stat -L -c%s x/foo.txt > ");
    assertThat(args.get(2)).endsWith("bin/x/x.size");
  }

  /** Use custom class instead of mock to make sure that the dynamic codecs lookup is correct. */
  static class SerializableConfiguredTarget implements ConfiguredTarget {

    @Override
    public ImmutableCollection<String> getFieldNames() {
      return null;
    }

    @Nullable
    @Override
    public String getErrorMessageForUnknownField(String field) {
      return null;
    }

    @Nullable
    @Override
    public Object getValue(String name) {
      return null;
    }

    @Override
    public Label getLabel() {
      return null;
    }

    @Nullable
    @Override
    public BuildConfigurationValue.Key getConfigurationKey() {
      return null;
    }

    @Nullable
    @Override
    public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
      return null;
    }

    @Nullable
    @Override
    public Object get(String providerKey) {
      return null;
    }

    @Override
    public <T extends Info> T get(NativeProvider<T> provider) {
      return provider.getValueClass().cast(get(provider.getKey()));
    }

    @Nullable
    @Override
    public Info get(Provider.Key providerKey) {
      return null;
    }

    @Override
    public void repr(Printer printer) {}

    @Override
    public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
      return null;
    }

    @Override
    public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
      return false;
    }
  }
}
