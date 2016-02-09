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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.junit.Before;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

abstract class ArtifactFunctionTestCase {
  protected static final SkyKey OWNER_KEY = new SkyKey(SkyFunctions.ACTION_LOOKUP, "OWNER");
  protected static final ActionLookupKey ALL_OWNER = new SingletonActionLookupKey();

  protected Predicate<PathFragment> allowedMissingInputsPredicate = Predicates.alwaysFalse();

  protected Set<Action> actions;
  protected boolean fastDigest = false;
  protected RecordingDifferencer differencer = new RecordingDifferencer();
  protected SequentialBuildDriver driver;
  protected MemoizingEvaluator evaluator;
  protected Path root;
  protected TimestampGranularityMonitor tsgm =
      new TimestampGranularityMonitor(BlazeClock.instance());

  /**
   * The test action execution function. The Skyframe evaluator's action execution function
   * delegates to this one.
   */
  protected SkyFunction delegateActionExecutionFunction;

  @Before
  public void baseSetUp() throws Exception  {
    setupRoot(new CustomInMemoryFs());
    AtomicReference<PathPackageLocator> pkgLocator = new AtomicReference<>(new PathPackageLocator(
        root.getFileSystem().getPath("/outputbase"), ImmutableList.of(root)));
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(pkgLocator, false);
    differencer = new RecordingDifferencer();
    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(SkyFunctions.FILE_STATE, new FileStateFunction(tsgm, externalFilesHelper))
                .put(SkyFunctions.FILE, new FileFunction(pkgLocator))
                .put(SkyFunctions.ARTIFACT,
                    new ArtifactFunction(allowedMissingInputsPredicate))
                .put(SkyFunctions.ACTION_EXECUTION, new SimpleActionExecutionFunction())
                .put(
                    SkyFunctions.PACKAGE,
                    new PackageFunction(null, null, null, null, null, null, null))
                .put(SkyFunctions.PACKAGE_LOOKUP, new PackageLookupFunction(null))
                .put(
                    SkyFunctions.WORKSPACE_AST,
                    new WorkspaceASTFunction(TestRuleClassProvider.getRuleClassProvider()))
                .put(
                    SkyFunctions.WORKSPACE_FILE,
                    new WorkspaceFileFunction(
                        TestRuleClassProvider.getRuleClassProvider(),
                        new PackageFactory(TestRuleClassProvider.getRuleClassProvider()),
                        new BlazeDirectories(root, root, root)))
                .put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction())
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    actions = new HashSet<>();
  }

  protected void setupRoot(CustomInMemoryFs fs) throws IOException {
    root = fs.getPath(TestUtils.tmpDir());
    FileSystemUtils.createDirectoryAndParents(root);
    FileSystemUtils.createEmptyFile(root.getRelative("WORKSPACE"));
  }

  protected static void writeFile(Path path, String contents) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(path, contents);
  }

  /** ActionExecutionFunction that delegates to our delegate. */
  private class SimpleActionExecutionFunction implements SkyFunction {
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      return delegateActionExecutionFunction.compute(skyKey, env);
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return delegateActionExecutionFunction.extractTag(skyKey);
    }
  }

  private static class SingletonActionLookupKey extends ActionLookupKey {
    @Override
    SkyKey getSkyKey() {
      return OWNER_KEY;
    }

    @Override
    SkyFunctionName getType() {
      throw new UnsupportedOperationException();
    }
  }

  /** InMemoryFileSystem that can pretend to do a fast digest. */
  protected class CustomInMemoryFs extends InMemoryFileSystem {
    @Override
    protected String getFastDigestFunctionType(Path path) {
      return fastDigest ? "MD5" : null;
    }

    @Override
    protected byte[] getFastDigest(Path path) throws IOException {
      return fastDigest ? getMD5Digest(path) : null;
    }
  }
}
