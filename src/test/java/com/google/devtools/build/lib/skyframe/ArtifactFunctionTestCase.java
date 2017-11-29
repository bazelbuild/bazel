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
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.LegacySkyKey;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;

abstract class ArtifactFunctionTestCase {
  static final ActionLookupKey ALL_OWNER = new SingletonActionLookupKey();
  static final SkyKey OWNER_KEY = LegacySkyKey.create(SkyFunctions.ACTION_LOOKUP, ALL_OWNER);

  protected Predicate<PathFragment> allowedMissingInputsPredicate = Predicates.alwaysFalse();

  protected Set<ActionAnalysisMetadata> actions;
  protected boolean fastDigest = false;
  protected RecordingDifferencer differencer = new SequencedRecordingDifferencer();
  protected SequentialBuildDriver driver;
  protected MemoizingEvaluator evaluator;
  protected Path root;
  protected Path middlemanPath;

  /**
   * The test action execution function. The Skyframe evaluator's action execution function
   * delegates to this one.
   */
  protected SkyFunction delegateActionExecutionFunction;

  @Before
  public void baseSetUp() throws Exception  {
    setupRoot(new CustomInMemoryFs());
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                root.getFileSystem().getPath("/outputbase"),
                ImmutableList.of(root),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(new ServerDirectories(root, root), root, TestConstants.PRODUCT_NAME);
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(
        pkgLocator,
        ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
        directories);
    differencer = new SequencedRecordingDifferencer();
    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(
                    SkyFunctions.FILE_STATE,
                    new FileStateFunction(
                        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper))
                .put(SkyFunctions.FILE, new FileFunction(pkgLocator))
                .put(SkyFunctions.ARTIFACT, new ArtifactFunction(allowedMissingInputsPredicate))
                .put(SkyFunctions.ACTION_EXECUTION, new SimpleActionExecutionFunction())
                .put(
                    SkyFunctions.PACKAGE,
                    new PackageFunction(null, null, null, null, null, null, null))
                .put(
                    SkyFunctions.PACKAGE_LOOKUP,
                    new PackageLookupFunction(
                        null,
                        CrossRepositoryLabelViolationStrategy.ERROR,
                        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY))
                .put(
                    SkyFunctions.WORKSPACE_AST,
                    new WorkspaceASTFunction(TestRuleClassProvider.getRuleClassProvider()))
                .put(
                    SkyFunctions.WORKSPACE_FILE,
                    new WorkspaceFileFunction(
                        TestRuleClassProvider.getRuleClassProvider(),
                        TestConstants.PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING
                            .builder()
                            .build(
                                TestRuleClassProvider.getRuleClassProvider(), root.getFileSystem()),
                        directories))
                .put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction())
                .put(
                    SkyFunctions.ACTION_TEMPLATE_EXPANSION,
                    new ActionTemplateExpansionFunction(Suppliers.ofInstance(false)))
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    actions = new HashSet<>();
  }

  protected void setupRoot(CustomInMemoryFs fs) throws IOException {
    Path tmpDir = fs.getPath(TestUtils.tmpDir());
    root = tmpDir.getChild("root");
    FileSystemUtils.createDirectoryAndParents(root);
    FileSystemUtils.createEmptyFile(root.getRelative("WORKSPACE"));
    middlemanPath = tmpDir.getChild("middlemanRoot");
    FileSystemUtils.createDirectoryAndParents(middlemanPath);
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
    protected SkyKey getSkyKeyInternal() {
      return OWNER_KEY;
    }

    @Override
    protected SkyFunctionName getType() {
      throw new UnsupportedOperationException();
    }
  }

  /** InMemoryFileSystem that can pretend to do a fast digest. */
  protected class CustomInMemoryFs extends InMemoryFileSystem {
    @Override
    protected byte[] getFastDigest(Path path, HashFunction hashFunction) throws IOException {
      return fastDigest ? getDigest(path, hashFunction) : null;
    }
  }
}
