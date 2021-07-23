// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ARTIFACT_OWNER;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.createArtifact;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.writeContentAsLatin1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.CompactPersistentActionCache;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics.MissDetail;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics.MissReason;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.FakeArtifactResolverBase;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.FakeMetadataHandlerBase;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.MissDetailsBuilder;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ActionCacheCheckerTest {
  private CorruptibleActionCache cache;
  private ActionCacheChecker cacheChecker;
  private Set<Path> filesToDelete;
  private DigestHashFunction digestHashFunction;
  private FileSystem fileSystem;
  private ArtifactRoot artifactRoot;

  @Before
  public void setupCache() throws Exception {
    Scratch scratch = new Scratch();
    Clock clock = new ManualClock();

    cache = new CorruptibleActionCache(scratch.resolve("/cache/test.dat"), clock);
    cacheChecker = createActionCacheChecker(/* storeOutputMetadata= */ false);
    digestHashFunction = DigestHashFunction.SHA256;
    fileSystem = new InMemoryFileSystem(digestHashFunction);
    artifactRoot = ArtifactRoot.asDerivedRoot(
        fileSystem.getPath("/output"),
        RootType.Output,
        "bin");
  }

  private byte[] digest(byte[] content) {
    return digestHashFunction.getHashFunction().hashBytes(content).asBytes();
  }

  private ActionCacheChecker createActionCacheChecker(boolean storeOutputMetadata) {
    return new ActionCacheChecker(
            cache,
            new FakeArtifactResolverBase(),
            new ActionKeyContext(),
            action -> true,
            ActionCacheChecker.CacheConfig.builder()
                .setEnabled(true)
                .setVerboseExplanations(false)
                .setStoreOutputMetadata(storeOutputMetadata)
                .build(),
            PathFragment.create("bazel-out"));
  }

  @Before
  public void clearFilesToDeleteAfterTest() throws Exception {
    filesToDelete = new HashSet<>();
  }

  @After
  public void deleteFilesCreatedDuringTest() throws Exception {
    for (Path path : filesToDelete) {
      path.delete();
    }
  }

  /** "Executes" the given action from the point of view of the cache's lifecycle. */
  private void runAction(Action action) throws Exception {
    runAction(action, new HashMap<>());
  }

  /**
   * "Executes" the given action from the point of view of the cache's lifecycle with a custom
   * client environment.
   */
  private void runAction(Action action, Map<String, String> clientEnv) throws Exception {
    runAction(action, clientEnv, ImmutableMap.of());
  }

  private void runAction(Action action, Map<String, String> clientEnv, Map<String, String> platform)
      throws Exception {
    MetadataHandler metadataHandler = new FakeMetadataHandler();

    for (Artifact artifact : action.getOutputs()) {
      Path path = artifact.getPath();

      // Record all action outputs as files to be deleted across tests to prevent cross-test
      // pollution.  We need to do this on a path basis because we don't know upfront which file
      // system they live in so we cannot just recreate the file system.  (E.g. all NullActions
      // share an in-memory file system to hold dummy outputs.)
      filesToDelete.add(path);

      Path parent = path.getParentDirectory();
      if (parent != null) {
        parent.createDirectoryAndParents();
      }
    }

    Token token =
        cacheChecker.getTokenIfNeedToExecute(
            action,
            /*resolvedCacheArtifacts=*/ null,
            clientEnv,
            /*handler=*/ null,
            metadataHandler,
            /*artifactExpander=*/ null,
            platform);
    if (token != null) {
      // Real action execution would happen here.
      ActionExecutionContext context = mock(ActionExecutionContext.class);
      when(context.getMetadataHandler()).thenReturn(metadataHandler);
      action.execute(context);

      cacheChecker.updateActionCache(
          action, token, metadataHandler, /*artifactExpander=*/ null, clientEnv, platform);
    }
  }

  /** Ensures that the cache statistics match exactly the given values. */
  private void assertStatistics(int hits, Iterable<MissDetail> misses) {
    ActionCacheStatistics.Builder builder = ActionCacheStatistics.newBuilder();
    cache.mergeIntoActionCacheStatistics(builder);
    ActionCacheStatistics stats = builder.build();

    assertThat(stats.getHits()).isEqualTo(hits);
    assertThat(stats.getMissDetailsList()).containsExactlyElementsIn(misses);
  }

  private void doTestNotCached(Action action, MissReason missReason) throws Exception {
    runAction(action);

    assertStatistics(0, new MissDetailsBuilder().set(missReason, 1).build());
  }

  private void doTestCached(Action action, MissReason missReason) throws Exception {
    int runs = 5;
    for (int i = 0; i < runs; i++) {
      runAction(action);
    }

    assertStatistics(runs - 1, new MissDetailsBuilder().set(missReason, 1).build());
  }

  private void doTestCorruptedCacheEntry(Action action) throws Exception {
    cache.corruptAllEntries();
    runAction(action);

    assertStatistics(
        0,
        new MissDetailsBuilder().set(MissReason.CORRUPTED_CACHE_ENTRY, 1).build());
  }

  @Test
  public void testNoActivity() throws Exception {
    assertStatistics(0, new MissDetailsBuilder().build());
  }

  @Test
  public void testNotCached() throws Exception {
    doTestNotCached(new WriteEmptyOutputAction(), MissReason.NOT_CACHED);
  }

  @Test
  public void testCached() throws Exception {
    doTestCached(new WriteEmptyOutputAction(), MissReason.NOT_CACHED);
  }

  @Test
  public void testCorruptedCacheEntry() throws Exception {
    doTestCorruptedCacheEntry(new WriteEmptyOutputAction());
  }

  @Test
  public void testDifferentActionKey() throws Exception {
    Action action =
        new WriteEmptyOutputAction() {
          @Override
          protected void computeKey(
              ActionKeyContext actionKeyContext,
              @Nullable ArtifactExpander artifactExpander,
              Fingerprint fp) {
            fp.addString("key1");
          }
        };
    runAction(action);
    action =
        new NullAction() {
          @Override
          protected void computeKey(
              ActionKeyContext actionKeyContext,
              @Nullable ArtifactExpander artifactExpander,
              Fingerprint fp) {
            fp.addString("key2");
          }
        };
    runAction(action);

    assertStatistics(
        0,
        new MissDetailsBuilder()
            .set(MissReason.DIFFERENT_ACTION_KEY, 1)
            .set(MissReason.NOT_CACHED, 1)
            .build());
  }

  @Test
  public void testDifferentEnvironment() throws Exception {
    Action action =
        new WriteEmptyOutputAction() {
          @Override
          public ImmutableList<String> getClientEnvironmentVariables() {
            return ImmutableList.of("used-var");
          }
        };
    Map<String, String> clientEnv = new HashMap<>();
    clientEnv.put("unused-var", "1");
    runAction(action, clientEnv);  // Not cached.
    clientEnv.remove("unused-var");
    runAction(action, clientEnv);  // Cache hit because we only modified uninteresting variables.
    clientEnv.put("used-var", "2");
    runAction(action, clientEnv);  // Cache miss because of different environment.
    runAction(action, clientEnv);  // Cache hit because we did not change anything.

    assertStatistics(
        2,
        new MissDetailsBuilder()
            .set(MissReason.DIFFERENT_ENVIRONMENT, 1)
            .set(MissReason.NOT_CACHED, 1)
            .build());
  }

  @Test
  public void testDifferentRemoteDefaultPlatform() throws Exception {
    Action action = new WriteEmptyOutputAction();
    Map<String, String> env = new HashMap<>();
    env.put("unused-var", "1");

    Map<String, String> platform = new HashMap<>();
    platform.put("used-var", "1");
    // Not cached.
    runAction(action, env, platform);
    // Cache hit because nothing changed.
    runAction(action, env, platform);
    // Cache miss because platform changed to an empty from a previous value.
    runAction(action, env, ImmutableMap.of());
    // Cache hit with an empty platform.
    runAction(action, env, ImmutableMap.of());
    // Cache miss because platform changed to a value from an empty one.
    runAction(action, env, ImmutableMap.copyOf(platform));
    platform.put("another-var", "1234");
    // Cache miss because platform value changed.
    runAction(action, env, ImmutableMap.copyOf(platform));

    assertStatistics(
        2,
        new MissDetailsBuilder()
            .set(MissReason.DIFFERENT_ENVIRONMENT, 3)
            .set(MissReason.NOT_CACHED, 1)
            .build());
  }

  @Test
  public void testDifferentFiles() throws Exception {
    Action action = new WriteEmptyOutputAction();
    runAction(action);  // Not cached.
    writeContentAsLatin1(action.getPrimaryOutput().getPath(), "modified");
    runAction(action);  // Cache miss because output files were modified.

    assertStatistics(
        0,
        new MissDetailsBuilder()
            .set(MissReason.DIFFERENT_FILES, 1)
            .set(MissReason.NOT_CACHED, 1)
            .build());
  }

  @Test
  public void testUnconditionalExecution() throws Exception {
    Action action = new WriteEmptyOutputAction() {
      @Override
      public boolean executeUnconditionally() {
        return true;
      }

      @Override
      public boolean isVolatile() {
        return true;
      }
    };

    int runs = 5;
    for (int i = 0; i < runs; i++) {
      runAction(action);
    }

    assertStatistics(
        0, new MissDetailsBuilder().set(MissReason.UNCONDITIONAL_EXECUTION, runs).build());
  }

  @Test
  public void testMiddleman_notCached() throws Exception {
    doTestNotCached(new NullMiddlemanAction(), MissReason.DIFFERENT_DEPS);
  }

  @Test
  public void testMiddleman_cached() throws Exception {
    doTestCached(new NullMiddlemanAction(), MissReason.DIFFERENT_DEPS);
  }

  @Test
  public void testMiddleman_corruptedCacheEntry() throws Exception {
    doTestCorruptedCacheEntry(new NullMiddlemanAction());
  }

  @Test
  public void testMiddleman_differentFiles() throws Exception {
    Action action =
        new NullMiddlemanAction() {
          @Override
          public synchronized NestedSet<Artifact> getInputs() {
            FileSystem fileSystem = getPrimaryOutput().getPath().getFileSystem();
            Path path = fileSystem.getPath("/input");
            ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(fileSystem.getPath("/")));
            return NestedSetBuilder.create(
                Order.STABLE_ORDER, ActionsTestUtil.createArtifact(root, path));
          }
        };
    runAction(action);  // Not cached so recorded as different deps.
    writeContentAsLatin1(action.getPrimaryInput().getPath(), "modified");
    runAction(action);  // Cache miss because input files were modified.
    writeContentAsLatin1(action.getPrimaryOutput().getPath(), "modified");
    runAction(action);  // Outputs are not considered for middleman actions, so this is a cache hit.
    runAction(action);  // Outputs are not considered for middleman actions, so this is a cache hit.

    assertStatistics(
        2,
        new MissDetailsBuilder()
            .set(MissReason.DIFFERENT_DEPS, 1)
            .set(MissReason.DIFFERENT_FILES, 1)
            .build());
  }

  @Test
  public void testDeletedConstantMetadataOutputCausesReexecution() throws Exception {
    SpecialArtifact output =
        new Artifact.SpecialArtifact(
            artifactRoot,
            PathFragment.create("bin/dummy"),
            NULL_ARTIFACT_OWNER,
            SpecialArtifactType.CONSTANT_METADATA);
    output.getPath().getParentDirectory().createDirectoryAndParents();
    Action action = new WriteEmptyOutputAction(output);
    runAction(action);
    output.getPath().delete();
    assertThat(
            cacheChecker.getTokenIfNeedToExecute(
                action,
                null,
                ImmutableMap.of(),
                null,
                new FakeMetadataHandler(),
                null,
                ImmutableMap.of()))
        .isNotNull();
  }

  private RemoteFileArtifactValue createRemoteMetadata(String content) {
    byte[] bytes = content.getBytes(UTF_8);
    return new RemoteFileArtifactValue(digest(bytes), bytes.length, 1, "action-id");
  }

  @Test
  public void saveOutputMetadata_remoteFileMetadataSaved() throws Exception {
    cacheChecker = createActionCacheChecker(/* storeOutputMetadata= */ true);
    Artifact output = createArtifact(artifactRoot, "bin/dummy");
    String content = "content";
    Action action = new InjectOutputMetadataAction(output, createRemoteMetadata(content));

    // Not cached.
    runAction(action);

    assertThat(output.getPath().exists()).isFalse();
    ActionCache.Entry entry = cache.get(output.getExecPathString());
    assertThat(entry).isNotNull();
    assertThat(entry.getOutputFile(output)).isEqualTo(createRemoteMetadata(content));
    assertStatistics(0, new MissDetailsBuilder().set(MissReason.NOT_CACHED, 1).build());
  }

  @Test
  public void saveOutputMetadata_localFileMetadataNotSaved() throws Exception {
    cacheChecker = createActionCacheChecker(/* storeOutputMetadata= */ true);
    Artifact output = createArtifact(artifactRoot, "bin/dummy");
    Action action = new WriteEmptyOutputAction(output);
    output.getPath().delete();

    // Not cached.
    runAction(action);

    assertThat(output.getPath().exists()).isTrue();
    ActionCache.Entry entry = cache.get(output.getExecPathString());
    assertThat(entry).isNotNull();
    assertThat(entry.getOutputFile(output)).isNull();
    assertStatistics(0, new MissDetailsBuilder().set(MissReason.NOT_CACHED, 1).build());
  }

  @Test
  public void saveOutputMetadata_notSavedIfDisabled() throws Exception {
    Artifact output = createArtifact(artifactRoot, "bin/dummy");
    String content = "content";
    Action action = new InjectOutputMetadataAction(output, createRemoteMetadata(content));

    // Not cached.
    runAction(action);

    assertThat(output.getPath().exists()).isFalse();
    ActionCache.Entry entry = cache.get(output.getExecPathString());
    assertThat(entry).isNotNull();
    assertThat(entry.getOutputFile(output)).isNull();
    assertStatistics(0, new MissDetailsBuilder().set(MissReason.NOT_CACHED, 1).build());
  }

  @Test
  public void saveOutputMetadata_remoteFileMetadataLoaded() throws Exception {
    cacheChecker = createActionCacheChecker(/* storeOutputMetadata= */ true);
    Artifact output = createArtifact(artifactRoot, "bin/dummy");
    String content = "content";
    Action action = new InjectOutputMetadataAction(output, createRemoteMetadata(content));
    MetadataHandler metadataHandler = new FakeMetadataHandler();

    // Not cached.
    runAction(action);
    Token token = cacheChecker.getTokenIfNeedToExecute(
        action,
        null,
        ImmutableMap.of(),
        null,
        metadataHandler,
        null,
        ImmutableMap.of());

    assertThat(output.getPath().exists()).isFalse();
    assertThat(token).isNull();
    ActionCache.Entry entry = cache.get(output.getExecPathString());
    assertThat(entry).isNotNull();
    assertThat(entry.getOutputFile(output)).isEqualTo(createRemoteMetadata(content));
    assertThat(metadataHandler.getMetadata(output)).isEqualTo(createRemoteMetadata(content));
  }

  @Test
  public void saveOutputMetadata_localMetadataIsSameAsRemoteMetadata_cached() throws Exception {
    cacheChecker = createActionCacheChecker(/* storeOutputMetadata= */ true);
    Artifact output = createArtifact(artifactRoot, "bin/dummy");
    String content = "content";
    Action action = new InjectOutputMetadataAction(output, createRemoteMetadata(content));

    // Not cached
    runAction(action);
    writeContentAsLatin1(output.getPath(), content);
    // Cached since local metadata is same as remote metadata
    runAction(action);

    assertStatistics(
        1,
        new MissDetailsBuilder()
            .set(MissReason.NOT_CACHED, 1)
            .build());
    ActionCache.Entry entry = cache.get(output.getExecPathString());
    assertThat(entry).isNotNull();
    assertThat(entry.getOutputFile(output)).isEqualTo(createRemoteMetadata(content));
  }

  @Test
  public void saveOutputMetadata_localMetadataIsDifferentFromRemoteMetadata_notCached() throws Exception {
    cacheChecker = createActionCacheChecker(/* storeOutputMetadata= */ true);
    Artifact output = createArtifact(artifactRoot, "bin/dummy");
    String content1 = "content1";
    String content2 = "content2";
    Action action =
        new InjectOutputMetadataAction(
            output, createRemoteMetadata(content1), createRemoteMetadata(content2));

    // Not cached
    runAction(action);
    writeContentAsLatin1(output.getPath(), content2);
    // Not cached since local file changed
    runAction(action);

    assertStatistics(
        0,
        new MissDetailsBuilder()
            .set(MissReason.NOT_CACHED, 1)
            .set(MissReason.DIFFERENT_FILES, 1)
            .build());
    ActionCache.Entry entry = cache.get(output.getExecPathString());
    assertThat(entry).isNotNull();
    assertThat(entry.getOutputFile(output)).isEqualTo(createRemoteMetadata(content2));
  }

  /** An {@link ActionCache} that allows injecting corruption for testing. */
  private static class CorruptibleActionCache implements ActionCache {
    private final CompactPersistentActionCache delegate;
    private boolean corrupted = false;

    CorruptibleActionCache(Path cacheRoot, Clock clock) throws IOException {
      this.delegate =
          CompactPersistentActionCache.create(cacheRoot, clock, NullEventHandler.INSTANCE);
    }

    void corruptAllEntries() {
      corrupted = true;
    }

    @Override
    public Entry get(String key) {
      return corrupted ? ActionCache.Entry.CORRUPTED : delegate.get(key);
    }

    @Override
    public void put(String key, Entry entry) {
      delegate.put(key, entry);
    }

    @Override
    public void remove(String key) {
      delegate.remove(key);
    }

    @Override
    public long save() throws IOException {
      return delegate.save();
    }

    @Override
    public void clear() {
      delegate.clear();
    }

    @Override
    public void dump(PrintStream out) {
      delegate.dump(out);
    }

    @Override
    public void accountHit() {
      delegate.accountHit();
    }

    @Override
    public void accountMiss(MissReason reason) {
      delegate.accountMiss(reason);
    }

    @Override
    public void mergeIntoActionCacheStatistics(ActionCacheStatistics.Builder builder) {
      delegate.mergeIntoActionCacheStatistics(builder);
    }

    @Override
    public void resetStatistics() {
      delegate.resetStatistics();
    }
  }

  /** A null middleman action. */
  private static class NullMiddlemanAction extends NullAction {
    @Override
    public MiddlemanType getActionType() {
      return MiddlemanType.RUNFILES_MIDDLEMAN;
    }
  }

  /** A fake metadata handler that is able to obtain metadata from the file system. */
  private static class FakeMetadataHandler extends FakeMetadataHandlerBase {
    private final Map<Artifact, FileArtifactValue> fileMetadata = new HashMap<>();
    private final Map<SpecialArtifact, TreeArtifactValue> treeMetadata = new HashMap<>();

    @Override
    public void injectFile(Artifact output, FileArtifactValue metadata) {
      fileMetadata.put(output, metadata);
    }

    @Override
    public void injectTree(SpecialArtifact treeArtifact, TreeArtifactValue tree) {
      treeMetadata.put(treeArtifact, tree);
    }

    @Override
    public FileArtifactValue getMetadata(ActionInput input) throws IOException {
      if (!(input instanceof Artifact)) {
        return null;
      }
      Artifact output = (Artifact) input;
      if (fileMetadata.containsKey(output)) {
        return fileMetadata.get(output);
      }
      return FileArtifactValue.createForTesting(output);
    }

    @Override
    public void setDigestForVirtualArtifact(Artifact artifact, byte[] digest) {}
  }

  private static class WriteEmptyOutputAction extends NullAction {
    public WriteEmptyOutputAction() {
    }

    public WriteEmptyOutputAction(Artifact... outputs) {
      super(outputs);
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext) {
      for (Artifact output : getOutputs()) {
        Path path = output.getPath();
        if (!path.exists()) {
          try {
            writeContentAsLatin1(path, "");
          } catch (IOException e) {
            throw new IllegalStateException("Failed to create output", e);
          }
        }
      }

      return super.execute(actionExecutionContext);
    }
  }

  private static class InjectOutputMetadataAction extends NullAction {
    private final Artifact output;
    private final FileArtifactValue[] metadata;
    private int times;

    public InjectOutputMetadataAction(Artifact output, FileArtifactValue... metadata) {
      super(output);

      this.output = output;
      this.metadata = metadata;
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext) {
      int index = Math.min(times, metadata.length - 1);
      if (index >= 0) {
        actionExecutionContext.getMetadataHandler().injectFile(output, metadata[index]);
      }
      ++times;
      return super.execute(actionExecutionContext);
    }
  }
}
