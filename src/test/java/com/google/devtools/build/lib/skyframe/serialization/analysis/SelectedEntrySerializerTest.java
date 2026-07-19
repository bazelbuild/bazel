// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.KeyBytesProvider;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatus;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.SettableWriteStatus;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.Version;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

@RunWith(JUnit4.class)
public final class SelectedEntrySerializerTest {

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  @Mock private LongVersionGetter versionGetter;
  @Mock private EventBus eventBus;

  private final InMemoryGraph graph = InMemoryGraph.create();
  private Root root;
  private ControlledStore valueStore;
  private ControlledStore invalidationStore;
  private FingerprintValueService fingerprintValueService;
  private ObjectCodecs codecs;

  @Before
  public void setUp() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    root = Root.fromPath(fs.getPath("/root"));
    root.asPath().createDirectoryAndParents();

    valueStore = new ControlledStore();
    invalidationStore = new ControlledStore();
    fingerprintValueService = FingerprintValueService.createForTesting(valueStore);
    codecs = new ObjectCodecs(AutoRegistry.get().getBuilder().add(new TestSkyValueCodec()).build());
    when(versionGetter.getFilePathOrSymlinkVersion(any())).thenReturn(1L);
  }

  private FileKey createFileKey(String path) throws Exception {
    Path filePath = root.getRelative(path);
    filePath.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(filePath, "test");
    RootedPath rootedPath = RootedPath.toRootedPath(root, PathFragment.create(path));
    FileKey fileKey = FileKey.create(rootedPath);
    FileValue fileValue =
        FileStateValue.create(rootedPath, SyscallCache.NO_CACHE, /* tsgm= */ null);
    addDoneNode(fileKey, fileValue, ImmutableList.of());
    return fileKey;
  }

  private void addDoneNode(SkyKey key, SkyValue value, List<? extends SkyKey> directDeps)
      throws InterruptedException {
    InMemoryNodeEntry entry =
        (InMemoryNodeEntry)
            graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key)).get(key);
    entry.addReverseDepAndCheckIfDone(null);
    entry.markRebuilding();
    for (SkyKey dep : directDeps) {
      entry.addSingletonTemporaryDirectDep(dep);
      entry.signalDep(Version.constant(), dep);
    }
    entry.setValue(value, Version.constant(), /* maxTransitiveSourceVersion= */ null);
  }

  private ConfiguredTargetKey createConfiguredTarget(
      String label, @Nullable String sharedData, List<FileKey> fileDeps)
      throws InterruptedException {
    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked(label)).build();
    addDoneNode(targetKey, new TestSkyValue(sharedData), fileDeps);
    return targetKey;
  }

  private static final class ControlledStore implements FingerprintValueStore, KeyValueWriter {
    private final List<SettableWriteStatus> putStatuses = new ArrayList<>();
    private final Semaphore putsSemaphore = new Semaphore(0);

    synchronized int getPutCount() {
      return putStatuses.size();
    }

    synchronized SettableWriteStatus getPutStatus(int index) {
      return putStatuses.get(index);
    }

    @Override
    public synchronized WriteStatus put(KeyBytesProvider fingerprint, byte[] serializedBytes) {
      var status = new SettableWriteStatus();
      putStatuses.add(status);
      putsSemaphore.release();
      return status;
    }

    @Override
    public PackedFingerprint fingerprint(byte[] input) {
      return FingerprintValueService.NONPROD_FINGERPRINTER.fingerprint(input);
    }

    @Override
    public ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) {
      throw new UnsupportedOperationException();
    }

    void waitForPuts(int count) throws InterruptedException {
      if (!putsSemaphore.tryAcquire(count, 10, SECONDS)) {
        throw new AssertionError("Timed out waiting for " + count + " puts; got " + getPutCount());
      }
    }
  }

  /** A fake "configured target value" whose only virtue is that it serializes a shared value. */
  private static final class TestSkyValue implements SkyValue {
    @Nullable private final String sharedData;

    TestSkyValue(@Nullable String sharedData) {
      this.sharedData = sharedData;
    }
  }

  /** A codec for shared strings. */
  private static final class SharedStringCodec extends DeferredObjectCodec<String> {
    private static final SharedStringCodec INSTANCE = new SharedStringCodec();

    @Override
    public Class<String> getEncodedClass() {
      return String.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(SerializationContext context, String obj, CodedOutputStream codedOut)
        throws IOException {
      codedOut.writeStringNoTag(obj);
    }

    @Override
    public DeferredObjectCodec.DeferredValue<String> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn) {
      throw new UnsupportedOperationException();
    }
  }

  /** A codec that uses a trivial shared value (a string). */
  private static final class TestSkyValueCodec extends DeferredObjectCodec<TestSkyValue> {
    @Override
    public Class<TestSkyValue> getEncodedClass() {
      return TestSkyValue.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(
        SerializationContext context, TestSkyValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      if (obj.sharedData != null) {
        context.putSharedValue(
            obj.sharedData, /* distinguisher= */ null, SharedStringCodec.INSTANCE, codedOut);
      } else {
        codedOut.writeBoolNoTag(true);
      }
    }

    @Override
    public DeferredObjectCodec.DeferredValue<TestSkyValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn) {
      throw new UnsupportedOperationException();
    }
  }

  private QuiescingFuture<ImmutableList<Throwable>> uploadSelection(List<SkyKey> keysToSerialize)
      throws Exception {
    var fileOpNodeMemoizingLookup =
        new FileOpNodeMemoizingLookup(
            new ForkJoinPool(4),
            graph,
            ImmutableSet.of(),
            /* shouldDiscardMemory= */ false,
            /* referencedPackages= */ null);
    return SelectedEntrySerializer.uploadSelection(
        graph,
        versionGetter,
        codecs,
        FrontierNodeVersion.CONSTANT_FOR_TESTING,
        ImmutableSet.copyOf(keysToSerialize),
        fingerprintValueService,
        invalidationStore,
        /* shouldDiscardMemory= */ false,
        eventBus,
        /* profileCollector= */ null,
        new SelectedEntrySerializer.SerializationStats(),
        /* emitUploadedEvents= */ false,
        fileOpNodeMemoizingLookup);
  }

  @Test
  public void topLevelUpload_waitsForSharedValueUpload() throws Exception {
    FileKey fileKey1 = createFileKey("foo1.txt");
    FileKey fileKey2 = createFileKey("foo2.txt");

    ConfiguredTargetKey targetKey =
        createConfiguredTarget(
            "//test:target", "shared-data", ImmutableList.of(fileKey1, fileKey2));

    var uploadFuture = uploadSelection(ImmutableList.of(targetKey));

    // Finish every upload to the invalidation store
    invalidationStore.waitForPuts(3);
    invalidationStore.getPutStatus(0).markSuccess();
    invalidationStore.getPutStatus(1).markSuccess();
    invalidationStore.getPutStatus(2).markSuccess();

    // Wait for the upload of the shared value to start and verify that the future is not completed
    valueStore.waitForPuts(1);
    assertThat(valueStore.getPutCount()).isEqualTo(1);
    assertThat(uploadFuture.isDone()).isFalse();

    // Mark the shared value as done
    valueStore.getPutStatus(0).markSuccess();

    // Wait for the upload of the top-level entry to start and check that the upload is not marked
    // as completed
    valueStore.waitForPuts(1);
    assertThat(valueStore.getPutCount()).isEqualTo(2);
    assertThat(uploadFuture.isDone()).isFalse();

    // Mark the top-level entry as successful and check that after that it's, done.
    valueStore.getPutStatus(1).markSuccess();
    assertThat(uploadFuture.get()).isEmpty();
  }

  @Test
  public void topLevelUpload_waitsForInvalidationDataUpload() throws Exception {
    FileKey fileKey1 = createFileKey("foo1.txt");
    FileKey fileKey2 = createFileKey("foo2.txt");

    ConfiguredTargetKey targetKey =
        createConfiguredTarget("//test:target", null, ImmutableList.of(fileKey1, fileKey2));

    var uploadFuture = uploadSelection(ImmutableList.of(targetKey));

    invalidationStore.waitForPuts(3);
    assertThat(invalidationStore.getPutCount()).isEqualTo(3);

    // Check that the upload of the value doesn't start before the upload of the invalidation data
    // completes
    assertThat(valueStore.getPutCount()).isEqualTo(0);
    assertThat(uploadFuture.isDone()).isFalse();

    // Complete the invalidation data and verify that the upload starts after that
    invalidationStore.getPutStatus(0).markSuccess();
    invalidationStore.getPutStatus(1).markSuccess();
    invalidationStore.getPutStatus(2).markSuccess();
    valueStore.waitForPuts(1);
    assertThat(valueStore.getPutCount()).isEqualTo(1);
    assertThat(uploadFuture.isDone()).isFalse();

    // Complete the upload and verify that then the future is marked as successful
    valueStore.getPutStatus(0).markSuccess();
    assertThat(uploadFuture.get()).isEmpty();
  }
}
