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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultiset;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutorOwner;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes.NestedFileOpNodesWithSource;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.RemoteFileOpNode;
import com.google.devtools.build.lib.skyframe.serialization.KeyBytesProvider;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.SettableWriteStatus;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantNodeData;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FileDataInfoOrFuture;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureFileDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureNodeDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeInvalidationDataInfo;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;
import org.roaringbitmap.RoaringBitmap;

@RunWith(JUnit4.class)
public final class FileDependencySerializerTest {

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private static final int THREAD_COUNT = 10;

  private static final CompressionService COMPRESSION_SERVICE = new CompressionServiceImpl();

  private final ForkJoinPool forkJoinPool = new ForkJoinPool(THREAD_COUNT);
  private final SafeExecutorOwner executor = new SafeExecutorOwner(forkJoinPool);

  @Mock private LongVersionGetter versionGetter;
  @Mock private InMemoryGraph graph;
  @Mock private KeyValueWriter writer;
  @Mock private InMemoryNodeEntry nodeEntry;

  private FileDependencySerializer serializer;
  private Root root;

  @Before
  public void setUp() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    root = Root.fromPath(fs.getPath("/root"));
    root.asPath().createDirectoryAndParents();
    serializer = newSerializer(LeafIdAssignment.lazy());
  }

  @Test
  public void missingNodeEntry_incrementsErrorCounter() {
    FileKey key = FileKey.create(RootedPath.toRootedPath(root, PathFragment.create("missing.txt")));
    when(graph.getIfPresent(key)).thenReturn(null);

    FileDataInfoOrFuture result = serializer.registerDependency(key);

    assertThat(result).isInstanceOf(FutureFileDataInfo.class);
    ExecutionException e =
        assertThrows(ExecutionException.class, () -> ((FutureFileDataInfo) result).get());
    assertThat(e).hasCauseThat().isInstanceOf(MissingSkyframeEntryException.class);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(1);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
  }

  @Test
  public void rootDirectoryDependency_isConstantAndDecrementsCounter() throws Exception {
    FileKey key = FileKey.create(RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT));

    FileDataInfoOrFuture result = serializer.registerDependency(key);

    assertThat(result).isEqualTo(InvalidationDataInfoOrFuture.ConstantFileData.CONSTANT_FILE);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(0);
  }

  @Test
  public void symlinkResolutionFailure_incrementsErrorCounter() throws Exception {
    PathFragment symlinkPathFragment = PathFragment.create("symlink.txt");
    PathFragment targetPathFragment = PathFragment.create("target.txt");
    RootedPath symlinkRootedPath = RootedPath.toRootedPath(root, symlinkPathFragment);
    FileKey symlinkKey = FileKey.create(symlinkRootedPath);

    FileValue symlinkFsv = mock(FileValue.class);
    when(symlinkFsv.isSymlink()).thenReturn(true);
    when(symlinkFsv.getUnresolvedLinkTarget()).thenReturn(targetPathFragment);
    when(symlinkFsv.realRootedPath(symlinkRootedPath)).thenReturn(symlinkRootedPath);
    when(symlinkFsv.exists()).thenReturn(true);
    when(symlinkFsv.isDirectory()).thenReturn(false);
    when(nodeEntry.getValue()).thenReturn(symlinkFsv);
    when(graph.getIfPresent(symlinkKey)).thenReturn(nodeEntry);

    // Symlink resolution calls getVersion on link path.
    when(versionGetter.getFilePathOrSymlinkVersion(symlinkRootedPath.asPath())).thenReturn(2L);

    // Create the failure mode where the symlink target does not exist in graph.
    RootedPath targetRootedPath = RootedPath.toRootedPath(root, targetPathFragment);
    when(graph.getIfPresent(targetRootedPath)).thenReturn(null);

    FileDataInfoOrFuture result = serializer.registerDependency(symlinkKey);

    assertThat(result).isInstanceOf(FutureFileDataInfo.class);
    ExecutionException e =
        assertThrows(ExecutionException.class, () -> ((FutureFileDataInfo) result).get());
    assertThat(e).hasCauseThat().isInstanceOf(MissingSkyframeEntryException.class);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(1);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
  }

  @Test
  public void registerFileDependency_recordsSamples() throws Exception {
    ProfileCollector profileCollector = new ProfileCollector();
    serializer =
        new FileDependencySerializer(
            versionGetter,
            graph,
            COMPRESSION_SERVICE,
            writer,
            executor,
            profileCollector,
            LeafIdAssignment.lazy());

    PathFragment filePathFragment = PathFragment.create("file.txt");
    RootedPath rootedPath = RootedPath.toRootedPath(root, filePathFragment);
    FileKey key = FileKey.create(rootedPath);

    FileValue fsv = mock(FileValue.class);
    when(fsv.isSymlink()).thenReturn(false);
    when(fsv.realRootedPath(rootedPath)).thenReturn(rootedPath);
    when(fsv.exists()).thenReturn(true);
    when(fsv.isDirectory()).thenReturn(false);
    when(nodeEntry.getValue()).thenReturn(fsv);
    when(graph.getIfPresent(key)).thenReturn(nodeEntry);

    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath.asPath())).thenReturn(2L);

    SettableWriteStatus writeStatus = new WriteStatuses.SettableWriteStatus();
    when(writer.put(any(), any())).thenReturn(writeStatus);

    FileDataInfoOrFuture result = serializer.registerDependency(key);
    ((FutureFileDataInfo) result).get();

    // Not novel yet, no samples.
    assertThat(profileCollector.toProto().getSampleCount()).isEqualTo(0);

    writeStatus.markSuccess(true); // was novel

    // Samples should be recorded now.
    Profile profile = profileCollector.toProto();
    assertThat(profile.getSampleCount()).isGreaterThan(0);
  }

  @Test
  public void registerNestedNodes_completesSuccessfully() throws Exception {
    PathFragment file1Fragment = PathFragment.create("file1.txt");
    PathFragment file2Fragment = PathFragment.create("file2.txt");
    RootedPath rootedPath1 = RootedPath.toRootedPath(root, file1Fragment);
    RootedPath rootedPath2 = RootedPath.toRootedPath(root, file2Fragment);
    FileKey key1 = FileKey.create(rootedPath1);
    FileKey key2 = FileKey.create(rootedPath2);

    FileValue fsv1 = mock(FileValue.class);
    when(fsv1.isSymlink()).thenReturn(false);
    when(fsv1.realRootedPath(rootedPath1)).thenReturn(rootedPath1);
    when(fsv1.exists()).thenReturn(true);
    when(fsv1.isDirectory()).thenReturn(false);
    when(graph.getIfPresent(key1)).thenReturn(nodeEntry);
    when(nodeEntry.getValue()).thenReturn(fsv1);

    InMemoryNodeEntry nodeEntry2 = mock(InMemoryNodeEntry.class);
    FileValue fsv2 = mock(FileValue.class);
    when(fsv2.isSymlink()).thenReturn(false);
    when(fsv2.realRootedPath(rootedPath2)).thenReturn(rootedPath2);
    when(fsv2.exists()).thenReturn(true);
    when(fsv2.isDirectory()).thenReturn(false);
    when(graph.getIfPresent(key2)).thenReturn(nodeEntry2);
    when(nodeEntry2.getValue()).thenReturn(fsv2);

    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath1.asPath())).thenReturn(1L);
    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath2.asPath())).thenReturn(2L);

    SettableWriteStatus writeStatus = new WriteStatuses.SettableWriteStatus();
    when(writer.put(any(), any())).thenReturn(writeStatus);
    when(writer.fingerprint(any())).thenReturn(new PackedFingerprint(1L, 2L));

    AbstractNestedFileOpNodes nested =
        (AbstractNestedFileOpNodes) AbstractNestedFileOpNodes.from(ImmutableList.of(key1, key2));

    var result = serializer.registerDependency(nested);
    assertThat(result).isInstanceOf(FutureNodeDataInfo.class);

    NodeDataInfo info = ((FutureNodeDataInfo) result).get();
    assertThat(info).isInstanceOf(NodeInvalidationDataInfo.class);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWaitingForUpload.get())
        .isEqualTo(3); // 2 files + 1 node

    writeStatus.markSuccess(true);
    assertThat(serializer.getCounters().nodesWaitingForUpload.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesUploaded.get()).isEqualTo(3);
  }

  @Test
  public void registerNestedNodes_childFails_incrementsErrorCounter() throws Exception {
    PathFragment file1Fragment = PathFragment.create("file1.txt");
    PathFragment file2Fragment = PathFragment.create("file2.txt");
    RootedPath rootedPath1 = RootedPath.toRootedPath(root, file1Fragment);
    RootedPath rootedPath2 = RootedPath.toRootedPath(root, file2Fragment);
    FileKey key1 = FileKey.create(rootedPath1);
    FileKey key2 = FileKey.create(rootedPath2);

    FileValue fsv1 = mock(FileValue.class);
    when(fsv1.isSymlink()).thenReturn(false);
    when(fsv1.realRootedPath(rootedPath1)).thenReturn(rootedPath1);
    when(fsv1.exists()).thenReturn(true);
    when(fsv1.isDirectory()).thenReturn(false);
    when(graph.getIfPresent(key1)).thenReturn(nodeEntry);
    when(nodeEntry.getValue()).thenReturn(fsv1);
    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath1.asPath())).thenReturn(1L);

    SettableWriteStatus writeStatus = new WriteStatuses.SettableWriteStatus();
    when(writer.put(any(), any())).thenReturn(writeStatus);

    // key2 is missing from graph
    when(graph.getIfPresent(key2)).thenReturn(null);

    var nested =
        (AbstractNestedFileOpNodes) AbstractNestedFileOpNodes.from(ImmutableList.of(key1, key2));

    var result = serializer.registerDependency(nested);
    assertThat(result).isInstanceOf(FutureNodeDataInfo.class);

    ExecutionException e =
        assertThrows(ExecutionException.class, () -> ((FutureNodeDataInfo) result).get());
    assertThat(e).hasCauseThat().isInstanceOf(MissingSkyframeEntryException.class);
    forkJoinPool.awaitQuiescence(5, SECONDS);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(2);
  }

  @Test
  public void registerNestedNodes_constantChildren_resolvesToConstantNode() throws Exception {
    FileKey rootKey1 = FileKey.create(RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT));
    FileKey rootKey2 = FileKey.create(RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT));

    var nested =
        (AbstractNestedFileOpNodes)
            AbstractNestedFileOpNodes.from(ImmutableList.of(rootKey1, rootKey2));

    var result = serializer.registerDependency(nested);
    assertThat(result).isInstanceOf(FutureNodeDataInfo.class);

    NodeDataInfo info = ((FutureNodeDataInfo) result).get();
    assertThat(info).isEqualTo(ConstantNodeData.CONSTANT_NODE);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(0);
  }

  @Test
  public void registerNestedNodes_unaryChild_unwrapsChildNode() throws Exception {
    PathFragment file1Fragment = PathFragment.create("file1.txt");
    PathFragment file2Fragment = PathFragment.create("file2.txt");
    RootedPath rootedPath1 = RootedPath.toRootedPath(root, file1Fragment);
    RootedPath rootedPath2 = RootedPath.toRootedPath(root, file2Fragment);
    FileKey key1 = FileKey.create(rootedPath1);
    FileKey key2 = FileKey.create(rootedPath2);

    FileValue fsv1 = mock(FileValue.class);
    when(fsv1.isSymlink()).thenReturn(false);
    when(fsv1.realRootedPath(rootedPath1)).thenReturn(rootedPath1);
    when(fsv1.exists()).thenReturn(true);
    when(fsv1.isDirectory()).thenReturn(false);
    when(graph.getIfPresent(key1)).thenReturn(nodeEntry);
    when(nodeEntry.getValue()).thenReturn(fsv1);

    InMemoryNodeEntry nodeEntry2 = mock(InMemoryNodeEntry.class);
    FileValue fsv2 = mock(FileValue.class);
    when(fsv2.isSymlink()).thenReturn(false);
    when(fsv2.realRootedPath(rootedPath2)).thenReturn(rootedPath2);
    when(fsv2.exists()).thenReturn(true);
    when(fsv2.isDirectory()).thenReturn(false);
    when(graph.getIfPresent(key2)).thenReturn(nodeEntry2);
    when(nodeEntry2.getValue()).thenReturn(fsv2);

    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath1.asPath())).thenReturn(1L);
    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath2.asPath())).thenReturn(2L);

    SettableWriteStatus writeStatus = new WriteStatuses.SettableWriteStatus();
    when(writer.put(any(), any())).thenReturn(writeStatus);
    PackedFingerprint childFingerprint = new PackedFingerprint(1L, 2L);
    when(writer.fingerprint(any())).thenReturn(childFingerprint);

    var childNested =
        (AbstractNestedFileOpNodes) AbstractNestedFileOpNodes.from(ImmutableList.of(key1, key2));

    FileKey constantKey =
        FileKey.create(RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT));
    var parentNested =
        (AbstractNestedFileOpNodes)
            AbstractNestedFileOpNodes.from(ImmutableList.of(childNested, constantKey));

    var parentResult = serializer.registerDependency(parentNested);
    NodeDataInfo parentInfo = ((FutureNodeDataInfo) parentResult).get();

    // parentNested should unwrap to childNested's NodeInvalidationDataInfo
    assertThat(parentInfo).isInstanceOf(NodeInvalidationDataInfo.class);
    assertThat(((NodeInvalidationDataInfo) parentInfo).cacheKey()).isEqualTo(childFingerprint);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(0);
  }

  private FileDependencySerializer newSerializer(LeafIdAssignment leafIds) {
    return new FileDependencySerializer(
        versionGetter,
        graph,
        COMPRESSION_SERVICE,
        writer,
        executor,
        /* profileCollector= */ null,
        leafIds);
  }

  private static LeafIdAssignment denseLeafIds(
      Map<? extends FileOpNode, Integer> analysisIds, Map<FileKey, Integer> sourceIds) {
    return LeafIdAssignment.dense(
        new Object2IntOpenHashMap<>(analysisIds), new Object2IntOpenHashMap<>(sourceIds));
  }

  /** Write statuses handed out by {@link #stubWriter}, by key. */
  private final Map<KeyBytesProvider, SettableWriteStatus> writeStatuses =
      new ConcurrentHashMap<>();

  /**
   * Gives every entry a distinct fingerprint, and every put a {@link SettableWriteStatus} recorded
   * in {@link #writeStatuses}.
   */
  private void stubWriter() {
    var nextFingerprint = new AtomicLong(1);
    when(writer.fingerprint(any()))
        .thenAnswer(
            _ -> {
              long id = nextFingerprint.getAndIncrement();
              return new PackedFingerprint(id, id);
            });
    when(writer.put(any(), any()))
        .thenAnswer(
            invocation ->
                writeStatuses.computeIfAbsent(
                    invocation.getArgument(0), _ -> new SettableWriteStatus()));
  }

  private FileKey mockFile(String filename, long version) throws Exception {
    PathFragment fragment = PathFragment.create(filename);
    RootedPath rootedPath = RootedPath.toRootedPath(root, fragment);
    FileKey key = FileKey.create(rootedPath);

    InMemoryNodeEntry entry = mock(InMemoryNodeEntry.class);
    FileValue fsv = mock(FileValue.class);
    when(fsv.isSymlink()).thenReturn(false);
    when(fsv.realRootedPath(rootedPath)).thenReturn(rootedPath);
    when(fsv.exists()).thenReturn(true);
    when(fsv.isDirectory()).thenReturn(false);
    when(graph.getIfPresent(key)).thenReturn(entry);
    when(entry.getValue()).thenReturn(fsv);

    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath.asPath())).thenReturn(version);
    return key;
  }

  private DirectoryListingKey mockDirectoryListing(
      String dirname, long dirVersion, long fileVersion) throws Exception {
    PathFragment fragment = PathFragment.create(dirname);
    RootedPath rootedPath = RootedPath.toRootedPath(root, fragment);
    DirectoryListingKey key = DirectoryListingValue.key(rootedPath);
    FileKey dirFileKey = FileValue.key(rootedPath);

    InMemoryNodeEntry entry = mock(InMemoryNodeEntry.class);
    FileValue fsv = mock(FileValue.class);
    when(fsv.isSymlink()).thenReturn(false);
    when(fsv.realRootedPath(rootedPath)).thenReturn(rootedPath);
    when(fsv.exists()).thenReturn(true);
    when(fsv.isDirectory()).thenReturn(true);
    when(graph.getIfPresent(dirFileKey)).thenReturn(entry);
    when(entry.getValue()).thenReturn(fsv);

    when(versionGetter.getDirectoryListingVersion(rootedPath.asPath())).thenReturn(dirVersion);
    when(versionGetter.getFilePathOrSymlinkVersion(rootedPath.asPath())).thenReturn(fileVersion);
    return key;
  }

  private static RemoteFileOpNode remoteNode(int id) {
    byte[] fingerprint = new byte[16];
    fingerprint[0] = (byte) id;
    return new RemoteFileOpNode(ByteString.copyFrom(fingerprint));
  }

  /**
   * A nested node over {@code deps}.
   *
   * <p>{@link AbstractNestedFileOpNodes#from} returns a lone dependency itself instead of wrapping
   * it, so a node over a single leaf is written by repeating the leaf.
   */
  private static AbstractNestedFileOpNodes nested(FileOpNode... deps) {
    return (AbstractNestedFileOpNodes) AbstractNestedFileOpNodes.from(ImmutableList.copyOf(deps));
  }

  private static NestedFileOpNodesWithSource nestedWithSource(FileKey source, FileOpNode... deps) {
    return (NestedFileOpNodesWithSource)
        AbstractNestedFileOpNodes.from(ImmutableList.copyOf(deps), source);
  }

  private NodeInvalidationDataInfo serialize(AbstractNestedFileOpNodes node) throws Exception {
    return serialize(serializer, node);
  }

  private static NodeInvalidationDataInfo serialize(
      FileDependencySerializer serializer, AbstractNestedFileOpNodes node) throws Exception {
    NodeDataInfo info =
        switch (serializer.registerDependency(node)) {
          case NodeDataInfo immediate -> immediate;
          case FutureNodeDataInfo future -> future.get();
        };
    return (NodeInvalidationDataInfo) info;
  }

  /** The header and child node references of an entry as written to the {@link #writer}. */
  private record PersistedNode(
      ImmutableMultiset<PackedFingerprint> nodeDependencies,
      int fileCount,
      int listingCount,
      boolean hasSource) {}

  private PersistedNode persisted(PackedFingerprint key) throws IOException {
    ArgumentCaptor<byte[]> bytes = ArgumentCaptor.forClass(byte[].class);
    verify(writer).put(eq(key), bytes.capture());
    CodedInputStream in = CodedInputStream.newInstance(bytes.getValue());
    int nodeCount = in.readInt32();
    int fileCount = in.readInt32();
    int listingCount = in.readInt32();
    boolean hasSource = in.readBool();
    var nodeDependencies = ImmutableMultiset.<PackedFingerprint>builder();
    for (int i = 0; i < nodeCount; i++) {
      nodeDependencies.add(PackedFingerprint.readFrom(in));
    }
    return new PersistedNode(nodeDependencies.build(), fileCount, listingCount, hasSource);
  }

  private void assertQuiescent() {
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(0);
  }

  /** The single run {@code [start, end)}, stored as a RunContainer. */
  private static RoaringBitmap runContainer(long start, long end) {
    var run = new RoaringBitmap();
    run.add(start, end);
    run.runOptimize();
    return run;
  }

  @Test
  public void diamondTransitiveReduction_prunesRedundantTransitiveEdges() throws Exception {
    stubWriter();
    FileKey file1 = mockFile("diamond_file1.txt", 1L);
    FileKey file2 = mockFile("diamond_file2.txt", 2L);
    FileKey file3 = mockFile("diamond_file3.txt", 3L);
    FileKey file4 = mockFile("diamond_file4.txt", 4L);
    AbstractNestedFileOpNodes nodeD = nested(file1, file2);
    AbstractNestedFileOpNodes nodeB = nested(nodeD, file3);
    AbstractNestedFileOpNodes nodeC = nested(nodeD, file4);
    var infoD = serialize(nodeD);
    var infoB = serialize(nodeB);
    var infoC = serialize(nodeC);
    assertThat(infoD.cardinality()).isEqualTo(2);
    assertThat(infoB.cardinality()).isEqualTo(3);
    assertThat(infoC.cardinality()).isEqualTo(3);

    // A depends on B, C, D and file1. D is contained in B and C, and file1 in D.
    var infoA = serialize(nested(nodeB, nodeC, nodeD, file1));

    assertThat(infoA.cardinality()).isEqualTo(4);
    assertThat(infoA.hasOpaqueDependencies()).isFalse();
    assertThat(persisted(infoA.cacheKey()))
        .isEqualTo(
            new PersistedNode(
                ImmutableMultiset.of(infoB.cacheKey(), infoC.cacheKey()),
                /* fileCount= */ 0,
                /* listingCount= */ 0,
                /* hasSource= */ false));
    assertQuiescent();
  }

  @Test
  public void crossChannelIsolation_analysisFileNotPrunedBySourceFileOnChild() throws Exception {
    stubWriter();
    FileKey fooFile = mockFile("foo.txt", 10L);
    FileKey childFile = mockFile("child.txt", 20L);
    // The child depends on childFile for analysis and on fooFile as a source.
    AbstractNestedFileOpNodes child = nestedWithSource(fooFile, childFile);
    var childInfo = serialize(child);
    assertThat(childInfo.cardinality()).isEqualTo(2);

    // The parent depends on fooFile for analysis, which the child's source edge must not cover.
    var parentInfo = serialize(nested(child, fooFile));

    // childFile, plus fooFile once per channel.
    assertThat(parentInfo.cardinality()).isEqualTo(3);
    assertThat(persisted(parentInfo.cacheKey()))
        .isEqualTo(
            new PersistedNode(
                ImmutableMultiset.of(childInfo.cacheKey()),
                /* fileCount= */ 1,
                /* listingCount= */ 0,
                /* hasSource= */ false));
    assertQuiescent();
  }

  @Test
  public void opaqueSubgraphRetention_retainsDistinctOpaqueNodesAndDeduplicatesIdentical()
      throws Exception {
    stubWriter();
    FileKey commonFile = mockFile("opaque_common.txt", 1L);
    AbstractNestedFileOpNodes nodeC1 = nested(commonFile, remoteNode(1));
    AbstractNestedFileOpNodes nodeC2 = nested(commonFile, remoteNode(2));
    var infoC1 = serialize(nodeC1);
    var infoC2 = serialize(nodeC2);
    for (var info : ImmutableList.of(infoC1, infoC2)) {
      assertThat(info.hasOpaqueDependencies()).isTrue();
      assertThat(info.cardinality()).isEqualTo(Integer.MAX_VALUE);
    }

    // Neither opaque child can be proven redundant by bitmaps, so both are retained. The repeated
    // C1 is deduplicated by fingerprint.
    var infoP = serialize(nested(nodeC1, nodeC2, nodeC1));

    assertThat(infoP.hasOpaqueDependencies()).isTrue();
    assertThat(infoP.cardinality()).isEqualTo(Integer.MAX_VALUE);
    assertThat(persisted(infoP.cacheKey()))
        .isEqualTo(
            new PersistedNode(
                ImmutableMultiset.of(infoC1.cacheKey(), infoC2.cacheKey()),
                /* fileCount= */ 0,
                /* listingCount= */ 0,
                /* hasSource= */ false));
    assertQuiescent();
  }

  @Test
  public void constantSourceFile_unwrapsDirectlyToChild() throws Exception {
    stubWriter();
    AbstractNestedFileOpNodes child =
        nested(mockFile("const_src_file1.txt", 1L), mockFile("const_src_file2.txt", 2L));
    var childInfo = serialize(child);
    FileKey constantFile =
        FileKey.create(RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT));

    // A constant source produces no invalidation data, leaving the parent a pure wrapper.
    assertThat(serialize(nestedWithSource(constantFile, child))).isSameInstanceAs(childInfo);
    assertQuiescent();
  }

  @Test
  public void subsumedSourceFile_prunedAndUnwrapsDirectlyToChild() throws Exception {
    stubWriter();
    FileKey sharedSource = mockFile("subsumed_src_source.txt", 2L);
    AbstractNestedFileOpNodes child =
        nestedWithSource(sharedSource, mockFile("subsumed_src_analysis.txt", 1L));
    var childInfo = serialize(child);
    assertThat(childInfo.cardinality()).isEqualTo(2);

    // The child already covers the parent's source, so the parent is left wrapping only the child.
    assertThat(serialize(nestedWithSource(sharedSource, child))).isSameInstanceAs(childInfo);
    assertQuiescent();
  }

  @Test
  public void concurrentParents_doNotMutateSharedChildBitmap() throws Exception {
    stubWriter();
    int parentCount = 25;
    AbstractNestedFileOpNodes sharedChild =
        nested(mockFile("shared_base1.txt", 1L), mockFile("shared_base2.txt", 2L));
    var childInfo = serialize(sharedChild);
    // Stubs are set up before any parent starts: Mockito stubbing is not safe against concurrent
    // invocations of the same mock.
    var parents = new ArrayList<AbstractNestedFileOpNodes>();
    for (int i = 0; i < parentCount; i++) {
      parents.add(nested(sharedChild, mockFile("parent_file_" + i + ".txt", 100L + i)));
    }

    var parentFutures = new ArrayList<FutureNodeDataInfo>();
    for (AbstractNestedFileOpNodes parent : parents) {
      parentFutures.add((FutureNodeDataInfo) serializer.registerDependency(parent));
    }
    for (FutureNodeDataInfo future : parentFutures) {
      // The shared child's two files plus the parent's own.
      assertThat(future.get().cardinality()).isEqualTo(3);
    }

    var childClosure = new RoaringBitmap();
    assertThat(childInfo.mergeInto(childClosure)).isTrue();
    assertThat(childClosure.getLongCardinality()).isEqualTo(2);
    assertQuiescent();
  }

  /**
   * Regression test for RoaringBitmap's {@code BitmapContainer.contains(RunContainer)}, which tests
   * each run as {@code [start, start + length)} and so ignores a run's last value. Reduction once
   * used {@code union.contains(child)} and dropped such a child, losing its run-end bit.
   */
  @Test
  public void mergeInto_childMissingOnlyRunEndFromBitmapUnion_mergesChild() {
    var union = bitmapContainerUnion();
    long cardinalityBefore = union.getLongCardinality();

    // Only the run's last value, 101, is outside the union.
    assertThat(transparentInfo(runContainer(50L, 102L)).mergeInto(union)).isTrue();

    assertThat(union.contains(101)).isTrue();
    assertThat(union.getLongCardinality()).isEqualTo(cardinalityBefore + 1);
  }

  @Test
  public void mergeInto_subsetChild_returnsFalseAndLeavesUnionUnchanged() {
    var union = bitmapContainerUnion();
    RoaringBitmap unionBefore = union.clone();

    assertThat(transparentInfo(runContainer(50L, 101L)).mergeInto(union)).isFalse();

    assertThat(union).isEqualTo(unionBefore);
  }

  private static NodeInvalidationDataInfo transparentInfo(RoaringBitmap bitmap) {
    return new NodeInvalidationDataInfo(
        new PackedFingerprint(1L, 1L),
        new SettableWriteStatus(),
        bitmap,
        /* hasOpaqueDependencies= */ false);
  }

  /**
   * {@code [0, 100]} plus every even number up to 10000, run-optimized.
   *
   * <p>That is more than 4096 values in one chunk, so Roaring stores it as a BitmapContainer. 101
   * is absent.
   */
  private static RoaringBitmap bitmapContainerUnion() {
    var union = new RoaringBitmap();
    union.add(0L, 101L);
    for (int i = 0; i <= 10000; i += 2) {
      union.add(i);
    }
    union.runOptimize();
    return union;
  }

  @Test
  public void denseLeafIdAssignment_succeedsWhenAllLeavesMapped() throws Exception {
    stubWriter();
    FileKey file1 = mockFile("dense_file1.txt", 1L);
    DirectoryListingKey listing = mockDirectoryListing("dense_dir", 2L, 3L);
    FileKey sourceFile = mockFile("dense_source.txt", 4L);
    var leafIds =
        denseLeafIds(ImmutableMap.of(file1, 0, listing, 1), ImmutableMap.of(sourceFile, 2));

    var info = serialize(newSerializer(leafIds), nestedWithSource(sourceFile, file1, listing));

    assertThat(info.cardinality()).isEqualTo(3);
    assertThat(leafIds.fallbackIdCount()).isEqualTo(0);
  }

  /** The individual miss cases are covered by {@link LeafIdAssignmentTest}. */
  @Test
  public void denseLeafIdAssignment_unmappedLeaf_failsNodeSerialization() throws Exception {
    stubWriter();
    FileKey unmapped = mockFile("unmapped_file1.txt", 1L);
    FileKey mapped = mockFile("mapped_file2.txt", 2L);
    var denseSerializer =
        newSerializer(denseLeafIds(ImmutableMap.of(mapped, 0), ImmutableMap.of()));

    ExecutionException e =
        assertThrows(
            ExecutionException.class, () -> serialize(denseSerializer, nested(unmapped, mapped)));

    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e).hasCauseThat().hasMessageThat().contains("Missing dense leaf ID for");
  }

  @Test
  public void denseLeafIdAssignment_fileWithAncestorDirectories_succeedsWithoutAncestorIds()
      throws Exception {
    stubWriter();
    // The parent directory "sub" is in the graph, but is not a leaf and so has no ID.
    FileKey subDirKey = FileKey.create(RootedPath.toRootedPath(root, PathFragment.create("sub")));
    InMemoryNodeEntry subEntry = mock(InMemoryNodeEntry.class);
    FileValue subFsv = mock(FileValue.class);
    when(subFsv.isSymlink()).thenReturn(false);
    when(subFsv.realRootedPath(any())).thenAnswer(inv -> inv.getArgument(0));
    when(subFsv.exists()).thenReturn(true);
    when(subFsv.isDirectory()).thenReturn(true);
    when(graph.getIfPresent(subDirKey)).thenReturn(subEntry);
    when(subEntry.getValue()).thenReturn(subFsv);
    FileKey file = mockFile("sub/file.txt", 1L);
    var leafIds = denseLeafIds(ImmutableMap.of(file, 0), ImmutableMap.of());

    var info = serialize(newSerializer(leafIds), nested(file, file));

    assertThat(info.cardinality()).isEqualTo(1);
    assertThat(leafIds.fallbackIdCount()).isEqualTo(0);
  }

  @Test
  public void writeStatus_prunedNodeDependency_writeStatusNotTracked() throws Exception {
    stubWriter();
    FileKey file1 = mockFile("ws_file1.txt", 1L);
    AbstractNestedFileOpNodes nodeB = nested(file1, mockFile("ws_file2.txt", 2L));
    // C's only file is also in B, so A prunes C.
    AbstractNestedFileOpNodes nodeC = nested(file1, file1);
    var infoC = serialize(nodeC);
    var infoA = serialize(nested(nodeB, nodeC, mockFile("ws_file3.txt", 3L)));

    // Complete every write except C's own.
    writeStatuses.forEach(
        (key, status) -> {
          if (!key.equals(infoC.cacheKey())) {
            status.markSuccess(true);
          }
        });

    assertThat(infoC.writeStatus().isDone()).isFalse();
    assertThat(infoA.writeStatus().isDone()).isTrue();
  }
}
