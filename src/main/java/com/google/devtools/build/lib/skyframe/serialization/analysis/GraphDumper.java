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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueCache;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.Fingerprinter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs.DebugContext;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyframeDependencyException;
import com.google.devtools.build.lib.skyframe.serialization.SkyframeLookupContinuation;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencies.AvailableFileDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencies.MissingFileDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyDeserializer.FutureNestedDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ListingDependencies.AvailableListingDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ListingDependencies.MissingListingDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.NestedDependencies.AvailableNestedDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.NestedDependencies.MissingNestedDependencies;
import com.google.devtools.build.lib.skyframe.serialization.proto.DataType;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;

/** Debugging tool to dump the key-value store entries for a given SkyValue. */
public final class GraphDumper {
  /** Slightly parsed data from a SkyValue entry in the key-value store. */
  public record SkyValueEntry(
      // What kind of entry this is
      DataType dataType,
      // If available, the path of the file/listing this node represents
      @Nullable String pathKey,
      // If available, the fingerprint of the invalidation data of this node
      @Nullable PackedFingerprint dependencyFingerprint,
      // The rest of the entry after headers and invalidation data
      ByteString rawValueBytes) {}

  /** An edge in a KV store graph (either for the SkyValue itself or its invalidation data) */
  public record Edge(PackedFingerprint parent, PackedFingerprint child) {}

  /** Whether the invalidation node was present, not present, or could not be fetched. */
  public enum NodeType {
    PRESENT,
    MISSING,
    ERROR
  }

  /** A node in the invalidation graph. */
  public record Node(
      NodeType nodeType,
      @Nullable ImmutableList<String> sources,
      @Nullable ImmutableList<String> fileDependencies,
      @Nullable ImmutableList<String> listingDependencies,
      @Nullable ImmutableList<PackedFingerprint> nestedNodeDependencies,
      @Nullable Exception exception) {

    public static final Node MISSING = new Node(NodeType.MISSING, null, null, null, null, null);

    public Node {
      if (nodeType == NodeType.MISSING && MISSING != null) {
        throw new IllegalArgumentException(
            "Use Node.missing() instead of constructing MISSING nodes.");
      }
    }

    public static Node present(
        List<String> sources,
        List<String> fileDependencies,
        List<String> listingDependencies,
        List<PackedFingerprint> nestedNodeDependencies) {
      return new Node(
          NodeType.PRESENT,
          ImmutableList.copyOf(sources),
          ImmutableList.copyOf(fileDependencies),
          ImmutableList.copyOf(listingDependencies),
          ImmutableList.copyOf(nestedNodeDependencies),
          null);
    }

    public static Node missing() {
      return MISSING;
    }

    public static Node error(Exception exception) {
      return new Node(NodeType.ERROR, null, null, null, null, exception);
    }
  }

  /** A graph of invalidation nodes. */
  public record InvalidationGraph(
      ImmutableList<Edge> edges, ImmutableMap<PackedFingerprint, Node> metadata) {}

  private final FingerprintValueStore store;
  private final FileDependencyDeserializer deserializer;
  private final Set<PackedFingerprint> visited;
  private final List<Edge> edges;
  private final Map<PackedFingerprint, Node> metadata;

  private GraphDumper(FingerprintValueStore store, FileDependencyDeserializer deserializer) {
    this.store = store;
    this.deserializer = deserializer;
    this.visited = new HashSet<>();
    this.edges = new ArrayList<>();
    this.metadata = new LinkedHashMap<>();
  }

  public static ImmutableList<Edge> collectEdgesForSkyValue(
      InMemoryGraph graph,
      ObjectCodecs codecs,
      FingerprintValueStore fingerprintValueStore,
      SkyValueEntry skyValue,
      PackedFingerprint targetFingerprint)
      throws SerializationException, ExecutionException, InterruptedException {
    ConcurrentLinkedQueue<Edge> edgeList = new ConcurrentLinkedQueue<>();
    BiConsumer<PackedFingerprint, PackedFingerprint> edgeReceiver =
        (parent, child) -> edgeList.add(new Edge(parent, child));

    var fingerprintValueService =
        new FingerprintValueService(
            directExecutor(),
            fingerprintValueStore,
            new FingerprintValueCache(FingerprintValueCache.SyncMode.NOT_LINKED),
            FingerprintValueService.NONPROD_FINGERPRINTER);

    Object deserialized =
        codecs.deserializeWithSkyframe(
            fingerprintValueService,
            skyValue.rawValueBytes().newCodedInput(),
            new DebugContext(targetFingerprint, edgeReceiver));

    if (deserialized instanceof ListenableFuture<?> futureContinuation) {
      SkyframeLookupContinuation continuation =
          (SkyframeLookupContinuation) futureContinuation.get();

      // Drive the continuation synchronously hoping that whatever Skyframe nodes it requests
      // are already there.
      driveContinuation(graph, continuation);
    }
    return ImmutableList.copyOf(edgeList);
  }

  private static void driveContinuation(
      InMemoryGraph graph, SkyframeLookupContinuation continuation)
      throws SerializationException, InterruptedException {
    LookupEnvironment env =
        new EnvironmentForUtilities(
            key -> {
              var entry = graph.getIfPresent(key);
              if (entry == null) {
                return null;
              }
              if (entry.getErrorInfo() != null && entry.getErrorInfo().getException() != null) {
                return entry.getErrorInfo().getException();
              }
              return entry.getValue();
            });
    try {
      ListenableFuture<?> resultFuture = continuation.process(env);
      if (resultFuture == null) {
        throw new SerializationException(
            "Read-only Skyframe lookup failed to resolve dependencies synchronously.");
      }
      resultFuture.get();
    } catch (SkyframeDependencyException e) {
      throw new SerializationException("Cannot find node in Skyframe: " + e.getMessage(), e);
    } catch (ExecutionException e) {
      throw new SerializationException(e);
    }
  }

  public static InvalidationGraph collectInvalidationGraph(
      PackedFingerprint rootFingerprint, FingerprintValueStore store, Fingerprinter fingerprinter)
      throws InterruptedException {

    FileDependencyDeserializer deserializer =
        new FileDependencyDeserializer(directExecutor(), fingerprinter);

    GraphDumper dumper = new GraphDumper(store, deserializer);
    dumper.collectRecursive(rootFingerprint);

    return new InvalidationGraph(
        ImmutableList.copyOf(dumper.edges), ImmutableMap.copyOf(dumper.metadata));
  }

  private void collectRecursive(PackedFingerprint fingerprint) throws InterruptedException {
    if (!visited.add(fingerprint)) {
      return;
    }

    FileDependencyDeserializer.NestedDependenciesOrFuture valueOrFuture =
        deserializer.getNestedDependencies(fingerprint, store);
    NestedDependencies nestedDeps;

    // We just do a single-thread DFS walk. Not particularly efficient, but this is for debugging so
    // that's OK.
    try {
      nestedDeps =
          switch (valueOrFuture) {
            case NestedDependencies nd -> nd;
            case FutureNestedDependencies fnd -> fnd.get();
          };
    } catch (ExecutionException e) {
      metadata.put(fingerprint, Node.error(e));
      return;
    }

    if (nestedDeps.isMissingData()) {
      metadata.put(fingerprint, Node.missing());
      return;
    }

    AvailableNestedDependencies available = (AvailableNestedDependencies) nestedDeps;
    List<String> sources = new ArrayList<>();
    List<String> fileDependencies = new ArrayList<>();
    List<String> listingDependencies = new ArrayList<>();
    List<PackedFingerprint> nestedNodeDependencies = new ArrayList<>();

    // Read direct file dependencies (sources)
    for (int i = 0; i < available.sourcesCount(); i++) {
      AvailableFileDependencies source = available.getSource(i);
      sources.add(source.resolvedPath());
    }

    // Read direct nested child dependencies (analysis dependencies)
    for (int i = 0; i < available.analysisDependenciesCount(); i++) {
      FileSystemDependencies child = available.getAnalysisDependency(i);
      switch (child) {
        case MissingFileDependencies m -> {}
        case MissingListingDependencies m -> {}
        case MissingNestedDependencies m -> {}
        case AvailableFileDependencies file -> fileDependencies.add(file.resolvedPath());
        case AvailableListingDependencies listing ->
            listingDependencies.add(listing.realDirectory().resolvedPath());
        case AvailableNestedDependencies availableChild -> {
          PackedFingerprint childFp = availableChild.fingerprint();
          if (childFp != null) {
            nestedNodeDependencies.add(childFp);
            edges.add(new Edge(fingerprint, childFp));
            collectRecursive(childFp);
          }
        }
      }
    }

    Node meta =
        Node.present(sources, fileDependencies, listingDependencies, nestedNodeDependencies);
    metadata.put(fingerprint, meta);
  }

  public static SkyValueEntry parseSkyValueEntry(ByteString entryBytes) throws IOException {
    CodedInputStream codedIn = entryBytes.newCodedInput();
    int dataTypeOrdinal = codedIn.readInt32();
    DataType dataType = DataType.forNumber(dataTypeOrdinal);
    if (dataType == null) {
      return new SkyValueEntry(
          DataType.UNRECOGNIZED, null, null, entryBytes.substring(codedIn.getTotalBytesRead()));
    }

    String pathKey = null;
    PackedFingerprint dependencyFingerprint = null;

    switch (dataType) {
      case DATA_TYPE_EMPTY, DATA_TYPE_UNSPECIFIED, UNRECOGNIZED -> {}
      case DATA_TYPE_FILE, DATA_TYPE_LISTING -> pathKey = codedIn.readString();
      case DATA_TYPE_ANALYSIS_NODE, DATA_TYPE_EXECUTION_NODE ->
          dependencyFingerprint = PackedFingerprint.readFrom(codedIn);
    }

    ByteString rawValueBytes = entryBytes.substring(codedIn.getTotalBytesRead());
    return new SkyValueEntry(dataType, pathKey, dependencyFingerprint, rawValueBytes);
  }
}
