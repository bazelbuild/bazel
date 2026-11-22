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
package com.google.devtools.build.lib.bazel.sbom;

import com.google.common.collect.ImmutableBiMap;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * Writes a protobom SBOM of the configured targets reachable from the requested targets.
 *
 * <p>The document is deterministic: nodes are sorted by id and edges by {@code (from, to)}, and the
 * timestamp honors {@code SOURCE_DATE_EPOCH}.
 *
 * <p>Fields Bazel does not model are omitted rather than fabricated: supplier, license, and hash.
 * Version is only available for targets that belong to a bzlmod module.
 */
public final class SbomWriter {
  private SbomWriter() {}

  // protobom field numbers, from protobom's api/sbom.proto.
  private static final int DOCUMENT_METADATA = 1;
  private static final int DOCUMENT_NODE_LIST = 2;
  private static final int METADATA_NAME = 3;
  private static final int METADATA_DATE = 4;
  private static final int METADATA_TOOLS = 5;
  private static final int METADATA_AUTHORS = 6;
  private static final int TIMESTAMP_SECONDS = 1;
  private static final int TOOL_NAME = 1;
  private static final int TOOL_VERSION = 2;
  private static final int PERSON_NAME = 1;
  private static final int PERSON_IS_ORG = 2;
  private static final int NODE_LIST_NODES = 1;
  private static final int NODE_LIST_EDGES = 2;
  private static final int NODE_LIST_ROOT_ELEMENTS = 3;
  private static final int NODE_ID = 1;
  private static final int NODE_NAME = 3;
  private static final int NODE_VERSION = 4;
  private static final int NODE_IDENTIFIERS = 28;
  private static final int EDGE_TYPE = 1;
  private static final int EDGE_FROM = 2;
  private static final int EDGE_TO = 3;
  private static final int EDGE_TYPE_DEPENDS_ON = 10;
  private static final int IDENTIFIER_TYPE_PURL = 1;

  /** Writes the SBOM for {@code result} to {@code output}, relative to the output base. */
  public static void write(CommandEnvironment env, BuildResult result, PathFragment output)
      throws IOException, InterruptedException {
    Path path = env.getOutputBase().getRelative(output);
    WalkableGraph graph = env.getSkyframeExecutor().getWalkableGraph();

    ImmutableBiMap<RepositoryName, ModuleKey> repoToModule = ImmutableBiMap.of();
    SkyValue depGraphValue = graph.getValue(BazelDepGraphValue.KEY);
    if (depGraphValue instanceof BazelDepGraphValue depGraph) {
      repoToModule = depGraph.getCanonicalRepoNameLookup();
    }

    Deque<ConfiguredTargetKey> queue = new ArrayDeque<>();
    Set<ConfiguredTargetKey> seen = new LinkedHashSet<>();
    List<String> roots = new ArrayList<>();
    for (ConfiguredTarget target : result.getActualTargets()) {
      roots.add(target.getLabel().getCanonicalForm());
      ConfiguredTargetKey key = ConfiguredTargetKey.fromConfiguredTarget(target);
      if (seen.add(key)) {
        queue.add(key);
      }
    }

    Map<String, Node> nodes = new TreeMap<>();
    Set<Edge> edges = new TreeSet<>();
    while (!queue.isEmpty()) {
      ConfiguredTargetKey key = queue.poll();
      Label label = key.getLabel();
      String id = label.getCanonicalForm();
      nodes.putIfAbsent(id, node(label, repoToModule));
      for (SkyKey dep : graph.getDirectDeps(key)) {
        // A target can appear as its own dep under another configuration; an SBOM node is keyed
        // by label, so that edge would point at itself.
        if (dep instanceof ConfiguredTargetKey depKey
            && !depKey.getLabel().equals(label)) {
          edges.add(new Edge(id, depKey.getLabel().getCanonicalForm()));
          if (seen.add(depKey)) {
            queue.add(depKey);
          }
        }
      }
    }

    byte[] document =
        msg(
            out -> {
              out.writeByteArray(DOCUMENT_METADATA, metadata(env.getClientEnv()));
              out.writeByteArray(DOCUMENT_NODE_LIST, nodeList(nodes, edges, roots));
            });

    path.getParentDirectory().createDirectoryAndParents();
    try (OutputStream stream = path.getOutputStream()) {
      stream.write(document);
    }
    env.getReporter().handle(Event.info("Wrote SBOM to " + path));
  }

  private static Node node(Label label, ImmutableBiMap<RepositoryName, ModuleKey> repoToModule) {
    ModuleKey module = repoToModule.get(label.getRepository());
    String version = module == null ? null : module.version().toString();
    if (version != null && version.isEmpty()) {
      version = null;
    }
    return new Node(label.getCanonicalForm(), label.getName(), version);
  }

  private static byte[] metadata(Map<String, String> clientEnv) throws IOException {
    long seconds = timestamp(clientEnv);
    String version = BlazeVersionInfo.instance().getVersion();
    return msg(
        out -> {
          out.writeString(METADATA_NAME, "bazel");
          out.writeByteArray(METADATA_DATE, msg(t -> t.writeInt64(TIMESTAMP_SECONDS, seconds)));
          out.writeByteArray(
              METADATA_TOOLS,
              msg(
                  t -> {
                    t.writeString(TOOL_NAME, "bazel");
                    t.writeString(TOOL_VERSION, version);
                  }));
          out.writeByteArray(
              METADATA_AUTHORS,
              msg(
                  p -> {
                    p.writeString(PERSON_NAME, "Bazel");
                    p.writeBool(PERSON_IS_ORG, true);
                  }));
        });
  }

  private static byte[] nodeList(Map<String, Node> nodes, Set<Edge> edges, List<String> roots)
      throws IOException {
    return msg(
        out -> {
          for (Node node : nodes.values()) {
            out.writeByteArray(NODE_LIST_NODES, nodeBytes(node));
          }
          for (Edge edge : edges) {
            out.writeByteArray(NODE_LIST_EDGES, edgeBytes(edge));
          }
          for (String root : roots) {
            out.writeString(NODE_LIST_ROOT_ELEMENTS, root);
          }
        });
  }

  private static byte[] nodeBytes(Node node) throws IOException {
    return msg(
        out -> {
          out.writeString(NODE_ID, node.id());
          out.writeString(NODE_NAME, node.name());
          if (node.version() != null) {
            out.writeString(NODE_VERSION, node.version());
            out.writeByteArray(
                NODE_IDENTIFIERS,
                msg(
                    m -> {
                      m.writeInt32(1, IDENTIFIER_TYPE_PURL);
                      m.writeString(2, "pkg:generic/" + node.name() + "@" + node.version());
                    }));
          }
        });
  }

  private static byte[] edgeBytes(Edge edge) throws IOException {
    return msg(
        out -> {
          out.writeEnum(EDGE_TYPE, EDGE_TYPE_DEPENDS_ON);
          out.writeString(EDGE_FROM, edge.from());
          out.writeString(EDGE_TO, edge.to());
        });
  }

  private static long timestamp(Map<String, String> clientEnv) {
    String epoch = clientEnv.get("SOURCE_DATE_EPOCH");
    if (epoch != null) {
      try {
        return Long.parseLong(epoch.trim());
      } catch (NumberFormatException e) {
        // Fall through to the wall clock.
      }
    }
    return System.currentTimeMillis() / 1000;
  }

  @FunctionalInterface
  private interface ProtoWriter {
    void write(CodedOutputStream out) throws IOException;
  }

  private static byte[] msg(ProtoWriter writer) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream out = CodedOutputStream.newInstance(bytes);
    writer.write(out);
    out.flush();
    return bytes.toByteArray();
  }

  private record Node(String id, String name, @Nullable String version) {}

  private record Edge(String from, String to) implements Comparable<Edge> {
    @Override
    public int compareTo(Edge other) {
      int byFrom = from.compareTo(other.from);
      return byFrom != 0 ? byFrom : to.compareTo(other.to);
    }
  }
}
