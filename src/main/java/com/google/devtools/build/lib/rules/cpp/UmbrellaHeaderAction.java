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
package com.google.devtools.build.lib.rules.cpp;


import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Action for generating an umbrella header. All the headers are #included in the umbrella header.
 */
@Immutable
public final class UmbrellaHeaderAction extends AbstractFileWriteAction {

  private static final String GUID = "62ea2952-bf28-92c3-efb1-e34621646910";

  // NOTE: If you add a field here, you'll likely need to add it to the cache key in computeKey().
  private final Artifact umbrellaHeader;
  private final ImmutableList<Artifact> publicHeaders;
  private final ImmutableList<PathFragment> additionalExportedHeaders;

  public UmbrellaHeaderAction(
      ActionOwner owner,
      Artifact umbrellaHeader,
      NestedSet<Artifact> publicHeaders,
      Iterable<PathFragment> additionalExportedHeaders) {
    this(owner, umbrellaHeader, publicHeaders.toList(), additionalExportedHeaders);
  }

  public UmbrellaHeaderAction(
      ActionOwner owner,
      Artifact umbrellaHeader,
      Iterable<Artifact> publicHeaders,
      Iterable<PathFragment> additionalExportedHeaders) {
    super(
        owner,
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(Iterables.filter(publicHeaders, Artifact::isTreeArtifact))
            .build(),
        umbrellaHeader,
        /*makeExecutable=*/ false);
    this.umbrellaHeader = umbrellaHeader;
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.additionalExportedHeaders = ImmutableList.copyOf(additionalExportedHeaders);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)  {
    final ArtifactExpander artifactExpander = ctx.getArtifactExpander();
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        StringBuilder content = new StringBuilder();
        HashSet<PathFragment> deduper = new HashSet<>();
        for (Artifact artifact : expandedHeaders(artifactExpander, publicHeaders)) {
          appendHeader(content, artifact.getExecPath(), deduper);
        }
        for (PathFragment additionalExportedHeader : additionalExportedHeaders) {
          appendHeader(content, additionalExportedHeader, deduper);
        }
        out.write(content.toString().getBytes(StandardCharsets.ISO_8859_1));
      }
    };
  }

  private static Iterable<Artifact> expandedHeaders(ArtifactExpander artifactExpander,
      Iterable<Artifact> unexpandedHeaders) {
    List<Artifact> expandedHeaders = new ArrayList<>();
    for (Artifact unexpandedHeader : unexpandedHeaders) {
      if (unexpandedHeader.isTreeArtifact()) {
        artifactExpander.expand(unexpandedHeader, expandedHeaders);
      } else {
        expandedHeaders.add(unexpandedHeader);
      }
    }
    return ImmutableList.copyOf(expandedHeaders);
  }

  private void appendHeader(StringBuilder content, PathFragment path, 
      HashSet<PathFragment> deduper) {
    if (deduper.contains(path)) {
      return;
    }
    deduper.add(path);
    // #include headers. The #import directive is incompatible with J2ObjC segmented headers.
    content.append("#include \"").append(path).append("\"");
    content.append("\n");
  }
  
  @Override
  public String getMnemonic() {
    return "UmbrellaHeader";
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addPath(umbrellaHeader.getExecPath());
    fp.addInt(publicHeaders.size());
    for (Artifact artifact : publicHeaders) {
      fp.addPath(artifact.getExecPath());
    }
    fp.addInt(additionalExportedHeaders.size());
    for (PathFragment path : additionalExportedHeaders) {
      fp.addPath(path);
    }
  }
}
