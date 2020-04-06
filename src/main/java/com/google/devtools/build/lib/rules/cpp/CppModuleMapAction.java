// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

/**
 * Creates C++ module map artifact genfiles. These are then passed to Clang to do dependency
 * checking.
 */
@Immutable
@AutoCodec
public final class CppModuleMapAction extends AbstractFileWriteAction {

  private static final String GUID = "4f407081-1951-40c1-befc-d6b4daff5de3";

  // C++ module map of the current target
  private final CppModuleMap cppModuleMap;

  /**
   * If set, the paths in the module map are relative to the current working directory instead of
   * relative to the module map file's location.
   */
  private final boolean moduleMapHomeIsCwd;

  // Data required to build the actual module map.
  // NOTE: If you add a field here, you'll likely need to add it to the cache key in computeKey().
  private final ImmutableList<Artifact> privateHeaders;
  private final ImmutableList<Artifact> publicHeaders;
  private final ImmutableList<CppModuleMap> dependencies;
  private final ImmutableList<PathFragment> additionalExportedHeaders;
  private final boolean compiledModule;
  private final boolean generateSubmodules;
  private final boolean externDependencies;

  public CppModuleMapAction(
      ActionOwner owner,
      CppModuleMap cppModuleMap,
      Iterable<Artifact> privateHeaders,
      Iterable<Artifact> publicHeaders,
      Iterable<CppModuleMap> dependencies,
      Iterable<PathFragment> additionalExportedHeaders,
      boolean compiledModule,
      boolean moduleMapHomeIsCwd,
      boolean generateSubmodules,
      boolean externDependencies) {
    super(
        owner,
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(Iterables.filter(privateHeaders, Artifact::isTreeArtifact))
            .addAll(Iterables.filter(publicHeaders, Artifact::isTreeArtifact))
            .build(),
        cppModuleMap.getArtifact(),
        /*makeExecutable=*/ false);
    this.cppModuleMap = cppModuleMap;
    this.moduleMapHomeIsCwd = moduleMapHomeIsCwd;
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.dependencies = ImmutableList.copyOf(dependencies);
    this.additionalExportedHeaders = ImmutableList.copyOf(additionalExportedHeaders);
    this.compiledModule = compiledModule;
    this.generateSubmodules = generateSubmodules;
    this.externDependencies = externDependencies;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)  {
    final ArtifactExpander artifactExpander = ctx.getArtifactExpander();
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        OutputStreamWriter content = new OutputStreamWriter(out, StandardCharsets.ISO_8859_1);
        PathFragment fragment = cppModuleMap.getArtifact().getExecPath();
        int segmentsToExecPath = fragment.segmentCount() - 1;
        Optional<Artifact> umbrellaHeader = cppModuleMap.getUmbrellaHeader();

        // For details about the different header types, see:
        // http://clang.llvm.org/docs/Modules.html#header-declaration
        String leadingPeriods = moduleMapHomeIsCwd ? "" : Strings.repeat("../", segmentsToExecPath);
        content.append("module \"").append(cppModuleMap.getName()).append("\" {\n");
        content.append("  export *\n");

        HashSet<PathFragment> deduper = new HashSet<>();
        if (umbrellaHeader.isPresent()) {
          appendHeader(
              content,
              "",
              umbrellaHeader.get().getExecPath(),
              leadingPeriods,
              /*canCompile=*/ false,
              deduper,
              /*isUmbrellaHeader*/ true);
        } else {
          for (Artifact artifact : expandedHeaders(artifactExpander, publicHeaders)) {
            appendHeader(
                content,
                "",
                artifact.getExecPath(),
                leadingPeriods,
                /*canCompile=*/ true,
                deduper,
                /*isUmbrellaHeader*/ false);
          }
          for (Artifact artifact : expandedHeaders(artifactExpander, privateHeaders)) {
            appendHeader(
                content,
                "private",
                artifact.getExecPath(),
                leadingPeriods,
                /*canCompile=*/ true,
                deduper,
                /*isUmbrellaHeader*/ false);
          }
          for (PathFragment additionalExportedHeader : additionalExportedHeaders) {
            appendHeader(
                content,
                "",
                additionalExportedHeader,
                leadingPeriods,
                /*canCompile*/ false,
                deduper,
                /*isUmbrellaHeader*/ false);
          }
        }
        for (CppModuleMap dep : dependencies) {
          content.append("  use \"").append(dep.getName()).append("\"\n");
        }
        content.append("}");
        if (externDependencies) {
          for (CppModuleMap dep : dependencies) {
            content
                .append("\nextern module \"")
                .append(dep.getName())
                .append("\" \"")
                .append(leadingPeriods)
                .append(dep.getArtifact().getExecPathString())
                .append("\"");
          }
        }
        content.flush();
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

  private void appendHeader(Appendable content, String visibilitySpecifier,
      PathFragment path, String leadingPeriods, boolean canCompile, HashSet<PathFragment> deduper,
      boolean isUmbrellaHeader) throws IOException {
    if (deduper.contains(path)) {
      return;
    }
    deduper.add(path);
    if (isUmbrellaHeader) {
      content.append("  umbrella header \"umbrella.h\"\n");
      return;
    }
    if (generateSubmodules) {
      content.append("  module \"").append(path.toString()).append("\" {\n");
      content.append("    export *\n  ");
    }
    content.append("  ");
    if (!visibilitySpecifier.isEmpty()) {
      content.append(visibilitySpecifier).append(" ");
    }
    if (!canCompile || !shouldCompileHeader(path)) {
      content.append("textual ");
    }
    content.append("header \"").append(leadingPeriods).append(path.toString()).append("\"");
    if (generateSubmodules) {
      content.append("\n  }");
    }
    content.append("\n");
  }

  private boolean shouldCompileHeader(PathFragment path) {
    return compiledModule && !CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(path);
  }

  @Override
  public String getMnemonic() {
    return "CppModuleMap";
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addInt(privateHeaders.size());
    for (Artifact artifact : privateHeaders) {
      fp.addPath(artifact.getExecPath());
    }
    fp.addInt(publicHeaders.size());
    for (Artifact artifact : publicHeaders) {
      fp.addPath(artifact.getExecPath());
    }
    fp.addInt(dependencies.size());
    for (CppModuleMap dep : dependencies) {
      fp.addPath(dep.getArtifact().getExecPath());
    }
    fp.addInt(additionalExportedHeaders.size());
    for (PathFragment path : additionalExportedHeaders) {
      fp.addPath(path);
    }
    fp.addPath(cppModuleMap.getArtifact().getExecPath());
    Optional<Artifact> umbrellaHeader = cppModuleMap.getUmbrellaHeader();
    if (umbrellaHeader.isPresent()) {
      fp.addPath(umbrellaHeader.get().getExecPath());
    }
    fp.addString(cppModuleMap.getName());
    fp.addBoolean(moduleMapHomeIsCwd);
    fp.addBoolean(compiledModule);
    fp.addBoolean(generateSubmodules);
    fp.addBoolean(externDependencies);
  }

  @VisibleForTesting
  public CppModuleMap getCppModuleMap() {
    return cppModuleMap;
  }

  @VisibleForTesting
  public Collection<Artifact> getPublicHeaders() {
    return publicHeaders;
  }

  @VisibleForTesting
  public Collection<Artifact> getPrivateHeaders() {
    return privateHeaders;
  }

  @VisibleForTesting
  public ImmutableList<PathFragment> getAdditionalExportedHeaders() {
    return additionalExportedHeaders;
  }

  @VisibleForTesting
  public Collection<Artifact> getDependencyArtifacts() {
    List<Artifact> artifacts = new ArrayList<>();
    for (CppModuleMap map : dependencies) {
      artifacts.add(map.getArtifact());
    }
    return artifacts;
  }
}
