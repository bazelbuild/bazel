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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.analysis.actions.PathMappers;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputPathsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.function.BiFunction;
import javax.annotation.Nullable;

/**
 * Creates C++ module map artifact genfiles. These are then passed to Clang to do dependency
 * checking.
 */
@Immutable
public final class CppModuleMapAction extends AbstractFileWriteAction {

  private static final String GUID = "4f407081-1951-40c1-befc-d6b4daff5de3";
  private static final String MNEMONIC = "CppModuleMap";

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
  private final ImmutableList<Artifact> separateModuleHeaders;
  private final boolean compiledModule;
  private final boolean generateSubmodules;
  private final boolean externDependencies;
  // Save memory by storing only the boolean value of whether path stripping is effectively enabled
  // for this action, which it is if and only if the execution info and the output path mode have
  // the required values. This avoids storing both, which would require two reference fields.
  // See getExecutionInfo() and getOutputPathsMode().
  private final boolean pathStrippingRequestedAndEnabled;

  public CppModuleMapAction(
      ActionOwner owner,
      CppModuleMap cppModuleMap,
      Iterable<Artifact> privateHeaders,
      Iterable<Artifact> publicHeaders,
      Iterable<CppModuleMap> dependencies,
      Iterable<PathFragment> additionalExportedHeaders,
      Iterable<Artifact> separateModuleHeaders,
      boolean compiledModule,
      boolean moduleMapHomeIsCwd,
      boolean generateSubmodules,
      boolean externDependencies,
      OutputPathsMode outputPathsMode,
      BiFunction<ImmutableMap<String, String>, String, ImmutableMap<String, String>>
          modifyExecutionInfo) {
    super(
        owner,
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(Iterables.filter(privateHeaders, Artifact::isTreeArtifact))
            .addAll(Iterables.filter(publicHeaders, Artifact::isTreeArtifact))
            .build(),
        cppModuleMap.getArtifact());
    this.cppModuleMap = cppModuleMap;
    this.moduleMapHomeIsCwd = moduleMapHomeIsCwd;
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.dependencies = ImmutableList.copyOf(dependencies);
    this.additionalExportedHeaders = ImmutableList.copyOf(additionalExportedHeaders);
    this.separateModuleHeaders = ImmutableList.copyOf(separateModuleHeaders);
    this.compiledModule = compiledModule;
    this.generateSubmodules = generateSubmodules;
    this.externDependencies = externDependencies;
    this.pathStrippingRequestedAndEnabled =
        modifyExecutionInfo
                .apply(ImmutableMap.of(), MNEMONIC)
                .containsKey(ExecutionRequirements.SUPPORTS_PATH_MAPPING)
            && outputPathsMode == OutputPathsMode.STRIP;
  }

  @Override
  public boolean makeExecutable() {
    // In theory, module maps should not be executable but, in practice, we don't care. As
    // 'executable' is the default (see ActionOutputMetadataStore.setPathReadOnlyAndExecutable()),
    // we want to avoid the extra file operation of making this file non-executable.
    // Note that the opposite is true for Bazel: making a file executable results in an extra file
    // operation in com.google.devtools.build.lib.exec.FileWriteStrategy.
    return true;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    final ArtifactExpander artifactExpander = ctx.getArtifactExpander();
    // TODO: It is possible that compile actions consuming the module map have path mapping disabled
    //  due to inputs conflicting across configurations. Since these inputs aren't inputs of the
    //  module map action, the generated map still contains mapped paths, which then results in
    //  compilation failures. This should be very rare as #include doesn't allow to disambiguate
    //  between headers from different configurations but with identical root-relative paths.
    final PathMapper pathMapper = PathMappers.create(this, getOutputPathsMode());
    return out -> {
      OutputStreamWriter content = new OutputStreamWriter(out, StandardCharsets.ISO_8859_1);
      PathFragment fragment = pathMapper.map(cppModuleMap.getArtifact().getExecPath());
      int segmentsToExecPath = fragment.segmentCount() - 1;
      Optional<Artifact> umbrellaHeader = cppModuleMap.getUmbrellaHeader();
      String leadingPeriods = moduleMapHomeIsCwd ? "" : "../".repeat(segmentsToExecPath);

      Iterable<Artifact> separateModuleHdrs =
          expandedHeaders(artifactExpander, separateModuleHeaders);

      // For details about the different header types, see:
      // http://clang.llvm.org/docs/Modules.html#header-declaration
      content.append("module \"").append(cppModuleMap.getName()).append("\" {\n");
      content.append("  export *\n");

      HashSet<PathFragment> deduper = new HashSet<>();
      if (umbrellaHeader.isPresent()) {
        appendHeader(
            content,
            "",
            umbrellaHeader.get().getExecPath(),
            leadingPeriods,
            /* canCompile= */ false,
            deduper,
            /*isUmbrellaHeader*/ true,
            pathMapper);
      } else {
        for (Artifact artifact : expandedHeaders(artifactExpander, publicHeaders)) {
          appendHeader(
              content,
              "",
              artifact.getExecPath(),
              leadingPeriods,
              /* canCompile= */ true,
              deduper,
              /*isUmbrellaHeader*/ false,
              pathMapper);
        }
        for (Artifact artifact : expandedHeaders(artifactExpander, privateHeaders)) {
          appendHeader(
              content,
              "private",
              artifact.getExecPath(),
              leadingPeriods,
              /* canCompile= */ true,
              deduper,
              /*isUmbrellaHeader*/ false,
              pathMapper);
        }
        for (Artifact artifact : separateModuleHdrs) {
          appendHeader(
              content,
              "",
              artifact.getExecPath(),
              leadingPeriods,
              /* canCompile= */ false,
              deduper,
              /*isUmbrellaHeader*/ false,
              pathMapper);
        }
        for (PathFragment additionalExportedHeader : additionalExportedHeaders) {
          appendHeader(
              content,
              "",
              additionalExportedHeader,
              leadingPeriods,
              /*canCompile*/ false,
              deduper,
              /*isUmbrellaHeader*/ false,
              pathMapper);
        }
      }
      for (CppModuleMap dep : dependencies) {
        content.append("  use \"").append(dep.getName()).append("\"\n");
      }

      if (!Iterables.isEmpty(separateModuleHdrs)) {
        String separateName = cppModuleMap.getName() + CppModuleMap.SEPARATE_MODULE_SUFFIX;
        content.append("  use \"").append(separateName).append("\"\n");
        content.append("}\n");
        content.append("module \"").append(separateName).append("\" {\n");
        content.append("  export *\n");
        deduper = new HashSet<>();
        for (Artifact artifact : separateModuleHdrs) {
          appendHeader(
              content,
              "",
              artifact.getExecPath(),
              leadingPeriods,
              /* canCompile= */ true,
              deduper,
              /*isUmbrellaHeader*/ false,
              pathMapper);
        }
        for (CppModuleMap dep : dependencies) {
          content.append("  use \"").append(dep.getName()).append("\"\n");
        }
      }
      content.append("}");

      if (externDependencies) {
        for (CppModuleMap dep : dependencies) {
          content
              .append("\nextern module \"")
              .append(dep.getName())
              .append("\" \"")
              .append(leadingPeriods)
              .append(pathMapper.getMappedExecPathString(dep.getArtifact()))
              .append("\"");
        }
      }
      content.flush();
    };
  }

  private static Iterable<Artifact> expandedHeaders(ArtifactExpander artifactExpander,
      Iterable<Artifact> unexpandedHeaders) {
    List<Artifact> expandedHeaders = new ArrayList<>();
    for (Artifact unexpandedHeader : unexpandedHeaders) {
      if (unexpandedHeader.isTreeArtifact()) {
        expandedHeaders.addAll(artifactExpander.expandTreeArtifact(unexpandedHeader));
      } else {
        expandedHeaders.add(unexpandedHeader);
      }
    }

    return ImmutableList.copyOf(expandedHeaders);
  }

  private void appendHeader(
      Appendable content,
      String visibilitySpecifier,
      PathFragment unmappedPath,
      String leadingPeriods,
      boolean canCompile,
      HashSet<PathFragment> deduper,
      boolean isUmbrellaHeader,
      PathMapper pathMapper)
      throws IOException {
    PathFragment path = pathMapper.map(unmappedPath);
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
    return MNEMONIC;
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    fp.addInt(privateHeaders.size());
    for (Artifact artifact : privateHeaders) {
      fp.addPath(artifact.getExecPath());
    }
    fp.addInt(publicHeaders.size());
    for (Artifact artifact : publicHeaders) {
      fp.addPath(artifact.getExecPath());
    }
    fp.addInt(separateModuleHeaders.size());
    for (Artifact artifact : separateModuleHeaders) {
      fp.addPath(artifact.getExecPath());
    }
    fp.addInt(dependencies.size());
    for (CppModuleMap dep : dependencies) {
      fp.addString(dep.getName());
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
    PathMappers.addToFingerprint(getMnemonic(), getExecutionInfo(), getOutputPathsMode(), fp);
  }

  @Override
  public ImmutableMap<String, String> getExecutionInfo() {
    return pathStrippingRequestedAndEnabled
        ? ImmutableMap.of(ExecutionRequirements.SUPPORTS_PATH_MAPPING, "")
        : ImmutableMap.of();
  }

  private OutputPathsMode getOutputPathsMode() {
    return pathStrippingRequestedAndEnabled ? OutputPathsMode.STRIP : OutputPathsMode.OFF;
  }

  @VisibleForTesting
  public CppModuleMap getCppModuleMap() {
    return cppModuleMap;
  }

  @VisibleForTesting
  public ImmutableList<Artifact> getPublicHeaders() {
    return publicHeaders;
  }

  @VisibleForTesting
  public ImmutableList<Artifact> getPrivateHeaders() {
    return privateHeaders;
  }

  @VisibleForTesting
  public ImmutableList<PathFragment> getAdditionalExportedHeaders() {
    return additionalExportedHeaders;
  }

  @VisibleForTesting
  public ImmutableList<Artifact> getSeparateModuleHeaders() {
    return separateModuleHeaders;
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
