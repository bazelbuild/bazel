// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Creates C++ module map artifact genfiles. These are then passed to Clang to
 * do dependency checking.
 */
public class CppModuleMapAction extends AbstractAction {

  private static final String GUID = "4f407081-1951-40c1-befc-d6b4daff5de3";

  // C++ module map of the current target
  private final CppModuleMap cppModuleMap;

  // Headers and dependencies list
  private final ImmutableList<Artifact> privateHeaders;
  private final ImmutableList<Artifact> publicHeaders;
  private final ImmutableList<CppModuleMap> dependencies;

  public CppModuleMapAction(ActionOwner owner, CppModuleMap cppModuleMap,
      Iterable<Artifact> privateHeaders, Iterable<Artifact> publicHeaders,
      Iterable<CppModuleMap> dependencies) {
    super(owner, ImmutableList.<Artifact>of(), ImmutableList.of(cppModuleMap.getArtifact()));
    this.cppModuleMap = cppModuleMap;
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.dependencies = ImmutableList.copyOf(dependencies);
  }

  @Override
  public void execute(
      ActionExecutionContext actionExecutionContext) throws ActionExecutionException {
    StringBuilder content = new StringBuilder();
    PathFragment fragment = cppModuleMap.getArtifact().getExecPath();
    int segmentsToExecPath = fragment.segmentCount() - 1;

    content.append("module \"" + cppModuleMap.getName() + "\" {\n");
    for (Artifact artifact : privateHeaders) {
      if (!CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(artifact.getExecPath())) {
        content.append("  private header \"" + Strings.repeat("../", segmentsToExecPath)
            + artifact.getExecPath() + "\"\n");
      } else {
        content.append("  exclude header \"" + Strings.repeat("../", segmentsToExecPath)
            + artifact.getExecPath() + "\"\n");        
      }
    }
    for (Artifact artifact : publicHeaders) {
      if (!CppFileTypes.CPP_TEXTUAL_INCLUDE.matches(artifact.getExecPath())) {
        content.append("  header \""
            + Strings.repeat("../", segmentsToExecPath)
            + artifact.getExecPath() + "\"\n");
      } else {
        content.append("  exclude header \"" + Strings.repeat("../", segmentsToExecPath)
            + artifact.getExecPath() + "\"\n");
      }
    }
    for (CppModuleMap dep : dependencies) {
      content.append("  use \"" + dep.getName() + "\"\n");
    }
    content.append("}");
    for (CppModuleMap dep : dependencies) {
      content.append("\nextern module \"" + dep.getName() + "\" \""
          + Strings.repeat("../", segmentsToExecPath)
          + dep.getArtifact().getExecPath() + "\"");
    }

    try {
      FileSystemUtils.writeIsoLatin1(cppModuleMap.getArtifact().getPath(), content.toString());
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create C++ module map '"
          + cppModuleMap.getArtifact().prettyPrint() + "' due to I/O error: " + e.getMessage(),
          e, this, false);
    }
  }

  @Override
  public String describeStrategy(Executor executor) {
    return "local";
  }

  @Override
  public String getMnemonic() {
    return "CppModuleMap";
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addInt(privateHeaders.size());
    for (Artifact artifact : privateHeaders) {
      f.addPath(artifact.getRootRelativePath());
    }
    f.addInt(publicHeaders.size());
    for (Artifact artifact : publicHeaders) {
      f.addPath(artifact.getRootRelativePath());
    }
    f.addInt(dependencies.size());
    for (CppModuleMap dep : dependencies) {
      f.addPath(dep.getArtifact().getExecPath());
    }
    f.addPath(cppModuleMap.getArtifact().getPath());
    f.addString(cppModuleMap.getName());
    return f.hexDigest();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return new ResourceSet(/*memoryMb=*/0, /*cpuUsage=*/0, /*ioUsage=*/0.02);
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
  public Collection<Artifact> getDependencyArtifacts() {
    List<Artifact> artifacts = new ArrayList<>();
    for (CppModuleMap map : dependencies) {
      artifacts.add(map.getArtifact());
    }
    return artifacts;
  }
}
