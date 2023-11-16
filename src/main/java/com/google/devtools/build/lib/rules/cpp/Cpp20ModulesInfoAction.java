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

import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import javax.annotation.Nullable;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

/**
 * Creates C++20 Modules info file (<target-name>.CXXModules.json)
 * e.g.
 * {
 *     "modules": {
 *         "a": "path/to/bmi/of/module/a",
 *         "b": "path/to/bmi/of/module/b",
 *     },
 *     "usages": {
 *         "a": ["b"]
 *     }
 * }
 * Each target has only one to record all modules information
 * 1. modules: the map of module-name -> module bmi
 * 2. usages: the direct dependencies of each module
 */
@Immutable
public final class Cpp20ModulesInfoAction extends AbstractFileWriteAction {

  private static final String GUID = "dd122dd8-72c2-4c98-b343-fac327adb923";
  NestedSet<Artifact> ddiFiles;
  ImmutableList<Pair<Artifact, Artifact>> pcmAndDdiPairList;
  NestedSet<Artifact> modulesInfoFiles;
  public Cpp20ModulesInfoAction(
      ActionOwner owner,
      NestedSet<Artifact> ddiFiles,
      ImmutableList<Pair<Artifact, Artifact>> pcmAndDdiPairList,
      NestedSet<Artifact> modulesInfoFiles,
      Artifact modulesInfoFile
      ) {
    super(
        owner,
        collectInput(ddiFiles, modulesInfoFiles),
        modulesInfoFile,
        /*makeExecutable=*/ true);
    this.ddiFiles = ddiFiles;
    this.pcmAndDdiPairList = pcmAndDdiPairList;
    this.modulesInfoFiles = modulesInfoFiles;
  }
  private static NestedSet<Artifact> collectInput(NestedSet<Artifact> ddiFiles, NestedSet<Artifact> modulesInfoFiles) {
    return NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(ddiFiles)
            .addTransitive(modulesInfoFiles)
            .build();
  }
  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)  {
    return out -> {
      var cxxModules = new Cpp20ModuleHelper.Cpp20ModulesInfo();
      OutputStreamWriter content = new OutputStreamWriter(out, StandardCharsets.ISO_8859_1);
      for (Pair<Artifact, Artifact> pair : pcmAndDdiPairList) {
        Artifact pcmFile = pair.first;
        Artifact ddiFile = pair.second;
        String s = FileSystemUtils.readContent(ctx.getInputPath(ddiFile), Charset.defaultCharset());
        var moduleDep = Cpp20ModuleHelper.parseScanResult(s);
        if (moduleDep.isNeedProduceBMI() && pcmFile != null) {
          cxxModules.putModulePath(moduleDep.getModuleName(), pcmFile.getExecPathString());
        }
        if (moduleDep.getRequireModules()!=null) {
          for (String requireModule : moduleDep.getRequireModules()) {
            cxxModules.addRequireModule(moduleDep.getModuleName(), requireModule);
          }
        }
      }
      Gson gson = new GsonBuilder().setPrettyPrinting().create();
      for (Artifact depModulesInfoFile : modulesInfoFiles.toList()) {
        String modulesInfoContent = FileSystemUtils.readContent(ctx.getInputPath(depModulesInfoFile), Charset.defaultCharset());
        Cpp20ModuleHelper.Cpp20ModulesInfo depCpp20ModulesInfo = Cpp20ModuleHelper.Cpp20ModulesInfo.fromJSON(modulesInfoContent);
        cxxModules.merge(depCpp20ModulesInfo);
      }
      String json = gson.toJson(cxxModules);
      content.append(json);
      content.flush();
    };
  }

  @Override
  public String getMnemonic() {
    return "Cpp20ModulesInfo";
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    ImmutableList<Artifact> ddiFileList = ddiFiles.toList();
    fp.addInt(ddiFileList.size());
    for (Artifact artifact : ddiFileList) {
      fp.addString(artifact.getExecPathString());
    }
  }

}
