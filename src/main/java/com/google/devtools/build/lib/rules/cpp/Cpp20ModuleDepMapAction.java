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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.*;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import javax.annotation.Nullable;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Creates C++20 module map (.modmap) file.
 * 1. the modules needed for compiling C++ Modules or ordinary sources
 * are provided by the ddi file
 * 2. all modules information provided by the modules info file
 * 3. the output file .modmap collect all modules, including indirect dependencies
 * e.g. a -> b -> c -> d
 * the ddi file only record module a require module b
 * However, in .modmap file, we need to put all modules (b, c, d) in it
 * the format of .modmap file is
 * -fmodule-file=<module-name1>=<module-path1>
 * -fmodule-file=<module-name2>=<module-path2>
 * ...
 * 4. the output file .modmap.input collects all module files' paths
 * it is convenient for CppCompileAction to handle dynamic modules input with the .modmap.input file
 * otherwise, we need to parse the .modmap file and get all modules' path in CppCompileAction
 * the format of .modmap file is
 * <module-path1>
 * <module-path2>
 * ...
 */
@Immutable
public final class Cpp20ModuleDepMapAction extends AbstractAction {
  private static final String GUID = "b02b043a-6c4e-4eeb-992a-09769b366fdd";
  Artifact ddiFile;
  Artifact modulesInfoFile;
  Artifact modmapFile;
  Artifact modmapInputFile;

  public Cpp20ModuleDepMapAction(
      ActionOwner owner,
      Artifact ddiFile,
      Artifact modulesInfoFile,
      Artifact modmapFile,
      Artifact modmapInputFile) {
    super(
        owner,
        NestedSetBuilder.<Artifact>stableOrder().add(ddiFile).add(modulesInfoFile).build(),
        ImmutableSet.of(modmapFile, modmapInputFile));
    this.ddiFile = ddiFile;
    this.modulesInfoFile = modulesInfoFile;
    this.modmapFile = modmapFile;
    this.modmapInputFile = modmapInputFile;
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    try {
      Map<String, String> moduleMap = computeModuleMap(actionExecutionContext);

      var result1 =
          writeOutputToFile(
              actionExecutionContext,
              out -> {
                OutputStreamWriter content =
                    new OutputStreamWriter(out, StandardCharsets.ISO_8859_1);
                for (Map.Entry<String, String> entry : moduleMap.entrySet()) {
                  String moduleName = entry.getKey();
                  String modulePath = entry.getValue();
                  content.append("-f");
                  content.append("module-file=");
                  content.append(moduleName);
                  content.append("=");
                  content.append(modulePath);
                  content.append('\n');
                }
                content.flush();
              },
              modmapFile);
      var result2 =
          writeOutputToFile(
              actionExecutionContext,
              out -> {
                OutputStreamWriter content =
                    new OutputStreamWriter(out, StandardCharsets.ISO_8859_1);
                for (String modulePath : moduleMap.values()) {
                  content.append(modulePath);
                  content.append('\n');
                }
                content.flush();
              },
              modmapInputFile);
      var builder = new ImmutableList.Builder<SpawnResult>();
      builder.addAll(result1);
      builder.addAll(result2);
      return ActionResult.create(builder.build());
    } catch (ExecException e) {
      throw ActionExecutionException.fromExecException(e, this);
    }
  }

  private Map<String, String> computeModuleMap(ActionExecutionContext ctx) throws ExecException {
    String ddi;
    try {
      ddi = FileSystemUtils.readContent(ctx.getInputPath(ddiFile), Charset.defaultCharset());
    } catch (IOException e) {
      throw new EnvironmentalExecException(
          e,
          createFailureDetail(
              String.format("read ddi file fail: %s", ddiFile.getExecPathString()),
              FailureDetails.CppCompile.Code.DDI_FILE_READ_FAILURE));
    }
    var moduleDep = Cpp20ModuleHelper.parseScanResult(ddi);
    String modules;
    try {
      modules =
          FileSystemUtils.readContent(ctx.getInputPath(modulesInfoFile), Charset.defaultCharset());
    } catch (IOException e) {
      throw new EnvironmentalExecException(
          e,
          createFailureDetail(
              String.format("read module info file fail: %s", modulesInfoFile.getExecPathString()),
              FailureDetails.CppCompile.Code.MODULES_INFO_FILE_READ_FAILURE));
    }

    var cxxModules = Cpp20ModuleHelper.Cpp20ModulesInfo.fromJSON(modules);
    Set<String> moduleNameSet = new HashSet<>();
    Queue<String> requireModuleQueue = new ArrayDeque<>();
    if (moduleDep.getRequireModules() != null) {
      requireModuleQueue.addAll(moduleDep.getRequireModules());
    }
    while (!requireModuleQueue.isEmpty()) {
      String requireModuleName = requireModuleQueue.poll();
      moduleNameSet.add(requireModuleName);
      var deps = cxxModules.getRequireModules(requireModuleName);
      for (String dep : deps) {
        if (moduleNameSet.contains(dep)) {
          continue;
        }
        requireModuleQueue.add(dep);
      }
    }
    Map<String, String> moduleMap = new HashMap<>(moduleNameSet.size());
    for (String moduleName : moduleNameSet) {
      String modulePath = cxxModules.getModulePath(moduleName);
      if (modulePath == null) {
        continue;
      }
      moduleMap.put(moduleName, modulePath);
    }
    return moduleMap;
  }

  private ImmutableList<SpawnResult> writeOutputToFile(
      ActionExecutionContext actionExecutionContext,
      DeterministicWriter deterministicWriter,
      Artifact output)
      throws ExecException {
    actionExecutionContext.getEventHandler().post(new RunningActionEvent(this, "local"));
    Path outputPath = actionExecutionContext.getInputPath(output);
    try {
      try (OutputStream out = new BufferedOutputStream(outputPath.getOutputStream())) {
        deterministicWriter.writeOutputFile(out);
      }
    } catch (IOException e) {
      throw new EnvironmentalExecException(
          e, FailureDetails.Execution.Code.FILE_WRITE_IO_EXCEPTION);
    }
    return ImmutableList.of();
  }

  @Override
  public String getMnemonic() {
    return "Cpp20ModuleDepMap";
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    fp.addString(ddiFile.getExecPathString());
    fp.addString(modulesInfoFile.getExecPathString());
    fp.addString(modmapFile.getExecPathString());
  }

  private static FailureDetails.FailureDetail createFailureDetail(
      String message, FailureDetails.CppCompile.Code detailedCode) {
    return FailureDetails.FailureDetail.newBuilder()
        .setMessage(message)
        .setCppCompile(FailureDetails.CppCompile.newBuilder().setCode(detailedCode))
        .build();
  }
}
