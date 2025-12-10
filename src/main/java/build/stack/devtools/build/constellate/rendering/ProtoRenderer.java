// Copyright 2019 The Bazel Authors. All rights reserved.
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

package build.stack.devtools.build.constellate.rendering;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.starlarkdocextract.ExtractionException;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.StarlarkFunction;

/** Produces output in proto form. */
public class ProtoRenderer {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final List<RuleInfo> ruleInfos;
  private final List<ProviderInfo> providerInfos;
  private final List<AspectInfo> aspectInfos;
  private final List<StarlarkFunctionInfo> functionInfos;
  private final List<RepositoryRuleInfo> repositoryRuleInfos;
  private final List<ModuleExtensionInfo> moduleExtensionInfos;
  private final List<MacroInfo> macroInfos;
  private final List<String> errors;
  private String moduleDocstring;

  public ProtoRenderer() {
    this.ruleInfos = new ArrayList<>();
    this.providerInfos = new ArrayList<>();
    this.aspectInfos = new ArrayList<>();
    this.functionInfos = new ArrayList<>();
    this.repositoryRuleInfos = new ArrayList<>();
    this.moduleExtensionInfos = new ArrayList<>();
    this.macroInfos = new ArrayList<>();
    this.errors = new ArrayList<>();
    this.moduleDocstring = "";
  }

  /** Appends {@link RuleInfo} protos. */
  public ProtoRenderer appendRuleInfos(Collection<RuleInfo> ruleInfos) {
    this.ruleInfos.addAll(ruleInfos);
    return this;
  }

  /** Appends {@link ProviderInfo} protos. */
  public ProtoRenderer appendProviderInfos(Collection<ProviderInfo> providerInfos) {
    this.providerInfos.addAll(providerInfos);
    return this;
  }

  /**
   * Appends {@link StarlarkFunctionInfo} protos.
   * If a function's docstring is malformed, collects the error and continues processing other functions.
   */
  public ProtoRenderer appendStarlarkFunctionInfos(Map<String, StarlarkFunction> funcInfosMap) {
    for (Map.Entry<String, StarlarkFunction> entry : funcInfosMap.entrySet()) {
      try {
        StarlarkFunctionInfo funcInfo = FunctionUtil.fromNameAndFunction(entry.getKey(), entry.getValue());
        this.functionInfos.add(funcInfo);
      } catch (ExtractionException e) {
        // Don't let one bad docstring prevent extracting docs for other functions in the file
        String errorMsg = String.format(
            "Skipping function '%s': %s",
            entry.getKey(),
            e.getMessage());
        logger.atWarning().log("%s", errorMsg);
        errors.add(errorMsg);
      }
    }
    return this;
  }

  /** Sets the module docstring. */
  public ProtoRenderer setModuleDocstring(String moduleDoc) {
    this.moduleDocstring = moduleDoc != null ? moduleDoc : "";
    return this;
  }

  /** Appends {@link AspectInfo} protos. */
  public ProtoRenderer appendAspectInfos(Collection<AspectInfo> aspectInfos) {
    this.aspectInfos.addAll(aspectInfos);
    return this;
  }

  /** Appends {@link RepositoryRuleInfo} protos. */
  public ProtoRenderer appendRepositoryRuleInfos(Collection<RepositoryRuleInfo> repositoryRuleInfos) {
    this.repositoryRuleInfos.addAll(repositoryRuleInfos);
    return this;
  }

  /** Appends {@link ModuleExtensionInfo} protos. */
  public ProtoRenderer appendModuleExtensionInfos(Collection<ModuleExtensionInfo> moduleExtensionInfos) {
    this.moduleExtensionInfos.addAll(moduleExtensionInfos);
    return this;
  }

  /** Appends {@link MacroInfo} protos. */
  public ProtoRenderer appendMacroInfos(Collection<MacroInfo> macroInfos) {
    this.macroInfos.addAll(macroInfos);
    return this;
  }

  /** Returns the collected rule infos. */
  public List<RuleInfo> getRuleInfos() {
    return ruleInfos;
  }

  /** Returns the collected provider infos. */
  public List<ProviderInfo> getProviderInfos() {
    return providerInfos;
  }

  /** Returns the collected aspect infos. */
  public List<AspectInfo> getAspectInfos() {
    return aspectInfos;
  }

  /** Returns the collected function infos. */
  public List<StarlarkFunctionInfo> getFunctionInfos() {
    return functionInfos;
  }

  /** Returns the collected repository rule infos. */
  public List<RepositoryRuleInfo> getRepositoryRuleInfos() {
    return repositoryRuleInfos;
  }

  /** Returns the collected module extension infos. */
  public List<ModuleExtensionInfo> getModuleExtensionInfos() {
    return moduleExtensionInfos;
  }

  /** Returns the collected macro infos. */
  public List<MacroInfo> getMacroInfos() {
    return macroInfos;
  }

  /** Returns the module docstring. */
  public String getModuleDocstring() {
    return moduleDocstring;
  }

  /** Returns the list of errors collected during extraction. */
  public List<String> getErrors() {
    return errors;
  }
}
