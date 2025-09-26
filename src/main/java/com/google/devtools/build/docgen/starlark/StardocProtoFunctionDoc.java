// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.docgen.starlark;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;

/**
 * A documentation entry for a Starlark function described by a Stardoc proto obtained via {@code
 * starlark_doc_extract} from a .bzl file.
 */
public final class StardocProtoFunctionDoc extends MemberDoc {
  private final String sourceFileLabel;
  private final String structName;
  private final String nameWithoutNamespace;
  private final StarlarkFunctionInfo functionInfo;
  private final ImmutableList<StardocProtoParamDoc> params;
  private final boolean isConstructor;

  public StardocProtoFunctionDoc(
      StarlarkDocExpander expander,
      ModuleInfo moduleInfo,
      String structName,
      StarlarkFunctionInfo functionInfo,
      boolean isConstructor) {
    super(expander);
    this.sourceFileLabel = moduleInfo.getFile();
    this.structName = structName;
    this.nameWithoutNamespace =
        functionInfo.getFunctionName().startsWith(structName + ".")
            ? functionInfo.getFunctionName().substring(structName.length() + 1)
            : functionInfo.getFunctionName();
    this.functionInfo = functionInfo;
    this.params =
        functionInfo.getParameterList().stream()
            .map(paramInfo -> new StardocProtoParamDoc(expander, paramInfo))
            .collect(toImmutableList());
    this.isConstructor = isConstructor;
  }

  public StardocProtoFunctionDoc(
      StarlarkDocExpander expander,
      ModuleInfo moduleInfo,
      String structName,
      StarlarkFunctionInfo functionInfo) {
    this(expander, moduleInfo, structName, functionInfo, /* isConstructor= */ false);
  }

  @Override
  public boolean documented() {
    return true;
  }

  @Override
  public boolean isCallable() {
    return true;
  }

  @Override
  public boolean isConstructor() {
    return isConstructor;
  }

  @Override
  public String getName() {
    return nameWithoutNamespace;
  }

  @Override
  public String getRawDocumentation() {
    // TODO(arostovtsev): emit something useful for 'returns' and 'deprecated' sections.
    return functionInfo.getDocString();
  }

  @Override
  public String getLoadStatement() {
    return String.format(
        "load(\"%s\", \"%s\")",
        sourceFileLabel, structName.isEmpty() ? functionInfo.getFunctionName() : structName);
  }

  @Override
  public String getReturnType() {
    // TODO(arostovtsev): parse return type from 'returns' section of docstring.
    return "unknown";
  }

  @Override
  public ImmutableList<StardocProtoParamDoc> getParams() {
    return params;
  }

  /**
   * Returns a string representing the method signature of the Starlark method, which contains HTML
   * links to the documentation of parameter types if available.
   */
  @Override
  public String getSignature() {
    return String.format(
        "%s %s(%s)", getReturnType(), functionInfo.getFunctionName(), getParameterString());
  }

  /** Documentation for a Starlark function parameter. */
  public static class StardocProtoParamDoc extends ParamDoc {
    private final FunctionParamInfo paramInfo;

    public StardocProtoParamDoc(StarlarkDocExpander expander, FunctionParamInfo paramInfo) {
      super(expander, Kind.fromProto(paramInfo.getRole()));
      this.paramInfo = paramInfo;
    }

    @Override
    public String getName() {
      return paramInfo.getName();
    }

    @Override
    public String getType() {
      // TODO(arostovtsev): parse return type from docstring.
      return "";
    }

    @Override
    public String getDefaultValue() {
      return paramInfo.getDefaultValue();
    }

    @Override
    public String getRawDocumentation() {
      return paramInfo.getDocString();
    }
  }
}
