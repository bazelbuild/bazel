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
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A documentation entry for a Starlark function described by a Stardoc proto obtained via {@code
 * starlark_doc_extract} from a .bzl file.
 */
public final class StardocProtoFunctionDoc extends MemberDoc {
  private final String sourceFileLabel;
  private final String structName;
  private final String nameWithoutNamespace;
  private final StarlarkFunctionInfo functionInfo;
  private final TypeParser.TypedDocstring typedReturnDocstring;
  private final ImmutableList<StardocProtoParamDoc> params;
  @Nullable private final String constructorType;

  public StardocProtoFunctionDoc(
      StarlarkDocExpander expander,
      ModuleInfo moduleInfo,
      String structName,
      StarlarkFunctionInfo functionInfo,
      @Nullable String constructorType) {
    super(expander);
    this.sourceFileLabel = moduleInfo.getFile();
    this.structName = structName;
    this.nameWithoutNamespace =
        functionInfo.getFunctionName().startsWith(structName + ".")
            ? functionInfo.getFunctionName().substring(structName.length() + 1)
            : functionInfo.getFunctionName();
    this.functionInfo = functionInfo;
    this.constructorType = constructorType;
    if (constructorType == null) {
      this.typedReturnDocstring =
          TypeParser.TypedDocstring.of(functionInfo.getReturn().getDocString());
    } else {
      // Constructors always return the type they construct
      this.typedReturnDocstring = new TypeParser.TypedDocstring(constructorType, "");
    }
    this.params =
        functionInfo.getParameterList().stream()
            .map(
                paramInfo ->
                    new StardocProtoParamDoc(expander, sourceFileLabel, functionInfo, paramInfo))
            .collect(toImmutableList());
  }

  public StardocProtoFunctionDoc(
      StarlarkDocExpander expander,
      ModuleInfo moduleInfo,
      String structName,
      StarlarkFunctionInfo functionInfo) {
    this(expander, moduleInfo, structName, functionInfo, /* constructorType= */ null);
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
    return constructorType != null;
  }

  @Override
  public String getName() {
    return nameWithoutNamespace;
  }

  @Override
  public String getRawDocumentation() {
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
    try {
      // TODO(arostovtsev): the "unknown" fallback text should be provided by the template.
      return expander.getTypeParser().getHtml(typedReturnDocstring.typeExpression(), "unknown");
    } catch (EvalException e) {
      throw new IllegalStateException(
          String.format(
              "Failed to parse return type for %s in %s",
              functionInfo.getFunctionName(), sourceFileLabel),
          e);
    }
  }

  @Override
  public String getReturnsStanza() {
    return expander.expand(typedReturnDocstring.remainder());
  }

  @Override
  public String getDeprecatedStanza() {
    // A provider constructor's deprecation stanza applies to the provider it constructs.
    return isConstructor() ? "" : expander.expand(functionInfo.getDeprecated().getDocString());
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
    private final String sourceFileLabel;
    private final StarlarkFunctionInfo functionInfo;
    private final FunctionParamInfo paramInfo;
    private final TypeParser.TypedDocstring typedDocstring;

    public StardocProtoParamDoc(
        StarlarkDocExpander expander,
        String sourceFileLabel,
        StarlarkFunctionInfo functionInfo,
        FunctionParamInfo paramInfo) {
      super(expander, Kind.fromProto(paramInfo.getRole()));
      this.sourceFileLabel = sourceFileLabel;
      this.functionInfo = functionInfo;
      this.paramInfo = paramInfo;
      this.typedDocstring = TypeParser.TypedDocstring.of(paramInfo.getDocString());
    }

    @Override
    public String getName() {
      return paramInfo.getName();
    }

    @Override
    public String getType() {
      try {
        // TODO(arostovtsev): the fallback text should be provided by the template.
        return expander.getTypeParser().getHtml(typedDocstring.typeExpression());
      } catch (EvalException e) {
        throw new IllegalStateException(
            String.format(
                "Failed to parse type for param %s of %s in %s",
                getName(), functionInfo.getFunctionName(), sourceFileLabel),
            e);
      }
    }

    @Override
    public String getDefaultValue() {
      return paramInfo.getDefaultValue();
    }

    @Override
    public String getRawDocumentation() {
      return paramInfo.getDocString();
    }

    @Override
    public String getDocumentation() {
      return expander.expand(typedDocstring.remainder());
    }
  }
}
