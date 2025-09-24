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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;

/**
 * A documentation page for a Starlark provider described by a Stardoc proto obtained via {@code
 * starlark_doc_extract} from a .bzl file.
 */
public final class StardocProtoProviderDocPage extends StarlarkDocPage {
  private final String sourceFileLabel;
  private final ProviderInfo providerInfo;

  public StardocProtoProviderDocPage(
      StarlarkDocExpander expander, ModuleInfo moduleInfo, ProviderInfo providerInfo) {
    super(expander);
    this.sourceFileLabel = moduleInfo.getFile();
    this.providerInfo = providerInfo;

    if (providerInfo.hasInit()) {
      // TODO(arostovtsev): hard-code return type to the provider's own symbol.
      setConstructor(
          new StardocProtoFunctionDoc(
              expander,
              moduleInfo,
              providerInfo.getProviderName(),
              providerInfo.getInit(),
              /* isConstructor= */ true));
    }
    for (ProviderFieldInfo fieldInfo : providerInfo.getFieldInfoList()) {
      addMember(new ProviderFieldDoc(expander, fieldInfo));
    }
  }

  @Override
  public String getName() {
    return providerInfo.getProviderName();
  }

  @Override
  public String getRawDocumentation() {
    return providerInfo.getDocString();
  }

  @Override
  public String getTitle() {
    return providerInfo.getProviderName();
  }

  @Override
  public String getSourceFile() {
    return getSourceFileFromLabel(sourceFileLabel);
  }

  @Override
  public String getLoadStatement() {
    String loadableSymbol = Splitter.on('.').splitToList(providerInfo.getProviderName()).getFirst();
    return String.format("load(\"%s\", \"%s\")", sourceFileLabel, loadableSymbol);
  }

  private static final class ProviderFieldDoc extends MemberDoc {
    private final ProviderFieldInfo fieldInfo;

    ProviderFieldDoc(StarlarkDocExpander expander, ProviderFieldInfo fieldInfo) {
      super(expander);
      this.fieldInfo = fieldInfo;
    }

    @Override
    public String getName() {
      return fieldInfo.getName();
    }

    @Override
    public boolean documented() {
      return true;
    }

    @Override
    public boolean isCallable() {
      return false;
    }

    @Override
    public ImmutableList<? extends ParamDoc> getParams() {
      return ImmutableList.of();
    }

    @Override
    public String getReturnType() {
      // TODO(arostovtsev): Parse return type from doc string.
      return "unknown";
    }

    @Override
    public String getRawDocumentation() {
      return fieldInfo.getDocString();
    }

    @Override
    public String getSignature() {
      return String.format("%s %s", getReturnType(), getName());
    }
  }
}
