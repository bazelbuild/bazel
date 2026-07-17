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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkOtherSymbolInfo;

/**
 * A documentation page for a Starlark struct described by a Stardoc proto obtained via {@code
 * starlark_doc_extract} from a .bzl file.
 */
public final class StardocProtoStructDocPage extends StarlarkDocPage {
  private final String sourceFileLabel;
  private final StarlarkOtherSymbolInfo structInfo;

  public StardocProtoStructDocPage(
      StarlarkDocExpander expander, ModuleInfo moduleInfo, StarlarkOtherSymbolInfo structInfo) {
    super(expander);
    this.sourceFileLabel = moduleInfo.getFile();
    this.structInfo = structInfo;
  }

  @Override
  public String getName() {
    return structInfo.getName();
  }

  @Override
  public String getRawDocumentation() {
    return structInfo.getDoc();
  }

  @Override
  public String getTitle() {
    return structInfo.getName();
  }

  @Override
  public String getSourceFile() {
    return getSourceFileFromLabel(sourceFileLabel);
  }

  @Override
  public String getLoadStatement() {
    return String.format("load(\"%s\", \"%s\")", sourceFileLabel, structInfo.getName());
  }

  public void addProviderAlias(ProviderInfo providerInfo) {
    addMember(new ProviderAliasDoc(expander, sourceFileLabel, structInfo.getName(), providerInfo));
  }

  private static final class ProviderAliasDoc extends MemberDoc {
    private final String sourceFileLabel;
    private final String structName;
    private final String nameWithoutNamespace;
    private final ProviderInfo providerInfo;

    ProviderAliasDoc(
        StarlarkDocExpander expander,
        String sourceFileLabel,
        String structName,
        ProviderInfo providerInfo) {
      super(expander);
      this.sourceFileLabel = sourceFileLabel;
      this.structName = structName;
      this.nameWithoutNamespace =
          providerInfo.getProviderName().startsWith(structName + ".")
              ? providerInfo.getProviderName().substring(structName.length() + 1)
              : providerInfo.getProviderName();
      this.providerInfo = providerInfo;
    }

    @Override
    public String getName() {
      return nameWithoutNamespace;
    }

    @Override
    public boolean documented() {
      return true;
    }

    @Override
    public boolean isCallable() {
      // For simplicity, we document a provider alias in its role as a symbol.
      return false;
    }

    @Override
    public ImmutableList<? extends ParamDoc> getParams() {
      return ImmutableList.of();
    }

    @Override
    public String getReturnType() {
      return expander.getTypeParser().getHtmlForIdentifier("Provider");
    }

    @Override
    public String getRawDocumentation() {
      return String.format("A convenience alias for the %s provider symbol.", getAliasedName());
    }

    @Override
    public String getDocumentation() {
      return String.format(
          "A convenience alias for the %s provider symbol.",
          expander.getTypeParser().getHtmlForIdentifier(getAliasedName()));
    }

    /**
     * Returns the documented name of the provider symbol that this one is aliasing; or this
     * provider's name without the struct namespace as fallback.
     */
    private String getAliasedName() {
      if (expander.getTypeParser().isDocumentedIdentifier(providerInfo.getOriginKey().getName())) {
        return providerInfo.getOriginKey().getName();
      } else {
        return getName();
      }
    }

    @Override
    public String getSignature() {
      return String.format("%s %s", getReturnType(), getName());
    }

    @Override
    public String getLoadStatement() {
      return String.format("load(\"%s\", \"%s\")", sourceFileLabel, structName);
    }
  }
}
