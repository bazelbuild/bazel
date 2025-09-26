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

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
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
}
