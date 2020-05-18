// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Interface for a type containing information about the attributes of a rule. */
@StarlarkBuiltin(
    name = "rule_attributes",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc = "Information about attributes of a rule an aspect is applied to.")
public interface StarlarkAttributesCollectionApi extends StarlarkValue {

  @StarlarkMethod(name = "attr", structField = true, doc = StarlarkRuleContextApi.ATTR_DOC)
  StructApi getAttr() throws EvalException;

  @StarlarkMethod(
      name = "executable",
      structField = true,
      doc = StarlarkRuleContextApi.EXECUTABLE_DOC)
  StructApi getExecutable() throws EvalException;

  @StarlarkMethod(name = "file", structField = true, doc = StarlarkRuleContextApi.FILE_DOC)
  StructApi getFile() throws EvalException;

  @StarlarkMethod(name = "files", structField = true, doc = StarlarkRuleContextApi.FILES_DOC)
  StructApi getFiles() throws EvalException;

  @StarlarkMethod(
      name = "kind",
      structField = true,
      doc = "The kind of a rule, such as 'cc_library'")
  String getRuleClassName() throws EvalException;
}
