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

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * Interface for a type containing information about the attributes of a rule.
 */
@SkylarkModule(
  name = "rule_attributes",
  category = SkylarkModuleCategory.NONE,
  doc = "Information about attributes of a rule an aspect is applied to."
)
public interface SkylarkAttributesCollectionApi extends SkylarkValue {

  @SkylarkCallable(name = "attr", structField = true, doc = SkylarkRuleContextApi.ATTR_DOC)
  public StructApi getAttr() throws EvalException;

  @SkylarkCallable(
      name = "executable",
      structField = true,
      doc = SkylarkRuleContextApi.EXECUTABLE_DOC)
  public StructApi getExecutable() throws EvalException;

  @SkylarkCallable(name = "file", structField = true, doc = SkylarkRuleContextApi.FILE_DOC)
  public StructApi getFile() throws EvalException;

  @SkylarkCallable(name = "files", structField = true, doc = SkylarkRuleContextApi.FILES_DOC)
  public StructApi getFiles() throws EvalException;

  @SkylarkCallable(
    name = "kind",
    structField = true,
    doc = "The kind of a rule, such as 'cc_library'"
  )
  public String getRuleClassName() throws EvalException;
}
