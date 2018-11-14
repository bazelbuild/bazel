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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAttrApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.skydoc.rendering.AttributeInfo.Type;

/**
 * Fake implementation of {@link SkylarkAttrApi}.
 */
public class FakeSkylarkAttrApi implements SkylarkAttrApi {

  @Override
  public Descriptor intAttribute(Integer defaultInt, String doc, Boolean mandatory,
      SkylarkList<?> values, FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.INT, doc, mandatory);
  }

  @Override
  public Descriptor stringAttribute(String defaultString, String doc, Boolean mandatory,
      SkylarkList<?> values, FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.STRING, doc, mandatory);
  }

  @Override
  public Descriptor labelAttribute(Object defaultO, String doc, Boolean executable,
      Object allowFiles, Object allowSingleFile, Boolean mandatory, SkylarkList<?> providers,
      Object allowRules, Boolean singleFile, Object cfg, SkylarkList<?> aspects,
      FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.LABEL, doc, mandatory);
  }

  @Override
  public Descriptor stringListAttribute(Boolean mandatory, Boolean nonEmpty, Boolean allowEmpty,
      SkylarkList<?> defaultList, String doc, FuncallExpression ast, Environment env)
      throws EvalException {
    return new FakeDescriptor(Type.STRING_LIST, doc, mandatory);
  }

  @Override
  public Descriptor intListAttribute(Boolean mandatory, Boolean nonEmpty, Boolean allowEmpty,
      SkylarkList<?> defaultList, String doc, FuncallExpression ast, Environment env)
      throws EvalException {
    return new FakeDescriptor(Type.INT_LIST, doc, mandatory);
  }

  @Override
  public Descriptor labelListAttribute(Boolean allowEmpty, Object defaultList, String doc,
      Object allowFiles, Object allowRules, SkylarkList<?> providers, SkylarkList<?> flags,
      Boolean mandatory, Boolean nonEmpty, Object cfg, SkylarkList<?> aspects,
      FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.LABEL_LIST, doc, mandatory);
  }

  @Override
  public Descriptor labelKeyedStringDictAttribute(Boolean allowEmpty, Object defaultList,
      String doc, Object allowFiles, Object allowRules, SkylarkList<?> providers,
      SkylarkList<?> flags, Boolean mandatory, Boolean nonEmpty, Object cfg, SkylarkList<?> aspects,
      FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.LABEL_STRING_DICT, doc, mandatory);
  }

  @Override
  public Descriptor boolAttribute(Boolean defaultO, String doc, Boolean mandatory,
      FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.BOOLEAN, doc, mandatory);
  }

  @Override
  public Descriptor outputAttribute(Object defaultO, String doc, Boolean mandatory,
      FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.OUTPUT, doc, mandatory);
  }

  @Override
  public Descriptor outputListAttribute(Boolean allowEmpty, Object defaultList, String doc,
      Boolean mandatory, Boolean nonEmpty, FuncallExpression ast, Environment env)
      throws EvalException {
    return new FakeDescriptor(Type.OUTPUT_LIST, doc, mandatory);
  }

  @Override
  public Descriptor stringDictAttribute(Boolean allowEmpty, SkylarkDict<?, ?> defaultO, String doc,
      Boolean mandatory, Boolean nonEmpty, FuncallExpression ast, Environment env)
      throws EvalException {
    return new FakeDescriptor(Type.STRING_DICT, doc, mandatory);
  }

  @Override
  public Descriptor stringListDictAttribute(Boolean allowEmpty, SkylarkDict<?, ?> defaultO,
      String doc, Boolean mandatory, Boolean nonEmpty, FuncallExpression ast, Environment env)
      throws EvalException {
    return new FakeDescriptor(Type.STRING_LIST_DICT, doc, mandatory);
  }

  @Override
  public Descriptor licenseAttribute(Object defaultO, String doc, Boolean mandatory,
      FuncallExpression ast, Environment env) throws EvalException {
    return new FakeDescriptor(Type.LICENSE, doc, mandatory);
  }

  @Override
  public void repr(SkylarkPrinter printer) {}
}
