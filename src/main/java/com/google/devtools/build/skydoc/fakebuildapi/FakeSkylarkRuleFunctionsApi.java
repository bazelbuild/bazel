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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.FileTypeApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAspectApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAttrApi.Descriptor;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleFunctionsApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Fake implementation of {@link SkylarkRuleFunctionsApi}.
 *
 * <p>This fake hooks into the global {@code rule()} function, noting calls of that function
 * with a {@link RuleInfoCollector} given in the class constructor.</p>
 */
public class FakeSkylarkRuleFunctionsApi implements SkylarkRuleFunctionsApi<FileApi> {

  private final List<RuleInfo> ruleInfoList;

  /**
   * Constructor.
   *
   * @param ruleInfoList the list of {@link RuleInfo} objects to which rule() invocation information
   *     will be added
   */
  public FakeSkylarkRuleFunctionsApi(List<RuleInfo> ruleInfoList) {
    this.ruleInfoList = ruleInfoList;
  }

  @Override
  public ProviderApi provider(String doc, Object fields, Location location) throws EvalException {
    return null;
  }

  @Override
  public BaseFunction rule(BaseFunction implementation, Boolean test, Object attrs,
      Object implicitOutputs, Boolean executable, Boolean outputToGenfiles,
      SkylarkList<?> fragments, SkylarkList<?> hostFragments, Boolean skylarkTestable,
      SkylarkList<?> toolchains, String doc, SkylarkList<?> providesArg,
      Boolean executionPlatformConstraintsAllowed, SkylarkList<?> execCompatibleWith,
      FuncallExpression ast, Environment funcallEnv) throws EvalException {
    Set<String> attrNames;
    if (attrs != null) {
      SkylarkDict<?, ?> attrsDict = (SkylarkDict<?, ?>) attrs;
      Map<String, Descriptor> attrsMap =
          attrsDict.getContents(String.class, Descriptor.class, "attrs");
      attrNames = attrsMap.keySet();
    } else {
      attrNames = ImmutableSet.of();
    }

    // TODO(cparsons): Improve details given to RuleInfo (for example, attribute types).
    ruleInfoList.add(new RuleInfo(ast.getLocation(), doc, attrNames));
    return implementation;
  }

  @Override
  public Label label(String labelString, Boolean relativeToCallerRepository, Location loc,
      Environment env) throws EvalException {
    return null;
  }

  @Override
  public FileTypeApi<FileApi> fileType(SkylarkList<?> types, Location loc, Environment env)
      throws EvalException {
    return null;
  }

  @Override
  public SkylarkAspectApi aspect(BaseFunction implementation, SkylarkList<?> attributeAspects,
      Object attrs, SkylarkList<?> requiredAspectProvidersArg, SkylarkList<?> providesArg,
      SkylarkList<?> fragments, SkylarkList<?> hostFragments, SkylarkList<?> toolchains, String doc,
      FuncallExpression ast, Environment funcallEnv) throws EvalException {
    return null;
  }
}
