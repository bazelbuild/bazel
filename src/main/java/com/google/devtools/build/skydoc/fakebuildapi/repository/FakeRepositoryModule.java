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

package com.google.devtools.build.skydoc.fakebuildapi.repository;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.repository.RepositoryModuleApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDescriptor;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkRuleFunctionsApi.AttributeNameComparator;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Fake implementation of {@link RepositoryModuleApi}.
 */
public class FakeRepositoryModule implements RepositoryModuleApi {
  private static final FakeDescriptor IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR =
      new FakeDescriptor(AttributeType.NAME, "A unique name for this repository.", true);

  private final List<RuleInfo> ruleInfoList;

  public FakeRepositoryModule(List<RuleInfo> ruleInfoList) {
    this.ruleInfoList = ruleInfoList;
  }

  @Override
  public BaseFunction repositoryRule(
      BaseFunction implementation,
      Object attrs,
      Boolean local,
      SkylarkList<String> environ,
      String doc,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    List<AttributeInfo> attrInfos;
    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && attrs != Runtime.NONE) {
      SkylarkDict<?, ?> attrsDict = (SkylarkDict<?, ?>) attrs;
      attrsMapBuilder.putAll(attrsDict.getContents(String.class, FakeDescriptor.class, "attrs"));
    }

    attrsMapBuilder.put("name", IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR);
    attrInfos =
        attrsMapBuilder.build().entrySet().stream()
            .filter(entry -> !entry.getKey().startsWith("_"))
            .map(entry -> entry.getValue().asAttributeInfo(entry.getKey()))
            .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    RepositoryRuleDefinitionIdentifier functionIdentifier =
        new RepositoryRuleDefinitionIdentifier();

    ruleInfoList.add(new RuleInfo(functionIdentifier, ast.getLocation(), doc, attrInfos));
    return functionIdentifier;
  }

  /**
   * A fake {@link BaseFunction} implementation which serves as an identifier for a rule definition.
   * A skylark invocation of 'rule()' should spawn a unique instance of this class and return it.
   * Thus, skylark code such as 'foo = rule()' will result in 'foo' being assigned to a unique
   * identifier, which can later be matched to a registered rule() invocation saved by the fake
   * build API implementation.
   */
  private static class RepositoryRuleDefinitionIdentifier extends BaseFunction {

    private static int idCounter = 0;

    public RepositoryRuleDefinitionIdentifier() {
      super("RepositoryRuleDefinitionIdentifier" + idCounter++);
    }

    @Override
    public boolean equals(@Nullable Object other) {
      // Use exact object matching.
      return this == other;
    }
  }
}
