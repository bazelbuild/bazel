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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDescriptor;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkRuleFunctionsApi.AttributeNameComparator;
import com.google.devtools.build.skydoc.rendering.RuleInfoWrapper;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Fake implementation of {@link RepositoryModuleApi}.
 */
public class FakeRepositoryModule implements RepositoryModuleApi {
  private static final FakeDescriptor IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR =
      new FakeDescriptor(
          AttributeType.NAME,
          Optional.of("A unique name for this repository."),
          true,
          ImmutableList.of(),
          "");

  private static final FakeDescriptor IMPLICIT_REPO_MAPPING_ATTRIBUTE_DESCRIPTOR =
      new FakeDescriptor(
          AttributeType.STRING_DICT,
          Optional.of(
              "A dictionary from local repository name to global repository name. "
                  + "This allows controls over workspace dependency resolution for dependencies of "
                  + "this repository."
                  + "<p>For example, an entry `\"@foo\": \"@bar\"` declares that, for any time "
                  + "this repository depends on `@foo` (such as a dependency on "
                  + "`@foo//some:target`, it should actually resolve that dependency within "
                  + "globally-declared `@bar` (`@bar//some:target`)."),
          true,
          ImmutableList.of(),
          "");

  private final List<RuleInfoWrapper> ruleInfoList;

  public FakeRepositoryModule(List<RuleInfoWrapper> ruleInfoList) {
    this.ruleInfoList = ruleInfoList;
  }

  @Override
  public StarlarkCallable repositoryRule(
      StarlarkCallable implementation,
      Object attrs,
      Boolean local,
      Sequence<?> environ, // <String> expected
      Boolean configure,
      Boolean remotable,
      Object doc,
      StarlarkThread thread)
      throws EvalException {
    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && attrs != Starlark.NONE) {
      attrsMapBuilder.putAll(Dict.cast(attrs, String.class, FakeDescriptor.class, "attrs"));
    }

    attrsMapBuilder.put("name", IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR);
    attrsMapBuilder.put("repo_mapping", IMPLICIT_REPO_MAPPING_ATTRIBUTE_DESCRIPTOR);
    List<AttributeInfo> attrInfos =
        attrsMapBuilder.build().entrySet().stream()
            .filter(entry -> !entry.getKey().startsWith("_"))
            .map(entry -> entry.getValue().asAttributeInfo(entry.getKey()))
            .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    RepositoryRuleDefinitionIdentifier functionIdentifier =
        new RepositoryRuleDefinitionIdentifier();

    // Only the Builder is passed to RuleInfoWrapper as the rule name is not yet available.
    RuleInfo.Builder ruleInfo = RuleInfo.newBuilder().addAllAttribute(attrInfos);
    Starlark.toJavaOptional(doc, String.class)
        .map(Starlark::trimDocString)
        .ifPresent(ruleInfo::setDocString);
    Location loc = thread.getCallerLocation();
    ruleInfoList.add(new RuleInfoWrapper(functionIdentifier, loc, ruleInfo));
    return functionIdentifier;
  }

  /**
   * A fake {@link StarlarkCallable} implementation which serves as an identifier for a rule
   * definition. A Starlark invocation of 'rule()' should spawn a unique instance of this class and
   * return it. Thus, Starlark code such as 'foo = rule()' will result in 'foo' being assigned to a
   * unique identifier, which can later be matched to a registered rule() invocation saved by the
   * fake build API implementation.
   */
  private static class RepositoryRuleDefinitionIdentifier implements StarlarkCallable {

    private static int idCounter = 0;
    private final String name = "RepositoryRuleDefinitionIdentifier" + idCounter++;

    @Override
    public String getName() {
      return name;
    }
  }

  @Override
  public Object moduleExtension(
      StarlarkCallable implementation,
      Dict<?, ?> tagClasses,
      Object doc,
      Sequence<?> environ,
      StarlarkThread thread)
      throws EvalException {
    return new Object();
  }

  @Override
  public TagClassApi tagClass(Dict<?, ?> attrs, Object doc, StarlarkThread thread)
      throws EvalException {
    return new TagClassApi() {};
  }

  @Override
  public void failWithIncompatibleUseCcConfigureFromRulesCc(StarlarkThread thread)
      throws EvalException {
    // Noop until --incompatible_use_cc_configure_from_rules_cc is implemented.
  }
}
