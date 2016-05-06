// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * A Skylark value that is a result of an 'aspect(..)' function call.
 */
@SkylarkModule(name = "aspect", doc = "", documented = false)
public class SkylarkAspect implements SkylarkValue {
  private final BaseFunction implementation;
  private final ImmutableList<String> attributeAspects;
  private final ImmutableList<Attribute> attributes;
  private final ImmutableSet<String> paramAttributes;
  private final ImmutableSet<String> fragments;
  private final ImmutableSet<String> hostFragments;
  private final Environment funcallEnv;
  private SkylarkAspectClass aspectClass;

  public SkylarkAspect(
      BaseFunction implementation,
      ImmutableList<String> attributeAspects,
      ImmutableList<Attribute> attributes,
      ImmutableSet<String> paramAttributes,
      ImmutableSet<String> fragments,
      ImmutableSet<String> hostFragments,
      Environment funcallEnv) {
    this.implementation = implementation;
    this.attributeAspects = attributeAspects;
    this.attributes = attributes;
    this.paramAttributes = paramAttributes;
    this.fragments = fragments;
    this.hostFragments = hostFragments;
    this.funcallEnv = funcallEnv;
    ImmutableList.Builder<Pair<String, Attribute>> builder = ImmutableList.builder();
  }

  public BaseFunction getImplementation() {
    return implementation;
  }

  public ImmutableList<String> getAttributeAspects() {
    return attributeAspects;
  }

  public Environment getFuncallEnv() {
    return funcallEnv;
  }

  public ImmutableList<Attribute> getAttributes() {
    return attributes;
  }

  @Override
  public boolean isImmutable() {
    return implementation.isImmutable();
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.append(buffer, "Aspect:");
    implementation.write(buffer, quotationMark);
  }

  public String getName() {
    return getAspectClass().getName();
  }

  public SkylarkAspectClass getAspectClass() {
    Preconditions.checkState(isExported());
    return aspectClass;
  }

  public ImmutableSet<String> getParamAttributes() {
    return paramAttributes;
  }

  public void export(Label extensionLabel, String name) {
    Preconditions.checkArgument(!isExported());
    this.aspectClass = new SkylarkAspectClass(extensionLabel, name);
  }

  public AspectDefinition getDefinition(AspectParameters aspectParams) {
    AspectDefinition.Builder builder = new AspectDefinition.Builder(getName());
    for (String attributeAspect : attributeAspects) {
      builder.attributeAspect(attributeAspect, aspectClass);
    }
    for (Attribute attribute : attributes) {
      Attribute attr = attribute;  // Might be reassigned.
      if (!aspectParams.getAttribute(attr.getName()).isEmpty()) {
        String value = aspectParams.getOnlyValueOfAttribute(attr.getName());
        Preconditions.checkState(!Attribute.isImplicit(attr.getName()));
        Preconditions.checkState(attr.getType() == Type.STRING);
        Preconditions.checkArgument(aspectParams.getAttribute(attr.getName()).size() == 1,
            String.format("Aspect %s parameter %s has %d values (must have exactly 1).",
                          getName(),
                          attr.getName(),
                          aspectParams.getAttribute(attr.getName()).size()));
        attr = attr.cloneBuilder(Type.STRING).value(value).build(attr.getName());
      }
      builder.add(attr);
    }
    builder.requiresConfigurationFragmentsBySkylarkModuleName(fragments);
    builder.requiresHostConfigurationFragmentsBySkylarkModuleName(hostFragments);
    return builder.build();
  }

  public boolean isExported() {
    return aspectClass != null;
  }

  public Function<Rule, AspectParameters> getDefaultParametersExtractor() {
    return new Function<Rule, AspectParameters>() {
      @Nullable
      @Override
      public AspectParameters apply(Rule rule) {
        AttributeMap ruleAttrs = RawAttributeMapper.of(rule);
        AspectParameters.Builder builder = new AspectParameters.Builder();
        for (Attribute aspectAttr : attributes) {
          if (!Attribute.isImplicit(aspectAttr.getName())) {
            String param = aspectAttr.getName();
            Attribute ruleAttr = ruleAttrs.getAttributeDefinition(param);
            if (paramAttributes.contains(aspectAttr.getName())) {
              // These are preconditions because if they are false, RuleFunction.call() should
              // already have generated an error.
              Preconditions.checkArgument(ruleAttr != null,
                  String.format("Cannot apply aspect %s to %s that does not define attribute '%s'.",
                                getName(),
                                rule.getTargetKind(),
                                param));
              Preconditions.checkArgument(ruleAttr.getType() == Type.STRING,
                  String.format("Cannot apply aspect %s to %s with non-string attribute '%s'.",
                                getName(),
                                rule.getTargetKind(),
                                param));
            }
            if (ruleAttr != null && ruleAttr.getType() == aspectAttr.getType()) {
              builder.addAttribute(param, (String) ruleAttrs.get(param, ruleAttr.getType()));
            }
          }
        }
        return builder.build();
      }
    };
  }
}
