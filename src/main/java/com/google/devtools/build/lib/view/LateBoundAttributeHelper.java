// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view;

import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;

import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * A helper class that takes a Rule and creates a map from attributes to labels, resolving any
 * late-bound attributes using the configuration. This can only be done during the analysis phase.
 */
final class LateBoundAttributeHelper {
  private final Rule rule;
  private final BuildConfiguration configuration;
  private final Set<ConfigMatchingProvider> configConditions;

  public LateBoundAttributeHelper(Rule rule, BuildConfiguration configuration,
      Set<ConfigMatchingProvider> configConditions) {
    this.rule = rule;
    this.configuration = configuration;
    this.configConditions = configConditions;
  }

  public ListMultimap<Attribute, Label> createAttributeMap() throws EvalException {
    final ImmutableSortedKeyListMultimap.Builder<Attribute, Label> builder =
        ImmutableSortedKeyListMultimap.builder();
    ConfiguredAttributeMapper attributes = ConfiguredAttributeMapper.of(rule, configConditions);

    attributes.validateAttributes();
    attributes.visitLabels(
        new AttributeMap.AcceptsLabelAttribute() {
          @Override
          public void acceptLabelAttribute(Label label, Attribute attribute) {
            String attributeName = attribute.getName();
            if (attributeName.equals("abi_deps")) {
              // abi_deps is handled specially: we visit only the branch that
              // needs to be taken based on the configuration.
              return;
            }

            if (attribute.getType() == Type.NODEP_LABEL) {
              return;
            }

            if (Attribute.isLateBound(attributeName)) {
              // Late-binding attributes are handled specially.
              return;
            }

            builder.put(attribute, label);
          }
        });

    if (attributes.getAttributeDefinition("abi_deps") != null) {
      Attribute depsAttribute = attributes.getAttributeDefinition("deps");
      MakeVariableExpander.Context context = new ConfigurationMakeVariableContext(
          rule.getPackage(), configuration);
      String abi = null;

      try {
        abi = MakeVariableExpander.expand(attributes.get("abi", Type.STRING), context);
      } catch (MakeVariableExpander.ExpansionException e) {
        // Ignore this. It will be handled during the analysis phase.
      }

      if (abi != null) {
        for (Pair<String, List<Label>> entry : attributes.get("abi_deps", Type.LABEL_LIST_DICT)) {
          try {
            if (Pattern.matches(entry.first, abi)) {
              for (Label label : entry.second) {
                builder.put(depsAttribute, label);
              }
            }
          } catch (PatternSyntaxException e) {
            // Ignore this. It will be handled during the analysis phase.
          }
        }
      }
    }

    // Handle late-bound attributes.
    for (Attribute attribute : rule.getAttributes()) {
      String attributeName = attribute.getName();
      if (Attribute.isLateBound(attributeName) && attribute.getCondition().apply(attributes)) {
        @SuppressWarnings("unchecked")
        LateBoundDefault<BuildConfiguration> lateBoundDefault =
            (LateBoundDefault<BuildConfiguration>) attribute.getLateBoundDefault();
        BuildConfiguration actualConfig = configuration;
        if (lateBoundDefault != null && lateBoundDefault.useHostConfiguration()) {
          actualConfig =
              configuration.getConfiguration(ConfigurationTransition.HOST);
        }

        if (attribute.getType() == Type.LABEL) {
          Label label;
          label = Type.LABEL.cast(lateBoundDefault.getDefault(rule, actualConfig));
          if (label != null) {
            builder.put(attribute, label);
          }
        } else if (attribute.getType() == Type.LABEL_LIST) {
          builder.putAll(attribute, Type.LABEL_LIST.cast(
              lateBoundDefault.getDefault(rule, actualConfig)));
        } else {
          throw new AssertionError("Unknown attribute: '" + attributeName + "'");
        }
      }
    }

    // Handle visibility
    builder.putAll(rule.getRuleClassObject().getAttributeByName("visibility"),
        rule.getVisibility().getDependencyLabels());

    return builder.build();
  }
}
