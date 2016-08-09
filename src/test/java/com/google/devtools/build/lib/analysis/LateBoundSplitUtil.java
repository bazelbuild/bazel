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

package com.google.devtools.build.lib.analysis;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.UnknownRuleConfiguredTarget;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.common.options.Option;

import java.util.List;

/**
 * Rule and configuration class definitions for testing late-bound split attributes.
 */
public class LateBoundSplitUtil {
  /**
   * A custom {@link FragmentOptions} with the option to be split.
   */
  public static class Options extends FragmentOptions { // public for options loader
    @Option(
      name = "foo",
      defaultValue = "",
      category = "undocumented"
    )
    public String fooFlag;

    @Override
    public List<Attribute.SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
      return ImmutableList.<Attribute.SplitTransition<BuildOptions>>of(SIMPLE_SPLIT);
    }
  }

  /**
   * The split.
   */
  private static final Attribute.SplitTransition<BuildOptions> SIMPLE_SPLIT =
      new Attribute.SplitTransition<BuildOptions>() {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions split1 = buildOptions.clone();
      split1.get(Options.class).fooFlag = "one";
      BuildOptions split2 = buildOptions.clone();
      split2.get(Options.class).fooFlag = "two";
      return ImmutableList.<BuildOptions>of(split1, split2);
    }

    @Override
    public boolean defaultsToSelf() {
      return false;
    }
  };

  /**
   * The {@link BuildConfiguration.Fragment} that contains the options.
   */
  private static class Fragment extends BuildConfiguration.Fragment {
  }

  /**
   * The fragment's loader.
   */
  static class FragmentLoader implements ConfigurationFragmentFactory {
    @Override
    public BuildConfiguration.Fragment create(ConfigurationEnvironment env,
        BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new Fragment();
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
     return Fragment.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  /**
   * The resolver that chooses the late-bound attribute's value.
   */
  private static final Attribute.LateBoundLabel<BuildConfiguration> SIMPLE_LATEBOUND_RESOLVER =
      new Attribute.LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
          return Label.parseAbsoluteUnchecked("//foo:latebound_dep");
        }
      };

  /**
   * A custom rule that applies a late-bound split attribute.
   */
  static class RuleWithLateBoundSplitAttribute implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr(":latebound_split_attr", BuildType.LABEL)
              .allowedFileTypes(FileTypeSet.ANY_FILE)
              .allowedRuleClasses(Attribute.ANY_RULE)
              .cfg(SIMPLE_SPLIT)
              .value(SIMPLE_LATEBOUND_RESOLVER))
          .requiresConfigurationFragments(Fragment.class)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("rule_with_latebound_split")
          .ancestors(BaseRuleClasses.RuleBase.class)
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .build();
    }
  }

  /**
   * Returns a rule class provider with standard test setup plus the above rules/configs.
   */
  static ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(new RuleWithLateBoundSplitAttribute());
    builder.addConfigurationFragment(new FragmentLoader());
    builder.addConfigurationOptions(Options.class);
    return builder.build();
  }
}