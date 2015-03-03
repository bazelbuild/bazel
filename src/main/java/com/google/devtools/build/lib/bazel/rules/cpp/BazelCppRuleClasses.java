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

package com.google.devtools.build.lib.bazel.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST_DICT;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.packages.Type.TRISTATE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ALWAYS_LINK_LIBRARY;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ALWAYS_LINK_PIC_LIBRARY;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ARCHIVE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.CPP_HEADER;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.CPP_SOURCE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.C_SOURCE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.OBJECT_FILE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.PIC_ARCHIVE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.PIC_OBJECT_FILE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.SHARED_LIBRARY;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.VERSIONED_SHARED_LIBRARY;

import com.google.common.base.Predicates;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.bazel.rules.BazelBaseRuleClasses;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcLibrary;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;

/**
 * Rule class definitions for C++ rules.
 */
public class BazelCppRuleClasses {
  static final SafeImplicitOutputsFunction CC_LIBRARY_DYNAMIC_LIB =
      fromTemplates("%{dirname}lib%{basename}.so");

  static final SafeImplicitOutputsFunction CC_BINARY_IMPLICIT_OUTPUTS =
      fromFunctions(CppRuleClasses.CC_BINARY_STRIPPED, CppRuleClasses.CC_BINARY_DEBUG_PACKAGE);

  static final FileTypeSet ALLOWED_SRC_FILES = FileTypeSet.of(
      CPP_SOURCE,
      C_SOURCE,
      CPP_HEADER,
      ASSEMBLER_WITH_C_PREPROCESSOR,
      ARCHIVE,
      PIC_ARCHIVE,
      ALWAYS_LINK_LIBRARY,
      ALWAYS_LINK_PIC_LIBRARY,
      SHARED_LIBRARY,
      VERSIONED_SHARED_LIBRARY,
      OBJECT_FILE,
      PIC_OBJECT_FILE);

  static final String[] DEPS_ALLOWED_RULES = new String[] {
      "cc_library",
   };

  /**
   * Miscellaneous configuration transitions. It would be better not to have this - please don't add
   * to it.
   */
  public static enum CppTransition implements Transition {
    /**
     * The configuration for LIPO information collection. Requesting this from a configuration that
     * does not have lipo optimization enabled may result in an exception.
     */
    LIPO_COLLECTOR,

    /**
     * The corresponding (target) configuration.
     */
    TARGET_CONFIG_FOR_LIPO;

    @Override
    public boolean defaultsToSelf() {
      return false;
    }
  }

  private static final RuleClass.Configurator<BuildConfiguration, Rule> LIPO_ON_DEMAND =
      new RuleClass.Configurator<BuildConfiguration, Rule>() {
    @Override
    public BuildConfiguration apply(Rule rule, BuildConfiguration configuration) {
      BuildConfiguration toplevelConfig =
          configuration.getConfiguration(CppTransition.TARGET_CONFIG_FOR_LIPO);
      // If LIPO is enabled, override the default configuration.
      if (toplevelConfig != null
          && toplevelConfig.getFragment(CppConfiguration.class).isLipoOptimization()
          && !configuration.isHostConfiguration()
          && !configuration.getFragment(CppConfiguration.class).isLipoContextCollector()) {
        // Switch back to data when the cc_binary is not the LIPO context.
        return (rule.getLabel().equals(
            toplevelConfig.getFragment(CppConfiguration.class).getLipoContextLabel()))
            ? toplevelConfig
            : configuration.getTransitions().getConfiguration(ConfigurationTransition.DATA);
      }
      return configuration;
    }
  };

  /**
   * Label of a pseudo-filegroup that contains all crosstool and libcfiles for
   * all configurations, as specified on the command-line.
   */
  public static final String CROSSTOOL_LABEL = "//tools/defaults:crosstool";

  public static final LateBoundLabel<BuildConfiguration> CC_TOOLCHAIN =
      new LateBoundLabel<BuildConfiguration>(CROSSTOOL_LABEL) {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(CppConfiguration.class).getCcToolchainRuleLabel();
        }
      };

  public static final LateBoundLabel<BuildConfiguration> DEFAULT_MALLOC =
      new LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(CppConfiguration.class).customMalloc();
        }
      };

  public static final LateBoundLabel<BuildConfiguration> STL =
      new LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return getStl(rule, configuration);
        }
      };

  /**
   * Implementation for the :lipo_context_collector attribute.
   */
  public static final LateBoundLabel<BuildConfiguration> LIPO_CONTEXT_COLLECTOR =
      new LateBoundLabel<BuildConfiguration>() {
    @Override
    public Label getDefault(Rule rule, BuildConfiguration configuration) {
      // This attribute connects a target to the LIPO context target configured with the
      // lipo input collector configuration.
      CppConfiguration cppConfiguration = configuration.getFragment(CppConfiguration.class);
      return !cppConfiguration.isLipoContextCollector()
          && (cppConfiguration.getLipoMode() == LipoMode.BINARY)
          ? cppConfiguration.getLipoContextLabel()
          : null;
    }
  };

  /**
   * Returns the STL prerequisite of the rule.
   *
   * <p>If rule has an implicit $stl attribute returns STL version set on the
   * command line or if not set, the value of the $stl attribute. Returns
   * {@code null} otherwise.
   */
  private static Label getStl(Rule rule, BuildConfiguration original) {
    Label stl = null;
    if (rule.getRuleClassObject().hasAttr("$stl", Type.LABEL)) {
      Label stlConfigLabel = original.getFragment(CppConfiguration.class).getStl();
      Label stlRuleLabel = RawAttributeMapper.of(rule).get("$stl", Type.LABEL);
      if (stlConfigLabel == null) {
        stl = stlRuleLabel;
      } else if (!stlConfigLabel.equals(rule.getLabel()) && stlRuleLabel != null) {
        // prevents self-reference and a cycle through standard STL in the dependency graph
        stl = stlConfigLabel;
      }
    }
    return stl;
  }

  /**
   * Common attributes for all rules that create C++ links. This may
   * include non-cc_* rules (e.g. py_binary).
   */
  @BlazeRule(name = "$cc_linking_rule",
               type = RuleClassType.ABSTRACT)
  public static final class CcLinkingRule implements RuleDefinition {
    @Override
    @SuppressWarnings("unchecked")
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(":cc_toolchain", LABEL).value(CC_TOOLCHAIN))
          .setPreferredDependencyPredicate(Predicates.<String>or(CPP_SOURCE, C_SOURCE, CPP_HEADER))
          .build();
    }
  }

  /**
   * Common attributes for C++ rules.
   */
  @BlazeRule(name = "$cc_base_rule",
               type = RuleClassType.ABSTRACT,
               ancestors = { CcLinkingRule.class })
  public static final class CcBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("copts", STRING_LIST))
          .add(attr("$stl", LABEL).value(env.getLabel("//tools/cpp:stl")))
          .add(attr(":stl", LABEL).value(STL))
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  @BlazeRule(name = "$cc_decl_rule",
               type = RuleClassType.ABSTRACT,
               ancestors = { BaseRuleClasses.RuleBase.class })
  public static final class CcDeclRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          // Deprecated in favor of configurable attributes.
          .add(attr("abi", STRING).value("$(ABI)"))
          // Deprecated in favor of configurable attributes.
          .add(attr("abi_deps", LABEL_LIST_DICT))
          .add(attr("defines", STRING_LIST))
          .add(attr("includes", STRING_LIST))
          .add(attr(":lipo_context_collector", LABEL)
              .cfg(CppTransition.LIPO_COLLECTOR)
              .value(LIPO_CONTEXT_COLLECTOR))
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  @BlazeRule(name = "$cc_rule",
             type = RuleClassType.ABSTRACT,
             ancestors = { CcDeclRule.class, CcBaseRule.class })
  public static final class CcRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          .add(attr("srcs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(ALLOWED_SRC_FILES))
          .override(attr("deps", LABEL_LIST)
              .allowedRuleClasses(DEPS_ALLOWED_RULES)
              .allowedFileTypes()
              .skipAnalysisTimeFileTypeCheck())
          .add(attr("linkopts", STRING_LIST))
          .add(attr("nocopts", STRING))
          .add(attr("hdrs_check", STRING).value("strict"))
          .add(attr("linkstatic", BOOLEAN).value(true))
          .override(attr("$stl", LABEL).value(new Attribute.ComputedDefault() {
            @Override
            public Object getDefault(AttributeMap rule) {
              // Every cc_rule depends implicitly on STL to make
              // sure that the correct headers are used for inclusion. The only exception is
              // STL itself to avoid cycles in the dependency graph.
              Label stl = env.getLabel("//tools/cpp:stl");
              return rule.getLabel().equals(stl) ? null : stl;
            }
          }))
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  @BlazeRule(name = "$cc_binary_base",
               type = RuleClassType.ABSTRACT,
               ancestors = CcRule.class)
  public static final class CcBinaryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("malloc", LABEL)
              .value(env.getLabel("//tools/cpp:malloc"))
              .allowedFileTypes()
              .allowedRuleClasses("cc_library"))
          .add(attr(":default_malloc", LABEL).value(DEFAULT_MALLOC))
          .add(attr("stamp", TRISTATE).value(TriState.AUTO))
          .build();
    }
  }

  /**
   * Rule definition for cc_binary rules.
   */
  @BlazeRule(name = "cc_binary",
               ancestors = { CcBinaryBaseRule.class,
                             BazelBaseRuleClasses.BinaryBaseRule.class },
               factoryClass = BazelCcBinary.class)
  public static final class CcBinaryRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setImplicitOutputsFunction(CC_BINARY_IMPLICIT_OUTPUTS)
          .add(attr("linkshared", BOOLEAN).value(false)
              .nonconfigurable("used to *determine* the rule's configuration"))
          .cfg(LIPO_ON_DEMAND)
          .build();
    }
  }
  
  /**
   * Implementation for the :lipo_context attribute.
   */
  private static final LateBoundLabel<BuildConfiguration> LIPO_CONTEXT =
      new LateBoundLabel<BuildConfiguration>() {
    @Override
    public Label getDefault(Rule rule, BuildConfiguration configuration) {
      Label result = configuration.getFragment(CppConfiguration.class).getLipoContextLabel();
      return (rule == null || rule.getLabel().equals(result)) ? null : result;
    }
  };
  
  /**
   * Rule definition for cc_test rules.
   */
  @BlazeRule(name = "cc_test",
      type = RuleClassType.TEST,
      ancestors = { CcBinaryBaseRule.class, BaseRuleClasses.TestBaseRule.class },
      factoryClass = BazelCcTest.class)
  public static final class CcTestRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setImplicitOutputsFunction(CppRuleClasses.CC_BINARY_DEBUG_PACKAGE)
          .override(attr("linkstatic", BOOLEAN).value(false))
          .override(attr("stamp", TRISTATE).value(TriState.NO))
          .add(attr(":lipo_context", LABEL).value(LIPO_CONTEXT))
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  @BlazeRule(name = "$cc_library",
               type = RuleClassType.ABSTRACT,
               ancestors = { CcRule.class })
  public static final class CcLibraryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("hdrs", LABEL_LIST).orderIndependent().direct_compile_time_input()
              .allowedFileTypes(CPP_HEADER))
          .add(attr("linkstamp", LABEL).allowedFileTypes(CPP_SOURCE, C_SOURCE))
          .build();
    }
  }

  /**
   * Rule definition for the cc_library rule.
   */
  @BlazeRule(name = "cc_library",
               ancestors = { CcLibraryBaseRule.class},
               factoryClass = BazelCcLibrary.class)
  public static final class CcLibraryRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      SafeImplicitOutputsFunction implicitOutputsFunction = new SafeImplicitOutputsFunction() {
        @Override
        public Iterable<String> getImplicitOutputs(AttributeMap rule) {
          boolean alwaysLink = rule.get("alwayslink", Type.BOOLEAN);
          boolean linkstatic = rule.get("linkstatic", Type.BOOLEAN);
          SafeImplicitOutputsFunction staticLib = fromTemplates(
              alwaysLink
                  ? "%{dirname}lib%{basename}.lo"
                  : "%{dirname}lib%{basename}.a");
          SafeImplicitOutputsFunction allLibs =
              linkstatic || CcLibrary.appearsToHaveNoObjectFiles(rule)
              ? staticLib
              : fromFunctions(staticLib, CC_LIBRARY_DYNAMIC_LIB);
          return allLibs.getImplicitOutputs(rule);
        }
      };

      return builder
          .setImplicitOutputsFunction(implicitOutputsFunction)
          .add(attr("alwayslink", BOOLEAN).
              nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
          .add(attr("implements", LABEL_LIST)
              .allowedFileTypes()
              .allowedRuleClasses("cc_public_library$headers"))
          .override(attr("linkstatic", BOOLEAN).value(false)
              .nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
          .build();
    }
  }
}
