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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A category of artifacts that are candidate input/output to an action, for which the toolchain can
 * select a single artifact.
 */
public enum ArtifactCategory {
  STATIC_LIBRARY {
    @Override
    public String getCategoryName() {
      return STATIC_LIBRARY_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  PIC_STATIC_LIBRARY {
    @Override
    public String getCategoryName() {
      return PIC_STATIC_LIBRARY_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  ALWAYS_LINK_STATIC_LIBRARY {
    @Override
    public String getCategoryName() {
      return ALWAYS_LINK_STATIC_LIBRARY_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  ALWAYS_LINK_PIC_STATIC_LIBRARY {
    @Override
    public String getCategoryName() {
      return ALWAYS_LINK_PIC_STATIC_LIBRARY_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  DYNAMIC_LIBRARY {
    @Override
    public String getCategoryName() {
      return DYNAMIC_LIBRARY_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  EXECUTABLE {
    @Override
    public String getCategoryName() {
      return EXECUTABLE_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  INTERFACE {
    @Override
    public String getCategoryName() {
      return INTERFACE_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of("base_name", ruleContext.getLabel().getName());
    }
  },

  DEBUG_SYMBOLS {
    @Override
    public String getCategoryName() {
      return DEBUG_SYMBOL_CATEGORY_NAME;
    }

    @Override
    public ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext) {
      return ImmutableMap.of();
    }
  };

  /** Error for template evaluation failure. */
  @VisibleForTesting
  public static final String TEMPLATE_EVAL_FAILURE =
      "Error evaluating file name pattern for artifact category %s: %s";

  @VisibleForTesting public static final String STATIC_LIBRARY_CATEGORY_NAME = "static_library";

  @VisibleForTesting
  public static final String PIC_STATIC_LIBRARY_CATEGORY_NAME = "pic_static_library";

  @VisibleForTesting
  public static final String ALWAYS_LINK_STATIC_LIBRARY_CATEGORY_NAME = "alwayslink_static_library";

  @VisibleForTesting
  public static final String ALWAYS_LINK_PIC_STATIC_LIBRARY_CATEGORY_NAME =
      "alwayslink_pic_static_library";

  @VisibleForTesting public static final String DYNAMIC_LIBRARY_CATEGORY_NAME = "dynamic_library";
  @VisibleForTesting public static final String EXECUTABLE_CATEGORY_NAME = "executable";
  @VisibleForTesting public static final String INTERFACE_CATEGORY_NAME = "interface_library";
  private static final String DEBUG_SYMBOL_CATEGORY_NAME = "debug_symbol";

  /** Returns the name of the category. */
  public abstract String getCategoryName();

  /** Returns an artifact given a templated name. */
  public Artifact getArtifactForName(String artifactName, RuleContext ruleContext) {
    PathFragment name =
        new PathFragment(ruleContext.getLabel().getName()).replaceName(artifactName);
    return ruleContext.getPackageRelativeArtifact(
        name, ruleContext.getConfiguration().getBinDirectory());
  }

  /**
   * Returns a map of candidate template variables to their values. For example, the entry (foo,
   * bar) indicates that the crosstool artifact name "library_%{foo}.extension" should be templated
   * to "library_bar.extension".
   */
  public abstract ImmutableMap<String, String> getTemplateVariables(RuleContext ruleContext);
}
