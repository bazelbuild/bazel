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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Label;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Tool for chasing filegroup redirects. This is mainly intended to be used during
 * BuildConfiguration creation.
 */
public final class RedirectChaser {

  /**
   * Custom attribute mapper that throws an exception if an attribute's value depends on the
   * build configuration.
   */
  private static class StaticValuedAttributeMapper extends AbstractAttributeMapper {
    public StaticValuedAttributeMapper(Rule rule) {
      super(rule.getPackage(), rule.getRuleClassObject(), rule.getLabel(),
          rule.getAttributeContainer());
    }

    /**
     * Returns the value of the given attribute.
     *
     * @throws InvalidConfigurationException if the value is configuration-dependent
     */
    public <T> T getAndValidate(String attributeName, Type<T> type)
        throws InvalidConfigurationException {
      if (getSelectorList(attributeName, type) != null) {
        throw new InvalidConfigurationException
            ("The value of '" + attributeName + "' cannot be configuration-dependent");
      }
      return super.get(attributeName, type);
    }
  }

  /**
   * Follows the 'srcs' attribute of the given label recursively. Keeps repeating as long as the
   * labels are filegroups with a single srcs entry.
   *
   * @param env for loading the packages
   * @param label the label to start at
   * @param name user-meaningful description of the content being resolved
   * @return the label which cannot be further resolved
   * @throws InvalidConfigurationException if something goes wrong
   */
  @Nullable
  public static Label followRedirects(ConfigurationEnvironment env, Label label, String name)
      throws InvalidConfigurationException {
    Set<Label> visitedLabels = new HashSet<>();
    visitedLabels.add(label);
    try {
      while (true) {
        Target possibleRedirect = env.getTarget(label);
        if (possibleRedirect == null) {
          return null;
        }
        if ((possibleRedirect instanceof Rule) &&
            "filegroup".equals(((Rule) possibleRedirect).getRuleClass())) {
          List<Label> labels = new StaticValuedAttributeMapper((Rule) possibleRedirect)
              .getAndValidate("srcs", Type.LABEL_LIST);
          if (labels.size() != 1) {
            // We can't distinguish redirects from the final filegroup, so we assume this must be
            // the final one.
            return label;
          }
          label = labels.get(0);
          if (!visitedLabels.add(label)) {
            throw new InvalidConfigurationException("The " + name + " points to a filegroup which "
                + "recursively includes itself. The label " + label + " is part of the loop");
          }
        } else {
          return label;
        }
      }
    } catch (NoSuchPackageException e) {
      throw new InvalidConfigurationException(e.getMessage(), e);
    } catch (NoSuchTargetException e) {
      return label;
    }
  }
}
