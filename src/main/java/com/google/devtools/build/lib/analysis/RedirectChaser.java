// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;

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
        Label newLabel = getFilegroupRedirect(possibleRedirect);
        if (newLabel == null) {
          newLabel = getBindRedirect(possibleRedirect);
        }

        if (newLabel == null) {
          return label;
        }

        newLabel = label.resolveRepositoryRelative(newLabel);
        label = newLabel;
        if (!visitedLabels.add(label)) {
          throw new InvalidConfigurationException("The " + name + " points to a filegroup which "
              + "recursively includes itself. The label " + label + " is part of the loop");
        }
      }
    } catch (NoSuchPackageException e) {
      env.getEventHandler().handle(Event.error(e.getMessage()));
      throw new InvalidConfigurationException(e.getMessage(), e);
    } catch (NoSuchTargetException e) {
      // TODO(ulfjack): Consider throwing an exception here instead of returning silently.
      return label;
    }
  }

  private static Label getFilegroupRedirect(Target target) throws InvalidConfigurationException {
    if (!(target instanceof Rule)) {
      return null;
    }

    Rule rule = (Rule) target;
    if (!rule.getRuleClass().equals("filegroup")) {
      return null;
    }

    List<Label> labels =
        new StaticValuedAttributeMapper(rule).getAndValidate("srcs", BuildType.LABEL_LIST);
    if (labels.size() != 1) {
      return null;
    }

    return labels.get(0);
  }

  private static Label getBindRedirect(Target target) throws InvalidConfigurationException {
    if (!(target instanceof Rule)) {
      return null;
    }

    Rule rule = (Rule) target;
    if (!rule.getRuleClass().equals("bind")) {
      return null;
    }

    return new StaticValuedAttributeMapper(rule).getAndValidate("actual", BuildType.LABEL);
  }
}
