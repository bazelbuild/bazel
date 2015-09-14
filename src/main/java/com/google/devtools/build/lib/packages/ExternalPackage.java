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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Map;

/**
 * This creates the //external package, where targets not homed in this repository can be bound.
 */
public class ExternalPackage extends Package {
  public static final String NAME = "external";
  public static final PackageIdentifier PACKAGE_IDENTIFIER =
      PackageIdentifier.createInDefaultRepo(NAME);

  private Map<RepositoryName, Rule> repositoryMap;

  ExternalPackage(String runfilesPrefix) {
    super(PACKAGE_IDENTIFIER, runfilesPrefix);
  }

  /**
   * Returns a description of the repository with the given name, or null if there's no such
   * repository.
   */
  public Rule getRepositoryInfo(RepositoryName repositoryName) {
    return repositoryMap.get(repositoryName);
  }

  /**
   * Given a workspace file path, creates an ExternalPackage.
   */
  public static class Builder extends Package.Builder {
    private Map<RepositoryName, Rule> repositoryMap = Maps.newLinkedHashMap();

    public Builder(Path workspacePath, String runfilesPrefix) {
      super(new ExternalPackage(runfilesPrefix));
      setFilename(workspacePath);
      setMakeEnv(new MakeEnvironment.Builder());
    }

    protected ExternalPackage externalPackage() {
      return (ExternalPackage) pkg;
    }

    @Override
    public ExternalPackage build() {
      for (Rule rule : repositoryMap.values()) {
        try {
          addRule(rule);
        } catch (NameConflictException e) {
          throw new IllegalStateException("Got a name conflict for " + rule
              + ", which can't happen: " + e.getMessage());
        }
      }
      externalPackage().repositoryMap = ImmutableMap.copyOf(repositoryMap);

      Package base = super.build();
      return (ExternalPackage) base;
    }

    /**
     * Sets the name for this repository.
     */
    @Override
    public Builder setWorkspaceName(String workspaceName) {
      pkg.workspaceName = workspaceName;
      return this;
    }

    private void overwriteRule(Rule rule) throws NameConflictException {
      Preconditions.checkArgument(rule.getOutputFiles().isEmpty());
      Target old = targets.get(rule.getName());
      if (old != null) {
        if (old instanceof Rule) {
          Verify.verify(((Rule) old).getOutputFiles().isEmpty());
        }

        targets.remove(rule.getName());
      }

      addRule(rule);
    }

    public void addBindRule(
        RuleClass bindRuleClass, Label virtual, Label actual, Location location)
        throws InvalidRuleException, NameConflictException, InterruptedException {

      Map<String, Object> attributes = Maps.newHashMap();
      // Bound rules don't have a name field, but this works because we don't want more than one
      // with the same virtual name.
      attributes.put("name", virtual.getName());
      if (actual != null) {
        attributes.put("actual", actual);
      }
      StoredEventHandler handler = new StoredEventHandler();
      Rule rule = RuleFactory.createRule(
          this, bindRuleClass, attributes, handler, null, location, null);
      overwriteRule(rule);
      rule.setVisibility(ConstantRuleVisibility.PUBLIC);
    }

    /**
     * Adds the rule to the map of rules. Overwrites rules that are already there, to allow "later"
     * WORKSPACE files to overwrite "earlier" ones.
     */
    public Builder createAndAddRepositoryRule(RuleClass ruleClass, RuleClass bindRuleClass,
        Map<String, Object> kwargs, FuncallExpression ast, Environment env)
        throws InvalidRuleException, NameConflictException, SyntaxException, InterruptedException {
      StoredEventHandler eventHandler = new StoredEventHandler();
      Rule tempRule = RuleFactory.createRule(
          this, ruleClass, kwargs, eventHandler, ast, ast.getLocation(), env);
      addEvents(eventHandler.getEvents());
      try {
        repositoryMap.put(RepositoryName.create("@" + tempRule.getName()), tempRule);
      } catch (TargetParsingException e) {
        throw new SyntaxException(e.getMessage());
      }
      for (Map.Entry<String, Label> entry :
        ruleClass.getExternalBindingsFunction().apply(tempRule).entrySet()) {
        Label nameLabel = Label.parseAbsolute("//external:" + entry.getKey());
        addBindRule(bindRuleClass, nameLabel, entry.getValue(), tempRule.getLocation());
      }
      return this;
    }
  }
}
