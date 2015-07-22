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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.Path;

import java.io.Serializable;
import java.util.Map;

/**
 * This creates the //external package, where targets not homed in this repository can be bound.
 */
public class ExternalPackage extends Package {
  public static final String NAME = "external";
  public static final PackageIdentifier PACKAGE_IDENTIFIER =
      PackageIdentifier.createInDefaultRepo(NAME);

  private Map<RepositoryName, Rule> repositoryMap;

  ExternalPackage() {
    super(PACKAGE_IDENTIFIER);
  }

  /**
   * Returns a description of the repository with the given name, or null if there's no such
   * repository.
   */
  public Rule getRepositoryInfo(RepositoryName repositoryName) {
    return repositoryMap.get(repositoryName);
  }

  /**
   * Checks if the given package is //external.
   */
  public static boolean isExternal(Package pkg) {
    return pkg != null && pkg.getName().equals(NAME);
  }

  /**
   * Holder for a binding's actual label and location.
   */
  public static class Binding implements Serializable {
    private final Label actual;
    private final Location location;

    public Binding(Label actual, Location location) {
      this.actual = actual;
      this.location = location;
    }

    public Label getActual() {
      return actual;
    }

    public Location getLocation() {
      return location;
    }

    /**
     * Checks if the label is bound, i.e., starts with {@code //external:}.
     */
    public static boolean isBoundLabel(Label label) {
      return label.getPackageName().equals(NAME);
    }
  }

  /**
   * Given a workspace file path, creates an ExternalPackage.
   */
  public static class Builder extends Package.Builder {
    private Map<RepositoryName, Rule> repositoryMap = Maps.newLinkedHashMap();

    public Builder(Path workspacePath) {
      super(new ExternalPackage());
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
        throws InvalidRuleException, NameConflictException {

      Map<String, Object> attributes = Maps.newHashMap();
      // Bound rules don't have a name field, but this works because we don't want more than one
      // with the same virtual name.
      attributes.put("name", virtual.getName());
      if (actual != null) {
        attributes.put("actual", actual);
      }
      StoredEventHandler handler = new StoredEventHandler();
      Rule rule = RuleFactory.createRule(this, bindRuleClass, attributes, handler, null, location);
      overwriteRule(rule);
      rule.setVisibility(ConstantRuleVisibility.PUBLIC);
    }

    /**
     * Adds the rule to the map of rules. Overwrites rules that are already there, to allow "later"
     * WORKSPACE files to overwrite "earlier" ones.
     */
    public Builder createAndAddRepositoryRule(RuleClass ruleClass, RuleClass bindRuleClass,
        Map<String, Object> kwargs, FuncallExpression ast)
        throws InvalidRuleException, NameConflictException, SyntaxException {
      StoredEventHandler eventHandler = new StoredEventHandler();
      Rule tempRule = RuleFactory.createRule(this, ruleClass, kwargs, eventHandler, ast,
          ast.getLocation());
      addEvents(eventHandler.getEvents());
      repositoryMap.put(RepositoryName.create("@" + tempRule.getName()), tempRule);
      for (Map.Entry<String, Label> entry :
        ruleClass.getExternalBindingsFunction().apply(tempRule).entrySet()) {
        Label nameLabel = Label.parseAbsolute("//external:" + entry.getKey());
        addBindRule(bindRuleClass, nameLabel, entry.getValue(), tempRule.getLocation());
      }
      return this;
    }
  }
}
