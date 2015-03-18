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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.Path;

import java.io.Serializable;
import java.util.Map;
import java.util.Map.Entry;

/**
 * This creates the //external package, where targets not homed in this repository can be bound.
 */
public class ExternalPackage extends Package {
  public static final String NAME = "external";

  private Map<Label, Binding> bindMap;
  private Map<RepositoryName, Rule> repositoryMap;

  ExternalPackage() {
    super(PackageIdentifier.createInDefaultRepo(NAME));
  }

  /**
   * Returns a description of the repository with the given name, or null if there's no such
   * repository.
   */
  public Rule getRepositoryInfo(RepositoryName repositoryName) {
    return repositoryMap.get(repositoryName);
  }

  /**
   * If the given label is bound, returns the (fully resolved) label it is bound to. Otherwise,
   * returns null.
   */
  public Label getActualLabel(Label label) {
    if (bindMap.containsKey(label)) {
      return bindMap.get(label).getActual();
    }
    return null;
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
  public static class Builder
      extends AbstractBuilder<ExternalPackage, Builder> {
    private Map<Label, Binding> bindMap = Maps.newHashMap();
    private Map<RepositoryName, Rule> repositoryMap = Maps.newHashMap();

    public Builder(Path workspacePath) {
      super(new ExternalPackage());
      setFilename(workspacePath);
      setMakeEnv(new MakeEnvironment.Builder());
    }

    @Override
    protected Builder self() {
      return this;
    }

    @Override
    public ExternalPackage build() {
      pkg.bindMap = ImmutableMap.copyOf(bindMap);
      pkg.repositoryMap = ImmutableMap.copyOf(repositoryMap);
      return super.build();
    }

    /**
     * Sets the name for this repository.
     */
    @Override
    public Builder setWorkspaceName(String workspaceName) {
      pkg.workspaceName = workspaceName;
      return this;
    }

    public void addBinding(Label label, Binding binding) {
      bindMap.put(label, binding);
    }

    public void resolveBindTargets(RuleClass ruleClass)
        throws EvalException, NoSuchBindingException {
      for (Entry<Label, Binding> entry : bindMap.entrySet()) {
        resolveLabel(entry.getKey(), entry.getValue());
      }

      for (Entry<Label, Binding> entry : bindMap.entrySet()) {
        try {
          addRule(ruleClass, entry);
        } catch (NameConflictException | InvalidRuleException e) {
          throw new EvalException(entry.getValue().location, e.getMessage());
        }
      }
    }

    // Uses tortoise and the hare algorithm to detect cycles.
    private void resolveLabel(final Label virtual, Binding binding)
        throws NoSuchBindingException {
      Label actual = binding.getActual();
      Label tortoise = virtual;
      Label hare = actual;
      boolean moveTortoise = true;
      while (Binding.isBoundLabel(actual)) {
        if (tortoise == hare) {
          throw new NoSuchBindingException("cycle detected resolving " + virtual + " binding",
              binding.getLocation());
        }

        Label previous = actual; // For the exception.
        Binding oldBinding = binding;
        binding = bindMap.get(actual);
        if (binding == null) {
          throw new NoSuchBindingException("no binding found for target " + previous + " (via "
              + virtual + ")", oldBinding.getLocation());
        }
        actual = binding.getActual();
        hare = actual;
        moveTortoise = !moveTortoise;
        if (moveTortoise) {
          tortoise = bindMap.get(tortoise).getActual();
        }
      }
      bindMap.put(virtual, binding);
    }

    private void addRule(RuleClass klass, Map.Entry<Label, Binding> bindingEntry)
        throws InvalidRuleException, NameConflictException {
      Label virtual = bindingEntry.getKey();
      Label actual = bindingEntry.getValue().actual;
      Location location = bindingEntry.getValue().location;

      Map<String, Object> attributes = Maps.newHashMap();
      // Bound rules don't have a name field, but this works because we don't want more than one
      // with the same virtual name.
      attributes.put("name", virtual.getName());
      attributes.put("actual", actual);
      StoredEventHandler handler = new StoredEventHandler();
      Rule rule = RuleFactory.createAndAddRule(this, klass, attributes, handler, null, location);
      rule.setVisibility(ConstantRuleVisibility.PUBLIC);
    }

    /**
     * Creates an external repository rule.
     * @throws SyntaxException if the repository name is invalid.
     */
    public Builder createAndAddRepositoryRule(RuleClass ruleClass,
        Map<String, Object> kwargs, FuncallExpression ast)
            throws InvalidRuleException, NameConflictException, SyntaxException {
      StoredEventHandler eventHandler = new StoredEventHandler();
      Rule rule = RuleFactory.createAndAddRule(this, ruleClass, kwargs, eventHandler, ast,
          ast.getLocation());
      // Propagate Rule errors to the builder.
      addEvents(eventHandler.getEvents());
      repositoryMap.put(RepositoryName.create("@" + rule.getName()), rule);
      return this;
    }

    /**
     * This is used when a binding is invalid, either because one of the targets is malformed,
     * refers to a package that does not exist, or creates a circular dependency.
     */
    public class NoSuchBindingException extends Exception {
      private Location location;

      public NoSuchBindingException(String message, Location location) {
        super(message);
        this.location = location;
      }

      public Location getLocation() {
        return location;
      }
    }
  }
}
