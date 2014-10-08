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

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Map;
import java.util.Map.Entry;

/**
 * This creates the //external package, where targets not homed in this repository can be bound.
 */
public class ExternalPackage extends Package {

  ExternalPackage() {
    super("external");
  }

  /**
   * Holder for a binding's actual label and location.
   */
  public static class Binding {
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
     * Checks if the label is bound, i.e., starts with //external:.
     */
    public static boolean isBoundLabel(Label label) {
      return label.getPackageName().equals("external");
    }
  }

  /**
   * Given a workspace file path, creates an ExternalPackage.
   */
  public static class ExternalPackageBuilder
  extends AbstractBuilder<ExternalPackage, ExternalPackageBuilder> {
    private Map<Label, Binding> bindMap;

    public ExternalPackageBuilder(Path workspacePath) {
      super(new ExternalPackage());
      setFilename(workspacePath);
      setMakeEnv(new MakeEnvironment.Builder());
      bindMap = Maps.newHashMap();
    }

    @Override
    protected ExternalPackageBuilder self() {
      return this;
    }

    @Override
    public ExternalPackageBuilder addEvents(Iterable<Event> events) {
      for (Event event : events) {
        if (event.getKind() == EventKind.ERROR) {
          setContainsErrors();
          break;
        }
      }
      return super.addEvents(events);
    }

    public void addBinding(Label label, Binding binding) {
      bindMap.put(label, binding);
    }

    public void resolveBindTargets(RuleClass ruleClass)
        throws NoSuchBindingException, InvalidRuleException, NameConflictException {
      for (Entry<Label, Binding> entry : bindMap.entrySet()) {
        resolveLabel(entry.getKey(), entry.getValue());
      }

      for (Entry<Label, Binding> entry : bindMap.entrySet()) {
        addRule(ruleClass, entry);
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
          throw new NoSuchBindingException("cycle detected resolving " + virtual + " binding");
        }

        Label previous = actual; // For the exception.
        binding = bindMap.get(actual);
        if (binding == null) {
          throw new NoSuchBindingException("no binding found for target " + previous + " (via "
              + virtual + ")");
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
      Rule rule = RuleFactory.createAndAddRule(this, klass, attributes, handler, null, false,
          location);
      rule.setVisibility(ConstantRuleVisibility.PUBLIC);
    }

    /**
     * This is used when a binding is invalid, either because one of the targets is malformed,
     * refers to a package that does not exist, or creates a circular dependency.
     */
    public class NoSuchBindingException extends NoSuchThingException {
      public NoSuchBindingException(String message) {
        super(message);
      }
    }
  }
}
