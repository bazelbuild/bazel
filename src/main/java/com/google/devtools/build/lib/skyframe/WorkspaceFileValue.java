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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.ExternalPackage.Binding;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Map;
import java.util.Map.Entry;

/**
 * The contents of a WORKSPACE file, as label-to-label mappings.
 */
public class WorkspaceFileValue implements SkyValue {

  private String workspaceName;
  private Map<Label, Binding> bindMap;

  WorkspaceFileValue(String workspaceName, Map<Label, Binding> labelMap)
      throws NoSuchBindingException {
    Preconditions.checkNotNull(labelMap);
    this.workspaceName = workspaceName;
    this.bindMap = labelMap;
    resolveLabels();
  }

  private void resolveLabels() throws NoSuchBindingException {
    for (Entry<Label, Binding> entry : bindMap.entrySet()) {
      resolveLabel(entry.getKey(), entry.getValue());
    }
  }

  // Uses tortoise and the hare algorithm to detect cycles.
  private void resolveLabel(final Label virtual, Binding binding)
      throws NoSuchBindingException {
    Label actual = binding.getActual();
    Label tortoise = virtual;
    Label hare = actual;
    boolean moveTortoise = true;
    while (LabelBindingValue.isBoundLabel(actual)) {
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

  /**
   * Returns the name of the repository or 'default' if none is set.
   */
  public String getWorkspace() {
    return workspaceName;
  }

  /**
   * Returns the label virtual is bound to.  Throws NoSuchBindingException if there is no matching
   * binding.
   */
  public Label getActualLabel(Label virtual) throws NoSuchBindingException {
    if (!bindMap.containsKey(virtual)) {
      throw new NoSuchBindingException("no binding found for target " + virtual);
    }
    return bindMap.get(virtual).getActual();
  }

  public Map<Label, Binding> getBindings() {
    return bindMap;
  }

  /**
   * Generates a SkyKey based on the path to the WORKSPACE file.
   */
  public static SkyKey key(Path workspacePath) {
    return new SkyKey(SkyFunctions.WORKSPACE_FILE, workspacePath);
  }

  /**
   * This is used when a binding is invalid, either because one of the targets is malformed, refers
   * to a package that does not exist, or creates a circular dependency.
   */
  public class NoSuchBindingException extends NoSuchThingException {
    public NoSuchBindingException(String message) {
      super(message);
    }
  }

}
