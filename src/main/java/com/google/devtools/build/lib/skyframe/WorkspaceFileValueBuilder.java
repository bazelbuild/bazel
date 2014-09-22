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

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.packages.ExternalPackage.Binding;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue.NoSuchBindingException;
import com.google.devtools.build.lib.syntax.Label;

import java.util.Map;

/**
 * Used by {@link WorkspaceFileFunction} to build a {@link WorkspaceFileValue} from the contents of
 * the WORKSPACE file.
 */
public class WorkspaceFileValueBuilder {
  private String workspaceName;
  private Map<Label, Binding> bindMap;

  WorkspaceFileValueBuilder() {
    workspaceName = "default";
    bindMap = Maps.newHashMap();
  }

  public WorkspaceFileValue build() throws NoSuchBindingException {
    return new WorkspaceFileValue(workspaceName, bindMap);
  }

  /**
   * Set the repository name.
   */
  public void setWorkspaceName(String name) {
    workspaceName = name;
  }

  /**
   * Add a binding to the bind map.
   */
  public void addBinding(Label nameLabel, Binding binding) {
    bindMap.put(nameLabel, binding);
  }
}
