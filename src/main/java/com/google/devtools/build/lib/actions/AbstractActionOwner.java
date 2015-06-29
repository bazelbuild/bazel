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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Label;

/**
 * An action owner base class that provides default implementations for some of
 * the {@link ActionOwner} methods.
 */
public abstract class AbstractActionOwner implements ActionOwner {

  @Override
  public String getAdditionalProgressInfo() {
    return null;
  }

  @Override
  public Location getLocation() {
    return null;
  }

  @Override
  public Label getLabel() {
    return null;
  }

  @Override
  public String getTargetKind() {
    return "empty target kind";
  }

  @Override
  public String getConfigurationName() {
    return "empty configuration";
  }

  /**
   * An action owner for special cases. Usage is strongly discouraged. 
   */
  public static final ActionOwner SYSTEM_ACTION_OWNER = new AbstractActionOwner() {
    @Override
    public final String getConfigurationName() {
      return "system";
    }

    @Override
    public String getConfigurationMnemonic() {
      return "system";
    }

    @Override
    public final String getConfigurationChecksum() {
      return "system";
    }
  };
}
