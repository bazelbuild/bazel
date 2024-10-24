// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Base class for tests that want to use builders to create platforms and constraints. */
public class PlatformTestCase extends BuildViewTestCase {

  ConstraintBuilder constraintBuilder(String name) {
    return new ConstraintBuilder(name);
  }

  PlatformBuilder platformBuilder(String name) {
    return new PlatformBuilder(name);
  }

  @Nullable
  ConstraintSettingInfo fetchConstraintSettingInfo(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    return PlatformProviderUtils.constraintSetting(target);
  }

  @Nullable
  PlatformInfo fetchPlatformInfo(String platformLabel) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(platformLabel);
    return PlatformProviderUtils.platform(target);
  }

  final class ConstraintBuilder {
    private final Label label;
    private final List<String> constraintValues = new ArrayList<>();
    private String defaultConstraintValue = null;

    public ConstraintBuilder(String name) {
      this.label = Label.parseCanonicalUnchecked(name);
    }

    @CanIgnoreReturnValue
    public ConstraintBuilder defaultConstraintValue(String defaultConstraintValue) {
      this.defaultConstraintValue = defaultConstraintValue;
      this.constraintValues.add(defaultConstraintValue);
      return this;
    }

    @CanIgnoreReturnValue
    public ConstraintBuilder addConstraintValue(String constraintValue) {
      this.constraintValues.add(constraintValue);
      return this;
    }

    public List<String> lines() {
      ImmutableList.Builder<String> lines = ImmutableList.builder();

      // Add the constraint setting.
      lines.add("constraint_setting(name = '" + label.getName() + "',");
      if (!Strings.isNullOrEmpty(defaultConstraintValue)) {
        lines.add("  default_constraint_value = ':" + defaultConstraintValue + "',");
      }
      lines.add(")");

      // Add the constraint values.
      for (String constraintValue : constraintValues) {
        lines.add(
            "constraint_value(",
            "  name = '" + constraintValue + "',",
            "  constraint_setting = ':" + label.getName() + "',",
            ")");
      }

      return lines.build();
    }

    public void write() throws Exception {
      List<String> lines = lines();
      String filename = label.getPackageFragment().getRelative("BUILD").getPathString();
      scratch.appendFile(filename, lines.toArray(new String[] {}));
    }
  }

  final class PlatformBuilder {
    private final Label label;
    private final List<String> constraintValues = new ArrayList<>();
    private Label parentLabel = null;
    private String remoteExecutionProperties = "";
    private ImmutableMap<String, String> execProperties;
    private List<String> flags = new ArrayList<>();

    public PlatformBuilder(String name) {
      this.label = Label.parseCanonicalUnchecked(name);
    }

    @CanIgnoreReturnValue
    public PlatformBuilder setParent(String parentLabel) {
      this.parentLabel = Label.parseCanonicalUnchecked(parentLabel);
      return this;
    }

    @CanIgnoreReturnValue
    public PlatformBuilder addConstraint(String value) {
      this.constraintValues.add(value);
      return this;
    }

    @CanIgnoreReturnValue
    public PlatformBuilder setRemoteExecutionProperties(String value) {
      this.remoteExecutionProperties = value;
      return this;
    }

    @CanIgnoreReturnValue
    public PlatformBuilder setExecProperties(ImmutableMap<String, String> value) {
      this.execProperties = value;
      return this;
    }

    @CanIgnoreReturnValue
    public PlatformBuilder addFlags(String... flags) {
      this.flags.addAll(ImmutableList.copyOf(flags));
      return this;
    }

    public List<String> lines() {
      ImmutableList.Builder<String> lines = ImmutableList.builder();

      lines.add("platform(", "  name = '" + label.getName() + "',");
      if (parentLabel != null) {
        lines.add("  parents = ['" + parentLabel + "'],");
      }
      lines.add("  constraint_values = [");
      for (String name : constraintValues) {
        lines.add("    ':" + name + "',");
      }
      lines.add("  ],");
      if (!Strings.isNullOrEmpty(remoteExecutionProperties)) {
        lines.add("  remote_execution_properties = '" + remoteExecutionProperties + "',");
      }
      if (execProperties != null && !execProperties.isEmpty()) {
        lines.add("  exec_properties = { ");
        for (Map.Entry<String, String> entry : execProperties.entrySet()) {
          lines.add("    \"" + entry.getKey() + "\": \"" + entry.getValue() + "\",");
        }
        lines.add("  }");
      }
      if (!flags.isEmpty()) {
        lines.add("  flags = [");
        for (String flag : flags) {
          lines.add("    '" + flag + "',");
        }
        lines.add("  ],");
      }
      lines.add(")");

      return lines.build();
    }

    public void write() throws Exception {
      List<String> lines = lines();
      String filename = label.getPackageFragment().getRelative("BUILD").getPathString();
      scratch.appendFile(filename, lines.toArray(new String[] {}));
    }
  }
}
