// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.skylark.annotations.processor.optiontestsources;

import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.skylark.annotations.SkylarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** A test case of SkylarkConfigurationFieldProcessorTest. */
@SkylarkModule(
    name = "module_name",
    doc = "A fake configuration fragment for a test.",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT)
public class HasMethodParameters extends Fragment {

  /**
   * Returns the label of the xcode_config rule to use for resolving the host system xcode version.
   */
  @SkylarkConfigurationField(
      name = "some_field",
      doc = "Documentation ",
      defaultLabel = "defaultLabel",
      defaultInToolRepository = true
  )
  public Label getXcodeConfigLabel(int x) {
    return null;
  }
}
