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
package com.google.devtools.build.lib.rules.java;

import java.util.Map;
import java.util.Properties;

/**
 * A class to describe how build information should be translated into the generated properties
 * file.
 */
public interface BuildInfoPropertiesTranslator {

  /** Translate build information into a property file. */
  public void translate(Map<String, String> buildInfo, Properties properties);

  /**
   * Returns a unique key for this translator to be used by the {@link
   * com.google.devtools.build.lib.actions.ActionExecutionMetadata#getKey(com.google.devtools.build.lib.actions.ActionKeyContext)}
   * method.
   */
  public String computeKey();
}
