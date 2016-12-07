// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skylarkinterface;

/** A category of a Java type exposed to Skylark */
public enum SkylarkModuleCategory {
  CONFIGURATION_FRAGMENT("Configuration Fragments",
      "Configuration fragments give rules access to "
      + "language-specific parts of <a href=\"/docs/skylark/lib/configuration.html\">"
      + "configuration</a>. "
      + "<p>Rule implementations can get them using "
      + "<code><a href=\"ctx.html#fragments\">ctx."
      + "fragments</a>.<i>[fragment name]</i></code>"),

  PROVIDER("Providers",
      "<a href=\"../rules.md#providers\">Providers</a> give rules access "
      + " to information from their dependencies. "
      + " <p>This section lists providers available on built-in rules. Rule implementation "
      + " functions can access them via "
      + "<code><a href=\"Target.html\">target</a>.<i>[provider name]</i></code>."
  ),

  BUILTIN("Built-in Types and Modules", ""),

  TOP_LEVEL_TYPE,
  NONE;

  private final String title;
  private final String description;


  SkylarkModuleCategory(String title, String description) {
    this.title = title;
    this.description = description;
  }

  SkylarkModuleCategory() {
    this.title = null;
    this.description = null;
  }

  public String getTemplateIdentifier() {
    return name().toLowerCase().replace("_", "-");
  }

  public String getTitle() {
    return title;
  }

  public String getDescription() {
    return description;
  }
}
