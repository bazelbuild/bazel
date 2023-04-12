// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.docgen.starlark;

import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;

/** A documentation page for a list of Starlark global methods in the same environment. */
public final class StarlarkGlobalsDoc extends StarlarkDocPage {
  private final Environment environment;

  public StarlarkGlobalsDoc(Environment environment, StarlarkDocExpander expander) {
    super(expander);
    this.environment = environment;
  }

  @Override
  public String getName() {
    return environment.getPath();
  }

  @Override
  public String getRawDocumentation() {
    return environment.getDescription();
  }

  @Override
  public String getTitle() {
    return environment.getTitle();
  }

  @Override
  public String getSourceFile() {
    return "NONE";
  }
}
