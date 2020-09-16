// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;

/** Stores information about a Starlark rule definition. */
public class RuleInfoWrapper {

  private final StarlarkCallable identifierFunction;
  private final Location location;
  // Only the Builder is passed to RuleInfoWrapper as the rule name is not yet available.
  private final RuleInfo.Builder ruleInfo;

  public RuleInfoWrapper(
      StarlarkCallable identifierFunction, Location location, RuleInfo.Builder ruleInfo) {
    this.identifierFunction = identifierFunction;
    this.location = location;
    this.ruleInfo = ruleInfo;
  }

  public StarlarkCallable getIdentifierFunction() {
    return identifierFunction;
  }

  public Location getLocation() {
    return location;
  }

  public RuleInfo.Builder getRuleInfo() {
    return ruleInfo;
  }
}
