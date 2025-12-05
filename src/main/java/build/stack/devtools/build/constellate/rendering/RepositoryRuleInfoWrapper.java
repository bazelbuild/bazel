// Copyright 2021 The Bazel Authors. All rights reserved.
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

package build.stack.devtools.build.constellate.rendering;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;

/** Stores information about a Starlark repository rule definition. */
public class RepositoryRuleInfoWrapper {
  private final StarlarkCallable identifierFunction;
  private final Location location;
  private final RepositoryRuleInfo.Builder repositoryRuleInfo;

  public RepositoryRuleInfoWrapper(
      StarlarkCallable identifierFunction,
      Location location,
      RepositoryRuleInfo.Builder repositoryRuleInfo) {
    this.identifierFunction = identifierFunction;
    this.location = location;
    this.repositoryRuleInfo = repositoryRuleInfo;
  }

  public StarlarkCallable getIdentifierFunction() {
    return identifierFunction;
  }

  public Location getLocation() {
    return location;
  }

  public RepositoryRuleInfo.Builder getRepositoryRuleInfo() {
    return repositoryRuleInfo;
  }
}
