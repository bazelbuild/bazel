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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import java.util.Map;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * An interface for {@link RepositoryRuleFunction} to create a repository rule instance with given
 * parameters.
 */
public interface BzlmodRepoRuleCreator {
  Rule createAndAddRule(
      Package.Builder packageBuilder,
      StarlarkSemantics semantics,
      Map<String, Object> kwargs,
      EventHandler handler)
      throws InterruptedException, InvalidRuleException, NameConflictException;
}
