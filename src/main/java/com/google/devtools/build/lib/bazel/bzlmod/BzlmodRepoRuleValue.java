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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/** The result of {@link BzlmodRepoRuleFunction}, holding a repository rule instance. */
@AutoCodec(explicitlyAllowClass = {Package.class})
public class BzlmodRepoRuleValue implements SkyValue {
  public static final SkyFunctionName BZLMOD_REPO_RULE =
      SkyFunctionName.createHermetic("BZLMOD_REPO_RULE");

  private final Package pkg;
  private final String ruleName;

  public BzlmodRepoRuleValue(Package pkg, String ruleName) {
    this.pkg = pkg;
    this.ruleName = ruleName;
  }

  public Rule getRule() {
    return pkg.getRule(ruleName);
  }

  public static Key key(String repositoryName) {
    return Key.create(repositoryName);
  }

  /** Represents an unsuccessful repository lookup. */
  public static final class RepoRuleNotFoundValue extends BzlmodRepoRuleValue {
    private RepoRuleNotFoundValue() {
      super(/*pkg=*/ null, /*ruleName=*/ null);
    }

    @Override
    public Rule getRule() {
      throw new IllegalStateException();
    }
  }

  public static final RepoRuleNotFoundValue REPO_RULE_NOT_FOUND_VALUE = new RepoRuleNotFoundValue();

  /** Argument for the SkyKey to request a BzlmodRepoRuleValue. */
  @AutoCodec
  public static class Key extends AbstractSkyKey<String> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(String arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(String arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return BZLMOD_REPO_RULE;
    }
  }
}
