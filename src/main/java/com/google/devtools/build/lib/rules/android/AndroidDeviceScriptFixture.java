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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;

/** An implementation of the {@code android_device_script_fixture} rule. */
public class AndroidDeviceScriptFixture implements RuleConfiguredTargetFactory {

  private final AndroidSemantics androidSemantics;

  protected AndroidDeviceScriptFixture(AndroidSemantics androidSemantics) {
    this.androidSemantics = androidSemantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    Artifact fixtureScript = getFixtureScript(ruleContext);
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>stableOrder().add(fixtureScript).build())
        .addProvider(
            RunfilesProvider.class,
            RunfilesProvider.simple(
                new Runfiles.Builder(ruleContext.getWorkspaceName())
                    .addArtifact(fixtureScript)
                    .build()))
        .addNativeDeclaredProvider(
            new AndroidDeviceScriptFixtureInfoProvider(
                fixtureScript,
                AndroidCommon.getSupportApks(ruleContext),
                ruleContext.attributes().get("daemon", Type.BOOLEAN),
                ruleContext.attributes().get("strict_exit", Type.BOOLEAN)))
        .build();
  }

  /**
   * Gets the fixture script from the {@code script} attribute if set or else writes a file
   * containing the contents of the {@code cmd} attribute. Also, checks that exactly one of {@code
   * script} and {@code cmd} is set.
   */
  private static Artifact getFixtureScript(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    String cmd = null;
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("cmd")) {
      cmd = ruleContext.attributes().get("cmd", Type.STRING);
    }
    TransitiveInfoCollection script = ruleContext.getPrerequisite("script", TransitionMode.TARGET);

    if (((cmd == null) && (script == null)) || ((cmd != null) && (script != null))) {
      ruleContext.throwWithRuleError(
          "android_host_service_fixture requires that exactly one of the script and cmd attributes "
              + "be specified");
    }

    if (cmd == null) {
      // The fact that there is only one file and that it has the right extension is enforced by the
      // rule definition.
      return script.getProvider(FileProvider.class).getFilesToBuild().getSingleton();
    } else {
      return writeFixtureScript(ruleContext, cmd);
    }
  }

  private static Artifact writeFixtureScript(RuleContext ruleContext, String cmd)
      throws InterruptedException {
    Artifact output =
        ruleContext.getUniqueDirectoryArtifact(
            "cmd_device_fixtures", "cmd.sh", ruleContext.getBinOrGenfilesDirectory());
    ruleContext.registerAction(FileWriteAction.create(ruleContext, output, cmd, false));
    return output;
  }
}
