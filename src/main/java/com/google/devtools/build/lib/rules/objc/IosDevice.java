// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * Implementation for the "ios_device" rule.
 */
public final class IosDevice implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext context) throws InterruptedException {
    IosDeviceProvider provider = new IosDeviceProvider.Builder()
        .setType(context.attributes().get("type", STRING))
        .setIosVersion(context.attributes().get("ios_version", STRING))
        .setLocale(context.attributes().get("locale", STRING))
        .build();

    return new RuleConfiguredTargetBuilder(context)
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .add(IosDeviceProvider.class, provider)
        .add(IosTestSubstitutionProvider.class, provider.iosTestSubstitutionProvider())
        .build();
  }
}
