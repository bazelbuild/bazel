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
package com.google.devtools.build.lib.testutil;

import com.google.common.base.Function;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystem;

import java.util.Map;

class PackageFactoryFactoryForBazelUnitTests extends PackageFactory.FactoryForTesting {
  static final PackageFactoryFactoryForBazelUnitTests INSTANCE =
      new PackageFactoryFactoryForBazelUnitTests();

  private PackageFactoryFactoryForBazelUnitTests() {
  }

  @Override
  protected PackageFactory create(
      RuleClassProvider ruleClassProvider,
      Map<String, String> platformSetRegexps,
      Function<RuleClass, AttributeContainer> attributeContainerFactory,
      Iterable<EnvironmentExtension> environmentExtensions,
      String version,
      FileSystem fs) {
    return new PackageFactory(
        ruleClassProvider,
        platformSetRegexps,
        attributeContainerFactory,
        environmentExtensions,
        version,
        Package.Builder.DefaultHelper.INSTANCE);
  }
}

