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

package com.google.devtools.build.skydoc.fakebuildapi.java;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaProtoCommonApi;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * Fake implementation of {@link JavaProtoCommonApi}.
 */
public class FakeJavaProtoCommon
    implements JavaProtoCommonApi<FileApi, SkylarkRuleContextApi, TransitiveInfoCollectionApi> {

  @Override
  public void createProtoCompileAction(SkylarkRuleContextApi skylarkRuleContext,
      TransitiveInfoCollectionApi target, FileApi sourceJar, String protoToolchainAttr,
      String flavour) throws EvalException {}

  @Override
  public boolean hasProtoSources(TransitiveInfoCollectionApi target) {
    return false;
  }

  @Override
  public JavaInfoApi<FileApi> getRuntimeToolchainProvider(SkylarkRuleContextApi skylarkRuleContext,
      String protoToolchainAttr) throws EvalException {
    return new FakeJavaInfo();
  }
}
