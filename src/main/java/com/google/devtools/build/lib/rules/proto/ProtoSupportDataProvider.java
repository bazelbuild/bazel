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

package com.google.devtools.build.lib.rules.proto;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provider that returns SupportData from proto_library used by language-specific protobuf
 * generators.
 */
@Immutable
public final class ProtoSupportDataProvider implements TransitiveInfoProvider {

  private final SupportData supportData;

  public ProtoSupportDataProvider(SupportData supportData) {
    this.supportData = supportData;
  }

  /**
   * Returns supportData created inside of ProtoLibrary.
   */
  public SupportData getSupportData() {
    return supportData;
  }
}
