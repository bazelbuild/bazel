// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import java.util.Collection;

/**
 * Stores information about a starlark provider definition, comprised of BaseFunction identifier and
 * a {@link ProviderInfo} proto.
 *
 * <p>For example, in
 *
 * <pre>FooInfo = provider(doc = 'My provider', fields = {'bar' : 'a bar'})</pre>
 *
 * , this contains all information about the definition of FooInfo for purposes of generating its
 * documentation, as well as a unique BaseFunction identifier.
 */
public class ProviderInfoWrapper {

  private final BaseFunction identifier;
  // Only the Builder is passed to ProviderInfoWrapper as the provider name is not yet available.
  private final ProviderInfo.Builder providerInfo;

  public ProviderInfoWrapper(
      BaseFunction identifier, String docString, Collection<ProviderFieldInfo> fieldInfos) {
    this.identifier = identifier;
    this.providerInfo =
        ProviderInfo.newBuilder().setDocString(docString).addAllFieldInfo(fieldInfos);
  }

  public BaseFunction getIdentifier() {
    return identifier;
  }

  public ProviderInfo.Builder getProviderInfo() {
    return providerInfo;
  }
}
