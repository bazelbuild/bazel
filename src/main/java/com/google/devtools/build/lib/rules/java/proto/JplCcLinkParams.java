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

package com.google.devtools.build.lib.rules.java.proto;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.java.JavaCcInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import java.util.ArrayList;
import java.util.List;

/** Methods that all java_xxx_proto_library rules use to construct JavaCcLinkParamsProvider's. */
public class JplCcLinkParams {

  /**
   * Creates a CcLinkingInfo based on 'deps' and an explicit list of proto runtimes, in the context
   * of a java_xxx_proto_library and its aspects.
   *
   * @param ruleContext used to extract 'deps'. the 'deps' are expected to provide
   *     JavaProtoLibraryAspectProvider, which is the case when a java_xxx_proto_library rule
   *     depends on proto_library's with the aspect, and when an aspect node depends on its
   *     dependency's aspect node.
   * @param protoRuntimes a list of java_library.
   */
  public static JavaCcInfoProvider createCcLinkingInfo(
      final RuleContext ruleContext, final ImmutableList<TransitiveInfoCollection> protoRuntimes) {
    List<CcInfo> providers = new ArrayList<>();
    for (TransitiveInfoCollection t : ruleContext.getPrerequisites("deps")) {
      JavaCcInfoProvider javaCcInfoProvider = JavaInfo.getProvider(JavaCcInfoProvider.class, t);
      if (javaCcInfoProvider != null) {
        providers.add(javaCcInfoProvider.getCcInfo());
      }
    }

    for (TransitiveInfoCollection t : protoRuntimes) {
      CcInfo ccInfo = t.get(CcInfo.PROVIDER);
      if (ccInfo != null) {
        providers.add(ccInfo);
      }

      JavaCcInfoProvider javaCcInfoProvider = JavaInfo.getProvider(JavaCcInfoProvider.class, t);
      if (javaCcInfoProvider != null) {
        providers.add(javaCcInfoProvider.getCcInfo());
      }
    }

    return new JavaCcInfoProvider(CcInfo.merge(providers));
  }
}
