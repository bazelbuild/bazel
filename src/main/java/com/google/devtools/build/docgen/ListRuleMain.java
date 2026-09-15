// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;

/**
 * Prints out list of known rules.
 */
public class ListRuleMain {

  private static ConfiguredRuleClassProvider createRuleClassProvider(String classProvider)
      throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException,
          IllegalAccessException {
    Class<?> providerClass = Class.forName(classProvider);
    Method createMethod = providerClass.getMethod("create");
    return (ConfiguredRuleClassProvider) createMethod.invoke(null);
  }

  public static void main(String[] args) throws Exception {
    if (args.length == 0) {
      System.err.println(
          "Expected one input parameter, please provide the name of the rule class provider");
    }

    RuleClassProvider provider = createRuleClassProvider(args[0]);
    Map<String, RuleClass> rcMap = provider.getRuleClassMap();
    for (String name : rcMap.keySet()) {
      System.out.println(name);
    }
  }
}
