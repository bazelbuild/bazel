// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import com.google.common.collect.ImmutableSet;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import org.junit.Before;
import org.junit.BeforeClass;

/** Useful setup for testing InvocationPolicy. */
public class InvocationPolicyEnforcerTestBase {

  /** Test converter that splits a string by commas to produce a list. */
  public static class ToListConverter extends Converter.Contextless<List<String>> {

    public ToListConverter() {}

    @Override
    public List<String> convert(String input) throws OptionsParsingException {
      return Arrays.asList(input.split(","));
    }

    @Override
    public String getTypeDescription() {
      return "a list of strings";
    }
  }

  public static InvocationPolicyEnforcer createOptionsPolicyEnforcer(
      InvocationPolicy.Builder invocationPolicyBuilder) throws Exception {
    InvocationPolicy policyProto = invocationPolicyBuilder.build();

    // An OptionsPolicyEnforcer could be constructed in the test directly from the InvocationPolicy
    // proto, however Blaze will actually take the policy as another flag with a Base64 encoded
    // binary proto and parse that, so exercise that code path in the test.

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    policyProto.writeTo(out);
    String policyBase64 = BaseEncoding.base64().encode(out.toByteArray());

    OptionsParser startupOptionsParser =
        OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build();
    String policyOption = "--invocation_policy=" + policyBase64;
    startupOptionsParser.parse(policyOption);

    return new InvocationPolicyEnforcer(
        InvocationPolicyParser.parsePolicy(
            startupOptionsParser.getOptions(BlazeServerStartupOptions.class).invocationPolicy),
        Level.INFO,
        /*conversionContext=*/ null);
  }

  OptionsParser parser;

  @Before
  public final void setParser() throws Exception  {
    parser = OptionsParser.builder().optionsClasses(TestOptions.class).build();
  }

  @BeforeClass
  public static void setCommandNameCache() throws Exception {
    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(
        new CommandNameCache() {
          @Override
          public ImmutableSet<String> get(String commandName) {
            return ImmutableSet.of(commandName);
          }
        });
  }

  TestOptions getTestOptions() {
    return parser.getOptions(TestOptions.class);
  }
}
