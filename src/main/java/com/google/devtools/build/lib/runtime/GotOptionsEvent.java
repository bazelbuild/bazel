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
package com.google.devtools.build.lib.runtime;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.common.options.OptionsParser.UnparsedOptionValueDescription;
import com.google.devtools.common.options.OptionsProvider;
import java.util.Collection;
import java.util.Objects;

/** An event in which the command line options are discovered. */
public class GotOptionsEvent implements BuildEventWithOrderConstraint {

  private final OptionsProvider startupOptions;
  private final OptionsProvider options;
  private final InvocationPolicy invocationPolicy;

  /**
   * Construct the options event.
   *
   * @param startupOptions the parsed startup options
   * @param options the parsed options
   */
  public GotOptionsEvent(
      OptionsProvider startupOptions, OptionsProvider options, InvocationPolicy invocationPolicy) {
    this.startupOptions = startupOptions;
    this.options = options;
    this.invocationPolicy = invocationPolicy;
  }

  /**
   * @return the parsed startup options
   */
  public OptionsProvider getStartupOptions() {
    return startupOptions;
  }

  /**
   * @return the parsed options.
   */
  public OptionsProvider getOptions() {
    return options;
  }

  /** @return the invocation policy. */
  public InvocationPolicy getInvocationPolicy() {
    return invocationPolicy;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.optionsParsedId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    BuildEventStreamProtos.OptionsParsed.Builder optionsBuilder =
        BuildEventStreamProtos.OptionsParsed.newBuilder();

    OptionsProvider options = getStartupOptions();
    optionsBuilder.addAllStartupOptions(OptionsUtils.asArgumentList(options));
    optionsBuilder.addAllExplicitStartupOptions(
        OptionsUtils.asArgumentList(
            Iterables.filter(
                options.asListOfExplicitOptions(),
                new Predicate<UnparsedOptionValueDescription>() {
                  @Override
                  public boolean apply(UnparsedOptionValueDescription input) {
                    return !Objects.equals(input.getSource(), "default");
                  }
                })));
    options = getOptions();
    optionsBuilder.addAllCmdLine(OptionsUtils.asArgumentList(options));
    optionsBuilder.addAllExplicitCmdLine(
        OptionsUtils.asArgumentList(
            Iterables.filter(
                options.asListOfExplicitOptions(),
                new Predicate<UnparsedOptionValueDescription>() {
                  @Override
                  public boolean apply(UnparsedOptionValueDescription input) {
                    // Source can be null coming from the OptionParser.
                    return Objects.equals(input.getSource(), "command line options");
                  }
                })));

    return GenericBuildEvent.protoChaining(this).setOptionsParsed(optionsBuilder.build()).build();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventId.buildStartedId(), BuildEventId.commandlineId());
  }
}
