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
package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

/** --run_under options converter. */
public class RunUnderConverter implements Converter<RunUnder> {
  private static final LabelConverter LABEL_CONVERTER = new LabelConverter();

  @Override
  public RunUnder convert(final String input, Object conversionContext)
      throws OptionsParsingException {
    final List<String> runUnderList = new ArrayList<>();
    try {
      ShellUtils.tokenize(runUnderList, input);
    } catch (TokenizationException e) {
      throw new OptionsParsingException("Not a valid command prefix " + e.getMessage());
    }
    if (runUnderList.isEmpty()) {
      throw new OptionsParsingException("Empty command");
    }
    final String runUnderCommand = runUnderList.get(0);
    ImmutableList<String> runUnderSuffix =
        ImmutableList.copyOf(runUnderList.subList(1, runUnderList.size()));
    if (runUnderCommand.startsWith("//") || runUnderCommand.startsWith("@")) {
      try {
        final Label runUnderLabel = LABEL_CONVERTER.convert(runUnderCommand, conversionContext);
        return new RunUnderLabel(input, runUnderLabel, runUnderSuffix);
      } catch (OptionsParsingException e) {
        throw new OptionsParsingException("Not a valid label " + e.getMessage());
      }
    } else {
      return new RunUnderCommand(input, runUnderCommand, runUnderSuffix);
    }
  }

  private static final class RunUnderLabel implements RunUnder {
    private final String input;
    private final Label runUnderLabel;
    private final ImmutableList<String> runUnderList;

    RunUnderLabel(String input, Label runUnderLabel, ImmutableList<String> runUnderList) {
      this.input = input;
      this.runUnderLabel = runUnderLabel;
      this.runUnderList = runUnderList;
    }

    @Override
    public String getValue() {
      return input;
    }

    @Override
    public Label getLabel() {
      return runUnderLabel;
    }

    @Override
    @Nullable
    public String getCommand() {
      return null;
    }

    @Override
    public ImmutableList<String> getOptions() {
      return runUnderList;
    }

    @Override
    public String toString() {
      return input;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      } else if (other instanceof RunUnderLabel otherRunUnderLabel) {
        return Objects.equals(input, otherRunUnderLabel.input)
            && Objects.equals(runUnderLabel, otherRunUnderLabel.runUnderLabel)
            && Objects.equals(runUnderList, otherRunUnderLabel.runUnderList);
      } else {
        return false;
      }
    }

    @Override
    public int hashCode() {
      return Objects.hash(input, runUnderLabel, runUnderList);
    }
  }

  private static final class RunUnderCommand implements RunUnder {
    private final String input;
    private final String runUnderCommand;
    private final ImmutableList<String> runUnderList;

    RunUnderCommand(String input, String runUnderCommand, ImmutableList<String> runUnderList) {
      this.input = input;
      this.runUnderCommand = runUnderCommand;
      this.runUnderList = runUnderList;
    }

    @Override
    public String getValue() {
      return input;
    }

    @Override
    @Nullable
    public Label getLabel() {
      return null;
    }

    @Override
    public String getCommand() {
      return runUnderCommand;
    }

    @Override
    public ImmutableList<String> getOptions() {
      return runUnderList;
    }

    @Override
    public String toString() {
      return input;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      } else if (other instanceof RunUnderCommand otherRunUnderCommand) {
        return Objects.equals(input, otherRunUnderCommand.input)
            && Objects.equals(runUnderCommand, otherRunUnderCommand.runUnderCommand)
            && Objects.equals(runUnderList, otherRunUnderCommand.runUnderList);
      } else {
        return false;
      }
    }

    @Override
    public int hashCode() {
      return Objects.hash(input, runUnderCommand, runUnderList);
    }
  }
  @Override
  public String getTypeDescription() {
    return "a prefix in front of command";
  }
}
