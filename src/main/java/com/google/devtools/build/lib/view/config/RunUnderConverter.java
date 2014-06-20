// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view.config;

import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * --run_under options converter.
 */
public class RunUnderConverter implements Converter<RunUnder> {
  @Override
  public RunUnder convert(final String input) throws OptionsParsingException {
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
    if (runUnderCommand.startsWith("//")) {
      try {
        final Label runUnderLable = Label.parseAbsolute(runUnderCommand);
        return new RunUnder() {
          @Override public String getValue() { return input; }
          @Override public Label getLabel() { return runUnderLable; }
          @Override public String getCommand() { return null; }
          @Override public List<String> getOptions() {
            return Collections.unmodifiableList(runUnderList.subList(1, runUnderList.size()));
          }
          @Override public String toString() { return input; }
        };
      } catch (SyntaxException e) {
        throw new OptionsParsingException("Not a valid label " + e.getMessage());
      }
    } else {
      return new RunUnder() {
        @Override public String getValue() { return input; }
        @Override public Label getLabel() { return null; }
        @Override public String getCommand() { return runUnderCommand; }
        @Override public List<String> getOptions() {
          return Collections.unmodifiableList(runUnderList.subList(1, runUnderList.size()));
        }
        @Override public String toString() { return input; }
      };
    }
  }

  @Override
  public String getTypeDescription() {
    return "a prefix in front of command";
  }
}
