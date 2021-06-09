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

package com.google.devtools.common.options;

import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.util.Arrays;
import java.util.List;

/**
 * Interface for parsing options from a single options specification class.
 *
 * <p>The {@link Options#parse(Class, String...)} method in this class has no clear use case.
 * Instead, use the {@link OptionsParser} class directly, as in this code snippet:
 *
 * <pre>
 * OptionsParser parser = OptionsParser.builder()
 *     .optionsClasses(FooOptions.class)
 *     .build();
 * try {
 *   parser.parse(FooOptions.class, args);
 * } catch (OptionsParsingException e) {
 *   System.err.print("Error parsing options: " + e.getMessage());
 *   System.err.print(options.getUsage());
 *   System.exit(1);
 * }
 * FooOptions foo = parser.getOptions(FooOptions.class);
 * List&lt;String&gt; otherArguments = parser.getResidue();
 * </pre>
 *
 * Using this class in this case actually results in more code.
 *
 * @see OptionsParser for parsing options from multiple options specification classes.
 */
public class Options<O extends OptionsBase> {

  /**
   * Parse the options provided in args, given the specification in
   * optionsClass.
   */
  public static <O extends OptionsBase> Options<O> parse(Class<O> optionsClass, String... args)
      throws OptionsParsingException {
    OptionsParser parser = OptionsParser.builder().optionsClasses(optionsClass).build();
    parser.parse(OptionPriority.PriorityCategory.COMMAND_LINE, null, Arrays.asList(args));
    List<String> remainingArgs = parser.getResidue();
    return new Options<>(parser.getOptions(optionsClass), remainingArgs.toArray(new String[0]));
  }

  /**
   * A convenience function for use in main methods. Parses the command line parameters, and exits
   * upon error. Also, prints out the usage message if "--help" appears anywhere within {@code
   * args}.
   */
  public static <O extends OptionsBase> Options<O> parseAndExitUponError(
      Class<O> optionsClass, boolean allowResidue, String... args) {
    OptionsParser parser = null;
    try {
      parser =
          OptionsParser.builder().optionsClasses(optionsClass).allowResidue(allowResidue).build();
    } catch (ConstructionException e) {
      System.err.println("Error constructing the options parser: " + e.getMessage());
      System.exit(2);
    }
    parser.parseAndExitUponError(args);
    List<String> remainingArgs = parser.getResidue();
    return new Options<>(parser.getOptions(optionsClass), remainingArgs.toArray(new String[0]));
  }

  /**
   * Returns an options object at its default values.  The returned object may
   * be freely modified by the caller, by assigning its fields.
   */
  public static <O extends OptionsBase> O getDefaults(Class<O> optionsClass) {
    try {
      return parse(optionsClass, new String[0]).getOptions();
    } catch (OptionsParsingException e) {
      String message = "Error while parsing defaults: " + e.getMessage();
      throw new AssertionError(message);
    }
  }

  /**
   * Returns a usage string (renders the help information, the defaults, and
   * of course the option names).
   */
  public static String getUsage(Class<? extends OptionsBase> optionsClass) {
    StringBuilder usage = new StringBuilder();
    OptionsUsage.getUsage(optionsClass, usage);
    return usage.toString();
  }

  private final O options;
  private final String[] remainingArgs;

  private Options(O options, String[] remainingArgs) {
    this.options = options;
    this.remainingArgs = remainingArgs;
  }

  /**
   * Returns an instance of options class O.
   */
  public O getOptions() {
    return options;
  }

  /**
   * Returns the arguments that we didn't parse.
   */
  public String[] getRemainingArgs() {
    return remainingArgs;
  }

}
