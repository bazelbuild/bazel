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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Command line args module. */
@SkylarkModule(
    name = "Args",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "An object that encapsulates, in a memory-efficient way, the data needed to build part or "
            + "all of a command line."
            + ""
            + "<p>It often happens that an action requires a large command line containing values "
            + "accumulated from transitive dependencies. For example, a linker command line might "
            + "list every object file needed by all of the libraries being linked. It is best "
            + "practice to store such transitive data in <a href='depset.html'><code>depset"
            + "</code></a>s, so that they can be shared by multiple targets. However, if the rule "
            + "author had to convert these depsets into lists of strings in order to construct an "
            + "action command line, it would defeat this memory-sharing optimization."
            + ""
            + "<p>For this reason, the action-constructing functions accept <code>Args</code> "
            + "objects in addition to strings. Each <code>Args</code> object represents a "
            + "concatenation of strings and depsets, with optional transformations for "
            + "manipulating the data. <code>Args</code> objects do not process the depsets they "
            + "encapsulate until the execution phase, when it comes time to calculate the command "
            + "line. This helps defer any expensive copying until after the analysis phase is "
            + "complete. See the <a href='../performance.$DOC_EXT'>Optimizing Performance</a> page "
            + "for more information."
            + ""
            + "<p><code>Args</code> are constructed by calling <a href='actions.html#args'><code>"
            + "ctx.actions.args()</code></a>. They can be passed as the <code>arguments</code> "
            + "parameter of <a href='actions.html#run'><code>ctx.actions.run()</code></a> or "
            + "<a href='actions.html#run_shell'><code>ctx.actions.run_shell()</code></a>. Each "
            + "mutation of an <code>Args</code> object appends values to the eventual command "
            + "line."
            + ""
            + "<p>The <code>map_each</code> feature allows you to customize how items are "
            + "transformed into strings. If you do not provide a <code>map_each</code> function, "
            + "the standard conversion is as follows: "
            + "<ul>"
            + "<li>Values that are already strings are left as-is."
            + "<li><a href='File.html'><code>File</code></a> objects are turned into their "
            + "    <code>File.path</code> values."
            + "<li>All other types are turned into strings in an <i>unspecified</i> manner. For "
            + "    this reason, you should avoid passing values that are not of string or "
            + "    <code>File</code> type to <code>add()</code>, and if you pass them to "
            + "    <code>add_all()</code> or <code>add_joined()</code> then you should provide a "
            + "    <code>map_each</code> function."
            + "</ul>"
            + ""
            + "<p>When using string formatting (<code>format</code>, <code>format_each</code>, and "
            + "<code>format_joined</code> params of the <code>add*()</code> methods), the format "
            + "template is interpreted in the same way as <code>%</code>-substitution on strings, "
            + "except that the template must have exactly one substitution placeholder and it must "
            + "be <code>%s</code>. Literal percents may be escaped as <code>%%</code>. Formatting "
            + "is applied after the value is converted to a string as per the above."
            + ""
            + "<p>Each of the <code>add*()</code> methods have an alternate form that accepts an "
            + "extra positional parameter, an \"arg name\" string to insert before the rest of the "
            + "arguments. For <code>add_all</code> and <code>add_joined</code> the extra string "
            + "will not be added if the sequence turns out to be empty. "
            + "For instance, the same usage can add either <code>--foo val1 val2 val3 --bar"
            + "</code> or just <code>--bar</code> to the command line, depending on whether the "
            + "given sequence contains <code>val1..val3</code> or is empty."
            + ""
            + "<p>If the size of the command line can grow longer than the maximum size allowed by "
            + "the system, the arguments can be spilled over into parameter files. See "
            + "<a href='#use_param_file'><code>use_param_file()</code></a> and "
            + "<a href='#set_param_file_format'><code>set_param_file_format()</code></a>."
            + ""
            + "<p>Example: Suppose we wanted to generate the command line: "
            + "<pre>\n"
            + "--foo foo1.txt foo2.txt ... fooN.txt --bar bar1.txt,bar2.txt,...,barM.txt --baz\n"
            + "</pre>"
            + "We could use the following <code>Args</code> object: "
            + "<pre class=language-python>\n"
            + "# foo_deps and bar_deps are depsets containing\n"
            + "# File objects for the foo and bar .txt files.\n"
            + "args = ctx.actions.args()\n"
            + "args.add_all(\"--foo\", foo_deps)\n"
            + "args.add_joined(\"--bar\", bar_deps, join_with=\",\")\n"
            + "args.add(\"--baz\")\n"
            + "ctx.actions.run(\n"
            + "  ...\n"
            + "  arguments = [args],\n"
            + "  ...\n"
            + ")\n"
            + "</pre>")
public interface CommandLineArgsApi extends StarlarkValue {
  @SkylarkCallable(
      name = "add",
      doc =
          "Appends an argument to this command line."
              + ""
              + "<p><b>Deprecation note:</b> The <code>before_each</code>, <code>join_with</code> "
              + "and <code>map_fn</code> params are replaced by the <a href='#add_all'><code>"
              + "add_all()</code></a> and <a href='#add_joined'><code>add_joined()</code></a> "
              + "methods. These parameters will be removed, and are currently disallowed if the "
              + "<a href='../backward-compatibility.$DOC_EXT#new-args-api'><code>"
              + "--incompatible_disallow_old_style_args_add</code></a> flag is set. Likewise, "
              + "<code>value</code> should now be a scalar value, not a list, tuple, or depset of "
              + "items.",
      parameters = {
        @Param(
            name = "arg_name_or_value",
            doc =
                "If two positional parameters are passed this is interpreted as the arg name. "
                    + "The arg name is added before the value without any processing. "
                    + "If only one positional parameter is passed, it is interpreted as "
                    + "<code>value</code> (see below)."),
        @Param(
            name = "value",
            defaultValue = "unbound",
            doc =
                "The object to append. It will be converted to a string using the standard "
                    + "conversion mentioned above. Since there is no <code>map_each</code> "
                    + "parameter for this function, <code>value</code> should be either a "
                    + "string or a <code>File</code>. A directory <code>File</code> must be "
                    + "passed to <a href='#add_all'><code>add_all()</code> or "
                    + "<a href='#add_joined'><code>add_joined()</code></a> instead of this method."
                    + "<p><i>Deprecated behavior:</i> <code>value</code> may also be a "
                    + "list, tuple, or depset of multiple items to append."),
        @Param(
            name = "format",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "A format string pattern, to be applied to the stringified version of <code>value"
                    + "</code>."
                    + ""
                    + "<p><i>Deprecated behavior:</i> If <code>value</code> is a list or depset, "
                    + "formatting is applied to each item."),
        @Param(
            name = "before_each",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "<i>Deprecated:</i> Only supported when <code>value</code> is a list, tuple, or "
                    + "depset. This string will be appended prior to appending each item."),
        @Param(
            name = "join_with",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "<i>Deprecated:</i> Only supported when <code>value</code> is a list, tuple, or "
                    + "depset. All items will be joined together using this string to form a "
                    + "single arg to append."),
        @Param(
            name = "map_fn",
            type = StarlarkCallable.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "<i>Deprecated:</i> Only supported when <code>value</code> is a list, tuple, or "
                    + "depset. This is a function that transforms the sequence of items into a "
                    + "list of strings. The sequence of items is given as a positional argument -- "
                    + "the function must not take any other parameters -- and the returned "
                    + "list's length must equal the number of items. Use <code>map_each</code> "
                    + "of <code>add_all</code> or <code>add_joined</code> instead.")
      },
      useStarlarkThread = true)
  CommandLineArgsApi addArgument(
      Object argNameOrValue,
      Object value,
      Object format,
      Object beforeEach,
      Object joinWith,
      Object mapFn,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "add_all",
      doc =
          "Appends multiple arguments to this command line. For depsets, the items are "
              + "evaluated lazily during the execution phase."
              + ""
              + "<p>Most of the processing occurs over a list of arguments to be appended, as per "
              + "the following steps:"
              + "<ol>"
              + "<li>Each directory <code>File</code> item is replaced by all <code>File</code>s "
              + "recursively contained in that directory."
              + "</li>"
              + "<li>If <code>map_each</code> is given, it is applied to each item, and the "
              + "    resulting lists of strings are concatenated to form the initial argument "
              + "    list. Otherwise, the initial argument list is the result of applying the "
              + "    standard conversion to each item."
              + "<li>Each argument in the list is formatted with <code>format_each</code>, if "
              + "    present."
              + "<li>If <code>uniquify</code> is true, duplicate arguments are removed. The first "
              + "    occurrence is the one that remains."
              + "<li>If a <code>before_each</code> string is given, it is inserted as a new "
              + "    argument before each existing argument in the list. This effectively doubles "
              + "    the number of arguments to be appended by this point."
              + "<li>Except in the case that the list is empty and <code>omit_if_empty</code> is "
              + "    true (the default), the arg name and <code>terminate_with</code> are "
              + "    inserted as the first and last arguments, respectively, if they are given."
              + "</ol>"
              + "Note that empty strings are valid arguments that are subject to all these "
              + "processing steps.",
      parameters = {
        @Param(
            name = "arg_name_or_values",
            doc =
                "If two positional parameters are passed this is interpreted as the arg name. "
                    + "The arg name is added before the <code>values</code> without any "
                    + "processing. This arg name will not be added if <code>omit_if_empty</code> "
                    + "is true (the default) and no other items are appended (as happens if "
                    + "<code>values</code> is empty or all of its items are filtered). "
                    + "If only one positional parameter is passed, it is interpreted as "
                    + "<code>values</code> (see below)."),
        @Param(
            name = "values",
            allowedTypes = {
              @ParamType(type = Sequence.class),
              @ParamType(type = Depset.class),
            },
            defaultValue = "unbound",
            doc = "The list, tuple, or depset whose items will be appended."),
        @Param(
            name = "map_each",
            type = StarlarkCallable.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "A function that converts each item to zero or more strings, which may be further "
                    + "processed before appending. If this param is not provided, the standard "
                    + "conversion is used."
                    + ""
                    + "<p>The function takes in the item as a positional parameter and must have "
                    + "no other parameters. The return value's type depends on how many arguments "
                    + "are to be produced for the item:"
                    + "<ul>"
                    + "<li>In the common case when each item turns into one string, the function "
                    + "    should return that string."
                    + "<li>If the item is to be filtered out entirely, the function should return "
                    + "    <code>None</code>."
                    + "<li>If the item turns into multiple strings, the function returns a list of "
                    + "    those strings."
                    + "</ul>"
                    + "Returning a single string or <code>None</code> has the same effect as "
                    + "returning a list of length 1 or length 0 respectively. However, it is more "
                    + "efficient and readable to avoid creating a list where it is not needed."
                    + ""
                    + "<p><i>Warning:</i> <a href='globals.html#print'><code>print()</code></a> "
                    + "statements that are executed during the call to <code>map_each</code> will "
                    + "not produce any visible output."),
        @Param(
            name = "format_each",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "An optional format string pattern, applied to each string returned by the "
                    + "<code>map_each</code> function. "
                    + "The format string must have exactly one '%s' placeholder."),
        @Param(
            name = "before_each",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "An optional string to append before each argument derived from "
                    + "<code>values</code> is appended."),
        @Param(
            name = "omit_if_empty",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "True",
            doc =
                "If true, if there are no arguments derived from <code>values</code> to be "
                    + "appended, then all further processing is suppressed and the command line "
                    + "will be unchanged. If false, the arg name and <code>terminate_with</code>, "
                    + "if provided, will still be appended regardless of whether or not there are "
                    + "other arguments."),
        @Param(
            name = "uniquify",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "False",
            doc =
                "If true, duplicate arguments that are derived from <code>values</code> will be "
                    + "omitted. Only the first occurrence of each argument will remain. Usually "
                    + "this feature is not needed because depsets already omit duplicates, "
                    + "but it can be useful if <code>map_each</code> emits the same string for "
                    + "multiple items."),
        @Param(
            name = "expand_directories",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "True",
            doc =
                "If true, any directories in <code>values</code> will be expanded to a flat list "
                    + "of files. This happens before <code>map_each</code> is applied."),
        @Param(
            name = "terminate_with",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "An optional string to append after all other arguments. This string will not be "
                    + "added if <code>omit_if_empty</code> is true (the default) and no other "
                    + "items are appended (as happens if <code>values</code> is empty or all of "
                    + "its items are filtered)."),
      },
      useStarlarkThread = true)
  CommandLineArgsApi addAll(
      Object argNameOrValue,
      Object values,
      Object mapEach,
      Object formatEach,
      Object beforeEach,
      Boolean omitIfEmpty,
      Boolean uniquify,
      Boolean expandDirectories,
      Object terminateWith,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "add_joined",
      doc =
          "Appends an argument to this command line by concatenating together multiple values "
              + "using a separator. For depsets, the items are evaluated lazily during the "
              + "execution phase."
              + ""
              + "<p>Processing is similar to <a href='#add_all'><code>add_all()</code></a>, but "
              + "the list of arguments derived from <code>values</code> is combined into a single "
              + "argument as if by <code>join_with.join(...)</code>, and then formatted using the "
              + "given <code>format_joined</code> string template. Unlike <code>add_all()</code>, "
              + "there is no <code>before_each</code> or <code>terminate_with</code> parameter "
              + "since these are not generally useful when the items are combined into a single "
              + "argument."
              + ""
              + "<p>If after filtering there are no strings to join into an argument, and if "
              + "<code>omit_if_empty</code> is true (the default), no processing is done. "
              + "Otherwise if there are no strings to join but <code>omit_if_empty</code> is "
              + "false, the joined string will be an empty string.",
      parameters = {
        @Param(
            name = "arg_name_or_values",
            doc =
                "If two positional parameters are passed this is interpreted as the arg name. "
                    + "The arg name is added before <code>values</code> without any processing. "
                    + "This arg will not be added if <code>omit_if_empty</code> is true "
                    + "(the default) and there are no strings derived from <code>values</code> "
                    + "to join together (which can happen if <code>values</code> is empty "
                    + "or all of its items are filtered)."
                    + "If only one positional parameter is passed, it is interpreted as "
                    + "<code>values</code> (see below)."),
        @Param(
            name = "values",
            allowedTypes = {
              @ParamType(type = Sequence.class),
              @ParamType(type = Depset.class),
            },
            defaultValue = "unbound",
            doc = "The list, tuple, or depset whose items will be joined."),
        @Param(
            name = "join_with",
            type = String.class,
            named = true,
            positional = false,
            doc =
                "A delimiter string used to join together the strings obtained from applying "
                    + "<code>map_each</code> and <code>format_each</code>, in the same manner as "
                    + "<a href='string.html#join'><code>string.join()</code></a>."),
        @Param(
            name = "map_each",
            type = StarlarkCallable.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc = "Same as for <a href='#add_all.map_each'><code>add_all</code></a>."),
        @Param(
            name = "format_each",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc = "Same as for <a href='#add_all.format_each'><code>add_all</code></a>."),
        @Param(
            name = "format_joined",
            type = String.class,
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true,
            doc =
                "An optional format string pattern applied to the joined string. "
                    + "The format string must have exactly one '%s' placeholder."),
        @Param(
            name = "omit_if_empty",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "True",
            doc =
                "If true, if there are no strings to join together (either because <code>values"
                    + "</code> is empty or all its items are filtered), then all further "
                    + "processing is suppressed and the command line will be unchanged. "
                    + "If false, then even if there are no strings to join together, two "
                    + "arguments will be appended: the arg name followed by an empty "
                    + "string (which is the logical join of zero strings)."),
        @Param(
            name = "uniquify",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "False",
            doc = "Same as for <a href='#add_all.uniquify'><code>add_all</code></a>."),
        @Param(
            name = "expand_directories",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "True",
            doc = "Same as for <a href='#add_all.expand_directories'><code>add_all</code></a>.")
      },
      useStarlarkThread = true)
  CommandLineArgsApi addJoined(
      Object argNameOrValue,
      Object values,
      String joinWith,
      Object mapEach,
      Object formatEach,
      Object formatJoined,
      Boolean omitIfEmpty,
      Boolean uniquify,
      Boolean expandDirectories,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "use_param_file",
      doc =
          "Spills the args to a params file, replacing them with a pointer to the param file. "
              + "Use when your args may be too large for the system's command length limits."
              + "<p>Bazel may choose to elide writing the params file to the output tree during "
              + "execution for efficiency."
              + "If you are debugging actions and want to inspect the param file, "
              + "pass <code>--materialize_param_files</code> to your build.",
      parameters = {
        @Param(
            name = "param_file_arg",
            type = String.class,
            named = true,
            doc =
                "A format string with a single \"%s\". "
                    + "If the args are spilled to a params file then they are replaced "
                    + "with an argument consisting of this string formatted with "
                    + "the path of the params file."),
        @Param(
            name = "use_always",
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "False",
            doc =
                "Whether to always spill the args to a params file. If false, "
                    + "bazel will decide whether the arguments need to be spilled "
                    + "based on your system and arg length.")
      })
  CommandLineArgsApi useParamsFile(String paramFileArg, Boolean useAlways) throws EvalException;

  @SkylarkCallable(
      name = "set_param_file_format",
      doc = "Sets the format of the param file when written to disk",
      parameters = {
        @Param(
            name = "format",
            type = String.class,
            named = true,
            doc =
                "The format of the param file. Must be one of:<ul><li>"
                    + "\"shell\": All arguments are shell quoted and separated by "
                    + "whitespace (space, tab, newline)</li><li>"
                    + "\"multiline\": All arguments are unquoted and separated by newline "
                    + "characters</li></ul>"
                    + "<p>The format defaults to \"shell\" if not called.")
      })
  CommandLineArgsApi setParamFileFormat(String format) throws EvalException;
}
