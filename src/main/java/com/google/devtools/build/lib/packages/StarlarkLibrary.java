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
package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.packages.PackageFactory.getContext;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.DocumentMethods;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.ClassObject;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.lib.json.Json;
import net.starlark.java.syntax.Location;

/**
 * A library of pre-declared Bazel Starlark functions.
 *
 * <p>For functions pre-declared in a BUILD file, use {@link #BUILD}. For Bazel functions such as
 * {@code select} and {@code depset} that are pre-declared in all BUILD, .bzl, and WORKSPACE files,
 * use {@link #COMMON}. For functions pre-declared in every Starlark file, use {@link
 * Starlark#UNIVERSE}.
 */
public final class StarlarkLibrary {

  private StarlarkLibrary() {} // uninstantiable

  /**
   * A library of Starlark values (keyed by name) that are not part of core Starlark but are common
   * to all Bazel Starlark file environments (BUILD, .bzl, and WORKSPACE). Examples: depset, select,
   * json.
   */
  public static final ImmutableMap<String, Object> COMMON = initCommon();

  private static ImmutableMap<String, Object> initCommon() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new CommonLibrary());
    env.put("json", Json.INSTANCE);
    env.put("proto", Proto.INSTANCE);
    return env.build();
  }

  /** Proto defines the "proto" Starlark module of utilities for protocol message processing. */
  @StarlarkBuiltin(
      name = "ProtoModule", // not "proto", to avoid conflict with ctx.fragments.proto data type
      category = DocCategory.BUILTIN,
      doc = "A module for protocol message processing.")
  static final class Proto implements StarlarkValue {

    // Note: in due course this is likely to move to net.starlark.java.lib.proto.
    // Do not add functions that would not belong there!
    // Functions related to running the protocol compiler belong in proto_common.

    static final Proto INSTANCE = new Proto();

    @StarlarkMethod(
        name = "encode_text",
        doc =
            "Returns the struct argument's encoding as a text-format protocol message.\n"
                + "The data structure must be recursively composed of strings, ints, floats, or"
                + " bools, or structs, sequences, and dicts of these types.\n"
                + "<p>A struct is converted to a message. Fields are emitted in name order.\n"
                + "<p>A sequence (such as a list or tuple) is converted to a repeated field.\n"
                + "Its elements must not be sequences or dicts.\n"
                + "<p>A dict is converted to a repeated field of messages with fields named 'key'"
                + " and 'value'.\n"
                + "Entries are emitted in iteration (insertion) order.\n"
                + "The dict's keys must be strings, ints, or bools, and its values must not be"
                + " sequences or dicts.\n"
                + "Examples:<br><pre class=language-python>struct(field=123).to_proto()\n"
                + "# field: 123\n\n"
                + "struct(field=True).to_proto()\n"
                + "# field: true\n\n"
                + "struct(field=[1, 2, 3]).to_proto()\n"
                + "# field: 1\n"
                + "# field: 2\n"
                + "# field: 3\n\n"
                + "struct(field='text').to_proto()\n"
                + "# field: \"text\"\n\n"
                + "struct(field=struct(inner_field='text')).to_proto()\n"
                + "# field {\n"
                + "#   inner_field: \"text\"\n"
                + "# }\n\n"
                + "struct(field=[struct(inner_field=1), struct(inner_field=2)]).to_proto()\n"
                + "# field {\n"
                + "#   inner_field: 1\n"
                + "# }\n"
                + "# field {\n"
                + "#   inner_field: 2\n"
                + "# }\n\n"
                + "struct(field=struct(inner_field=struct(inner_inner_field='text'))).to_proto()\n"
                + "# field {\n"
                + "#    inner_field {\n"
                + "#     inner_inner_field: \"text\"\n"
                + "#   }\n"
                + "# }\n\n"
                + "struct(foo={4: 3, 2: 1}).to_proto()\n"
                + "# foo: {\n"
                + "#   key: 4\n"
                + "#   value: 3\n"
                + "# }\n"
                + "# foo: {\n"
                + "#   key: 2\n"
                + "#   value: 1\n"
                + "# }\n"
                + "</pre>",
        parameters = {@Param(name = "x")})
    public String encodeText(ClassObject x) throws EvalException {
      TextEncoder enc = new TextEncoder();
      enc.message(x);
      return enc.out.toString();
    }

    private static final class TextEncoder {

      private final StringBuilder out = new StringBuilder();
      private int indent = 0;

      // Encodes ClassObject x as a protocol message.
      private void message(ClassObject x) throws EvalException {
        // For determinism, sort fields.
        String[] fields = x.getFieldNames().toArray(new String[0]);
        Arrays.sort(fields);
        for (String field : fields) {
          try {
            field(field, x.getValue(field));
          } catch (EvalException ex) {
            throw Starlark.errorf("in %s field .%s: %s", Starlark.type(x), field, ex.getMessage());
          }
        }
      }

      // Encodes ClassObject field (name, v) as a message field
      // (a repeated field, if v is a dict or sequence.)
      private void field(String name, Object v) throws EvalException {
        // dict?
        if (v instanceof Dict) {
          Dict<?, ?> dict = (Dict) v;
          for (Map.Entry<?, ?> entry : dict.entrySet()) {
            Object key = entry.getKey();
            if (!(key instanceof String || key instanceof StarlarkInt)) {
              throw Starlark.errorf(
                  "invalid dict key: got %s, want int or string", Starlark.type(key));
            }
            emitLine(name, " {");
            indent++;
            fieldElement("key", key); // can't fail
            try {
              fieldElement("value", entry.getValue());
            } catch (EvalException ex) {
              throw Starlark.errorf(
                  "in value for dict key %s: %s", Starlark.repr(key), ex.getMessage());
            }
            indent--;
            emitLine("}");
          }
          return;
        }

        // list or tuple?
        if (v instanceof Sequence) {
          int i = 0;
          for (Object item : (Sequence) v) {
            try {
              fieldElement(name, item);
            } catch (EvalException ex) {
              throw Starlark.errorf("at %s index %d: %s", Starlark.type(v), i, ex.getMessage());
            }
            i++;
          }
          return;
        }

        // non-repeated field
        fieldElement(name, v);
      }

      // Emits field (name, v) as a message field, or one element of a repeated field.
      // v must be an int, float, string, bool, or ClassObject.
      private void fieldElement(String name, Object v) throws EvalException {
        if (v instanceof ClassObject) {
          emitLine(name, " {");
          indent++;
          message((ClassObject) v);
          indent--;
          emitLine("}");

        } else if (v instanceof String) {
          String s = (String) v;
          emitLine(
              name,
              ": \"",
              s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n"),
              "\"");

        } else if (v instanceof StarlarkInt || v instanceof StarlarkFloat || v instanceof Boolean) {
          emitLine(name, ": ", v.toString());

        } else {
          throw Starlark.errorf("got %s, want string, int, bool, or struct", Starlark.type(v));
        }
      }

      // Emits items on an indented line.
      private void emitLine(String... items) {
        for (int i = 0; i < indent; i++) {
          out.append("  ");
        }
        for (String item : items) {
          out.append(item);
        }
        out.append('\n');
      }
    }
  }

  @DocumentMethods
  private static final class CommonLibrary {

    @StarlarkMethod(
        name = "depset",
        doc =
            "Creates a <a href=\"depset.html\">depset</a>. The <code>direct</code> parameter is a"
                + " list of direct elements of the depset, and <code>transitive</code> parameter"
                + " is a list of depsets whose elements become indirect elements of the created"
                + " depset. The order in which elements are returned when the depset is converted"
                + " to a list is specified by the <code>order</code> parameter. See the <a"
                + " href=\"../depsets.md\">Depsets overview</a> for more information.\n" //
                + "<p>All"
                + " elements (direct and indirect) of a depset must be of the same type, as"
                + " obtained by the expression <code>type(x)</code>.\n" //
                + "<p>Because a hash-based set is used to eliminate duplicates during iteration,"
                + " all elements of a depset should be hashable. However, this invariant is not"
                + " currently checked consistently in all constructors. Use the"
                + " --incompatible_always_check_depset_elements flag to enable consistent"
                + " checking; this will be the default behavior in future releases;  see <a"
                + " href='https://github.com/bazelbuild/bazel/issues/10313'>Issue 10313</a>.\n" //
                + "<p>In addition, elements must currently be immutable, though this restriction"
                + " will be relaxed in future.\n" //
                + "<p> The order of the created depset should be <i>compatible</i> with the order"
                + " of its <code>transitive</code> depsets. <code>\"default\"</code> order is"
                + " compatible with any other order, all other orders are only compatible with"
                + " themselves.\n" //
                + "<p> Note on backward/forward compatibility. This function currently accepts a"
                + " positional <code>items</code> parameter. It is deprecated and will be removed"
                + " in the future, and after its removal <code>direct</code> will become a sole"
                + " positional parameter of the <code>depset</code> function. Thus, both of the"
                + " following calls are equivalent and future-proof:<br>\n" //
                + "<pre class=language-python>depset(['a', 'b'], transitive = [...])\n" //
                + "depset(direct = ['a', 'b'], transitive = [...])\n" //
                + "</pre>",
        parameters = {
          @Param(
              name = "x",
              defaultValue = "None",
              positional = true,
              named = false,
              doc =
                  "A positional parameter distinct from other parameters for legacy support. "
                      + "\n" //
                      + "<p>If <code>--incompatible_disable_depset_items</code> is false, this "
                      + "parameter serves as the value of <code>items</code>.</p> "
                      + "\n" //
                      + "<p>If <code>--incompatible_disable_depset_items</code> is true, this "
                      + "parameter serves as the value of <code>direct</code>.</p> "
                      + "\n" //
                      + "<p>See the documentation for these parameters for more details."),
          // TODO(cparsons): Make 'order' keyword-only.
          @Param(
              name = "order",
              defaultValue = "\"default\"",
              doc =
                  "The traversal strategy for the new depset. See "
                      + "<a href=\"depset.html\">here</a> for the possible values.",
              named = true),
          @Param(
              name = "direct",
              defaultValue = "None",
              positional = false,
              named = true,
              doc = "A list of <i>direct</i> elements of a depset. "),
          @Param(
              name = "transitive",
              named = true,
              positional = false,
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = Depset.class),
                @ParamType(type = NoneType.class),
              },
              doc = "A list of depsets whose elements will become indirect elements of the depset.",
              defaultValue = "None"),
          @Param(
              name = "items",
              defaultValue = "[]",
              positional = false,
              doc =
                  "Deprecated: Either an iterable whose items become the direct elements of "
                      + "the new depset, in left-to-right order, or else a depset that becomes "
                      + "a transitive element of the new depset. In the latter case, "
                      + "<code>transitive</code> cannot be specified.",
              disableWithFlag = BuildLanguageOptions.INCOMPATIBLE_DISABLE_DEPSET_ITEMS,
              valueWhenDisabled = "[]",
              named = true),
        },
        useStarlarkThread = true)
    public Depset depset(
        Object x,
        String orderString,
        Object direct,
        Object transitive,
        Object items,
        StarlarkThread thread)
        throws EvalException {
      return Depset.depset(x, orderString, direct, transitive, items, thread.getSemantics());
    }

    @StarlarkMethod(
        name = "select",
        doc =
            "<code>select()</code> is the helper function that makes a rule attribute "
                + "<a href=\"$BE_ROOT/common-definitions.html#configurable-attributes\">"
                + "configurable</a>. See "
                + "<a href=\"$BE_ROOT/functions.html#select\">build encyclopedia</a> for details.",
        parameters = {
          @Param(
              name = "x",
              positional = true,
              doc =
                  "A dict that maps configuration conditions to values. Each key is a label string"
                      + " that identifies a config_setting instance."),
          @Param(
              name = "no_match_error",
              defaultValue = "''",
              doc = "Optional custom error to report if no condition matches.",
              named = true),
        })
    public Object select(Dict<?, ?> dict, String noMatchError) throws EvalException {
      return SelectorList.select(dict, noMatchError);
    }
  }

  /**
   * A library of Starlark functions (keyed by name) pre-declared in BUILD files. A superset of
   * {@link #COMMON} (e.g. select). Excludes functions in the native module, such as exports_files.
   * Examples: environment_group, select.
   */
  public static final ImmutableMap<String, Object> BUILD = initBUILD();

  private static ImmutableMap<String, Object> initBUILD() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new BuildLibrary());
    env.putAll(COMMON);
    return env.build();
  }

  @DocumentMethods
  private static class BuildLibrary {
    @StarlarkMethod(
        name = "environment_group",
        doc =
            "Defines a set of related environments that can be tagged onto rules to prevent"
                + "incompatible rules from depending on each other.",
        parameters = {
          @Param(name = "name", positional = false, named = true, doc = "The name of the rule."),
          // Both parameter below are lists of label designators
          @Param(
              name = "environments",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = Label.class),
              },
              positional = false,
              named = true,
              doc = "A list of Labels for the environments to be grouped, from the same package."),
          @Param(
              name = "defaults",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = Label.class),
              },
              positional = false,
              named = true,
              doc = "A list of Labels.")
        }, // TODO(bazel-team): document what that is
        // Not documented by docgen, as this is only available in BUILD files.
        // TODO(cparsons): Devise a solution to document BUILD functions.
        documented = false,
        useStarlarkThread = true)
    public NoneType environmentGroup(
        String name,
        Sequence<?> environmentsList, // <Label>
        Sequence<?> defaultsList, // <Label>
        StarlarkThread thread)
        throws EvalException {
      PackageContext context = getContext(thread);
      List<Label> environments =
          BuildType.LABEL_LIST.convert(
              environmentsList,
              "'environment_group argument'",
              context.pkgBuilder.getBuildFileLabel());
      List<Label> defaults =
          BuildType.LABEL_LIST.convert(
              defaultsList, "'environment_group argument'", context.pkgBuilder.getBuildFileLabel());

      if (environments.isEmpty()) {
        throw Starlark.errorf("environment group %s must contain at least one environment", name);
      }
      try {
        Location loc = thread.getCallerLocation();
        context.pkgBuilder.addEnvironmentGroup(
            name, environments, defaults, context.eventHandler, loc);
        return Starlark.NONE;
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("environment group has invalid name: %s: %s", name, e.getMessage());
      } catch (Package.NameConflictException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    }

    @StarlarkMethod(
        name = "licenses",
        doc = "Declare the license(s) for the code in the current package.",
        parameters = {
          @Param(
              name = "license_strings",
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
              doc = "A list of strings, the names of the licenses used.")
        },
        // Not documented by docgen, as this is only available in BUILD files.
        // TODO(cparsons): Devise a solution to document BUILD functions.
        documented = false,
        useStarlarkThread = true)
    public NoneType licenses(
        Sequence<?> licensesList, // list of license strings
        StarlarkThread thread)
        throws EvalException {
      PackageContext context = getContext(thread);
      try {
        License license = BuildType.LICENSE.convert(licensesList, "'licenses' operand");
        context.pkgBuilder.setDefaultLicense(license);
      } catch (ConversionException e) {
        context.eventHandler.handle(
            Package.error(thread.getCallerLocation(), e.getMessage(), Code.LICENSE_PARSE_FAILURE));
        context.pkgBuilder.setContainsErrors();
      }
      return Starlark.NONE;
    }

    @StarlarkMethod(
        name = "distribs",
        doc = "Declare the distribution(s) for the code in the current package.",
        parameters = {@Param(name = "distribution_strings", doc = "The distributions.")},
        // Not documented by docgen, as this is only available in BUILD files.
        // TODO(cparsons): Devise a solution to document BUILD functions.
        documented = false,
        useStarlarkThread = true)
    public NoneType distribs(Object object, StarlarkThread thread) throws EvalException {
      PackageContext context = getContext(thread);

      try {
        Set<DistributionType> distribs =
            BuildType.DISTRIBUTIONS.convert(object, "'distribs' operand");
        context.pkgBuilder.setDefaultDistribs(distribs);
      } catch (ConversionException e) {
        context.eventHandler.handle(
            Package.error(
                thread.getCallerLocation(), e.getMessage(), Code.DISTRIBUTIONS_PARSE_FAILURE));
        context.pkgBuilder.setContainsErrors();
      }
      return Starlark.NONE;
    }
  }
}
