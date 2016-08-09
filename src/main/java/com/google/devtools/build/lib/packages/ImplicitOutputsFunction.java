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
package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.syntax.SkylarkType.castMap;
import static java.util.Collections.singleton;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.escape.Escaper;
import com.google.common.escape.Escapers;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A function interface allowing rules to specify their set of implicit outputs
 * in a more dynamic way than just simple template-substitution.  For example,
 * the set of implicit outputs may be a function of rule attributes.
 */
public abstract class ImplicitOutputsFunction {

  /**
   * Implicit output functions for Skylark supporting key value access of expanded implicit outputs.
   */
  public abstract static class SkylarkImplicitOutputsFunction extends ImplicitOutputsFunction {

    public abstract ImmutableMap<String, String> calculateOutputs(AttributeMap map)
        throws EvalException, InterruptedException;

    @Override
    public Iterable<String> getImplicitOutputs(AttributeMap map)
        throws EvalException, InterruptedException {
      return calculateOutputs(map).values();
    }
  }

  /**
   * Implicit output functions executing Skylark code.
   */
  public static final class SkylarkImplicitOutputsFunctionWithCallback
      extends SkylarkImplicitOutputsFunction {

    private final SkylarkCallbackFunction callback;
    private final Location loc;

    public SkylarkImplicitOutputsFunctionWithCallback(
        SkylarkCallbackFunction callback, Location loc) {
      this.callback = callback;
      this.loc = loc;
    }

    @Override
    public ImmutableMap<String, String> calculateOutputs(AttributeMap map)
        throws EvalException, InterruptedException {
      Map<String, Object> attrValues = new HashMap<>();
      for (String attrName : map.getAttributeNames()) {
        Type<?> attrType = map.getAttributeType(attrName);
        // Don't include configurable attributes: we don't know which value they might take
        // since we don't yet have a build configuration.
        if (!map.isConfigurable(attrName, attrType)) {
          Object value = map.get(attrName, attrType);
          attrValues.put(attrName, value == null ? Runtime.NONE : value);
        }
      }
      ClassObject attrs = SkylarkClassObjectConstructor.STRUCT.create(
          attrValues,
          "Attribute '%s' either doesn't exist "
          + "or uses a select() (i.e. could have multiple values)");
      try {
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        for (Map.Entry<String, String> entry : castMap(callback.call(attrs),
            String.class, String.class, "implicit outputs function return value").entrySet()) {
          Iterable<String> substitutions = fromTemplates(entry.getValue()).getImplicitOutputs(map);
          if (!Iterables.isEmpty(substitutions)) {
            builder.put(entry.getKey(), Iterables.getOnlyElement(substitutions));
          }
        }
        return builder.build();
      } catch (IllegalArgumentException e) {
        throw new EvalException(loc, e.getMessage());
      }
    }
  }

  /**
   * Implicit output functions using a simple an output map.
   */
  public static final class SkylarkImplicitOutputsFunctionWithMap
      extends SkylarkImplicitOutputsFunction {

    private final ImmutableMap<String, String> outputMap;

    public SkylarkImplicitOutputsFunctionWithMap(ImmutableMap<String, String> outputMap) {
      this.outputMap = outputMap;
    }

    @Override
    public ImmutableMap<String, String> calculateOutputs(AttributeMap map) {
      ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
      for (Map.Entry<String, String> entry : outputMap.entrySet()) {
        Iterable<String> substitutions = fromTemplates(entry.getValue()).getImplicitOutputs(map);
        if (!Iterables.isEmpty(substitutions)) {
          builder.put(entry.getKey(), Iterables.getOnlyElement(substitutions));
        }
      }
      return builder.build();
    }
  }

  /**
   * Implicit output functions which can not throw an EvalException.
   */
  public abstract static class SafeImplicitOutputsFunction extends ImplicitOutputsFunction {
    @Override
    public abstract Iterable<String> getImplicitOutputs(AttributeMap map);
  }

  /**
   * An interface to objects that can retrieve rule attributes.
   */
  public interface AttributeValueGetter {
    /**
     * Returns the value(s) of attribute "attr" in "rule", or empty set if attribute unknown.
     */
    Set<String> get(AttributeMap rule, String attr);
  }

  /**
   * The default rule attribute retriever.
   *
   * <p>Custom {@link AttributeValueGetter} implementations may delegate to this object as a
   * fallback mechanism.
   */
  public static final AttributeValueGetter DEFAULT_RULE_ATTRIBUTE_GETTER =
      new AttributeValueGetter() {
        @Override
        public Set<String> get(AttributeMap rule, String attr) {
          return attributeValues(rule, attr);
        }
      };

  private static final Escaper PERCENT_ESCAPER = Escapers.builder().addEscape('%', "%%").build();

  /**
   * Given a newly-constructed Rule instance (with attributes populated),
   * returns the list of output files that this rule produces implicitly.
   */
  public abstract Iterable<String> getImplicitOutputs(AttributeMap rule)
      throws EvalException, InterruptedException;

  /**
   * The implicit output function that returns no files.
   */
  public static final SafeImplicitOutputsFunction NONE = new SafeImplicitOutputsFunction() {
      @Override public Iterable<String> getImplicitOutputs(AttributeMap rule) {
        return Collections.emptyList();
      }
    };

  /**
   * A convenience wrapper for {@link #fromTemplates(Iterable)}.
   */
  public static SafeImplicitOutputsFunction fromTemplates(String... templates) {
    return fromTemplates(Arrays.asList(templates));
  }

  /**
   * The implicit output function that generates files based on a set of
   * template substitutions using rule attribute values.
   *
   * @param templates The templates used to construct the name of the implicit
   *   output file target.  The substring "%{name}" will be replaced by the
   *   actual name of the rule, the substring "%{srcs}" will be replaced by the
   *   name of each source file without its extension.  If multiple %{}
   *   substrings exist, the cross-product of them is generated.
   */
  public static SafeImplicitOutputsFunction fromTemplates(final Iterable<String> templates) {
    return new SafeImplicitOutputsFunction() {
      // TODO(bazel-team): parse the templates already here
      @Override
      public Iterable<String> getImplicitOutputs(AttributeMap rule) {
        Iterable<String> result = null;
        for (String template : templates) {
          List<String> substitutions = substitutePlaceholderIntoTemplate(template, rule);
          if (substitutions.isEmpty()) {
            continue;
          }
          if (result == null) {
            result = substitutions;
          } else {
            result = Iterables.concat(result, substitutions);
          }
        }
        if (result == null) {
          return ImmutableList.<String>of();
        } else {
          return Sets.newLinkedHashSet(result);
        }
      }

      @Override
      public String toString() {
        return StringUtil.joinEnglishList(templates);
      }
    };
  }

  /**
   * A convenience wrapper for {@link #fromFunctions(Iterable)}.
   */
  public static SafeImplicitOutputsFunction fromFunctions(
      SafeImplicitOutputsFunction... functions) {
    return fromFunctions(Arrays.asList(functions));
  }

  /**
   * The implicit output function that generates files based on a set of
   * template substitutions using rule attribute values.
   *
   * @param functions The functions used to construct the name of the implicit
   *   output file target.  The substring "%{name}" will be replaced by the
   *   actual name of the rule, the substring "%{srcs}" will be replaced by the
   *   name of each source file without its extension.  If multiple %{}
   *   substrings exist, the cross-product of them is generated.
   */
  public static SafeImplicitOutputsFunction fromFunctions(
      final Iterable<SafeImplicitOutputsFunction> functions) {
    return new SafeImplicitOutputsFunction() {
      @Override
      public Iterable<String> getImplicitOutputs(AttributeMap rule) {
        Collection<String> result = new LinkedHashSet<>();
        for (SafeImplicitOutputsFunction function : functions) {
          Iterables.addAll(result, function.getImplicitOutputs(rule));
        }
        return result;
      }
      @Override
      public String toString() {
        return StringUtil.joinEnglishList(functions);
      }
    };
  }

  /**
   * Coerces attribute "attrName" of the specified rule into a sequence of
   * strings.  Helper function for {@link #fromTemplates(Iterable)}.
   */
  private static Set<String> attributeValues(AttributeMap rule, String attrName) {
    // Special case "name" since it's not treated as an attribute.
    if (attrName.equals("name")) {
      return singleton(rule.getName());
    } else if (attrName.equals("dirname")) {
      PathFragment dir = new PathFragment(rule.getName()).getParentDirectory();
      return (dir.segmentCount() == 0) ? singleton("") : singleton(dir.getPathString() + "/");
    } else if (attrName.equals("basename")) {
      return singleton(new PathFragment(rule.getName()).getBaseName());
    }

    Type<?> attrType = rule.getAttributeType(attrName);
    if (attrType == null) { return Collections.emptySet(); }
    // String attributes and lists are easy.
    if (Type.STRING == attrType) {
      return singleton(rule.get(attrName, Type.STRING));
    } else if (Type.STRING_LIST == attrType) {
      return Sets.newLinkedHashSet(rule.get(attrName, Type.STRING_LIST));
    } else if (BuildType.LABEL == attrType) {
      // Labels are most often used to change the extension,
      // e.g. %.foo -> %.java, so we return the basename w/o extension.
      Label label = rule.get(attrName, BuildType.LABEL);
      return singleton(FileSystemUtils.removeExtension(label.getName()));
    } else if (BuildType.LABEL_LIST == attrType) {
      // Labels are most often used to change the extension,
      // e.g. %.foo -> %.java, so we return the basename w/o extension.
      return Sets.newLinkedHashSet(
          Iterables.transform(rule.get(attrName, BuildType.LABEL_LIST),
              new Function<Label, String>() {
                @Override
                public String apply(Label label) {
                  return FileSystemUtils.removeExtension(label.getName());
                }
              }));
    } else if (BuildType.OUTPUT == attrType) {
      Label out = rule.get(attrName, BuildType.OUTPUT);
      return singleton(out.getName());
    } else if (BuildType.OUTPUT_LIST == attrType) {
      return Sets.newLinkedHashSet(
          Iterables.transform(rule.get(attrName, BuildType.OUTPUT_LIST),
              new Function<Label, String>() {
                @Override
                public String apply(Label label) {
                  return label.getName();
                }
              }));
    }
    throw new IllegalArgumentException(
        "Don't know how to handle " + attrName + " : " + attrType);
  }

  /**
   * Collects all named placeholders from the template while replacing them with %s.
   *
   * <p>Example: for {@code template} "%{name}_%{locales}.foo", it will return "%s_%s.foo" and
   * store "name" and "locales" in {@code placeholders}.
   *
   * <p>Incomplete placeholders are treated like text: for "a-%{x}-%{y" this method returns
   * "a-%s-%%{y" and stores "x" in {@code placeholders}.
   *
   * @param template a string with placeholders of the format %{...}
   * @param placeholders a collection to collect placeholders into; may contain duplicates if not a
   *     Set
   * @return a format string for {@link String#format}, created from the template string with every
   *     placeholder replaced by %s
   */
  public static String createPlaceholderSubstitutionFormatString(String template,
      Collection<String> placeholders) {
    return createPlaceholderSubstitutionFormatStringRecursive(template, placeholders,
        new StringBuilder());
  }

  private static String createPlaceholderSubstitutionFormatStringRecursive(String template,
      Collection<String> placeholders, StringBuilder formatBuilder) {
    int start = template.indexOf("%{");
    if (start < 0) {
      return formatBuilder.append(PERCENT_ESCAPER.escape(template)).toString();
    }

    int end = template.indexOf('}', start + 2);
    if (end < 0) {
      return formatBuilder.append(PERCENT_ESCAPER.escape(template)).toString();
    }

    formatBuilder.append(PERCENT_ESCAPER.escape(template.substring(0, start))).append("%s");
    placeholders.add(template.substring(start + 2, end));
    return createPlaceholderSubstitutionFormatStringRecursive(template.substring(end + 1),
        placeholders, formatBuilder);
  }

  /**
   * Given a template string, replaces all placeholders of the form %{...} with
   * the values from attributeSource.  If there are multiple placeholders, then
   * the output is the cross product of substitutions.
   */
  public static ImmutableList<String> substitutePlaceholderIntoTemplate(String template,
      AttributeMap rule) {
    return substitutePlaceholderIntoTemplate(template, rule, DEFAULT_RULE_ATTRIBUTE_GETTER, null);
  }

  /**
   * Substitutes attribute-placeholders in a template string, producing all possible combinations.
   *
   * @param template the template string, may contain named placeholders for rule attributes, like
   *     <code>%{name}</code> or <code>%{deps}</code>
   * @param rule the rule whose attributes the placeholders correspond to
   * @param placeholdersInTemplate if specified, will contain all placeholders found in the
   *     template; may contain duplicates
   * @return all possible combinations of the attributes referenced by the placeholders,
   *     substituted into the template; empty if any of the placeholders expands to no values
   */
  public static ImmutableList<String> substitutePlaceholderIntoTemplate(String template,
      AttributeMap rule, AttributeValueGetter attributeGetter,
      @Nullable List<String> placeholdersInTemplate) {
    List<String> placeholders = (placeholdersInTemplate == null)
        ? Lists.<String>newArrayList()
        : placeholdersInTemplate;
    String formatStr = createPlaceholderSubstitutionFormatString(template, placeholders);
    if (placeholders.isEmpty()) {
      return ImmutableList.of(template);
    }

    List<Set<String>> values = Lists.newArrayListWithCapacity(placeholders.size());
    for (String placeholder : placeholders) {
      Set<String> attrValues = attributeGetter.get(rule, placeholder);
      if (attrValues.isEmpty()) {
        return ImmutableList.<String>of();
      }
      values.add(attrValues);
    }
    ImmutableList.Builder<String> out = new ImmutableList.Builder<>();
    for (List<String> combination : Sets.cartesianProduct(values)) {
      out.add(String.format(formatStr, combination.toArray()));
    }
    return out.build();
  }
}
