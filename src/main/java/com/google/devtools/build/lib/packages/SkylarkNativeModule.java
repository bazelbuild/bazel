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

import static com.google.devtools.build.lib.packages.PackageFactory.getContext;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.packages.PackageFactory.NotRepresentableException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.RuleClass.Builder.ThirdPartyLicenseExistencePolicy;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkNativeModuleApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.syntax.Tuple;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** The Skylark native module. */
// TODO(cparsons): Move the definition of native.package() to this class.
public class SkylarkNativeModule implements SkylarkNativeModuleApi {

  /**
   * This map contains all the (non-rule) functions of the native module (keyed by their symbol
   * name). These native module bindings should be added (without the 'native' module namespace) to
   * the global Starlark environment for BUILD files.
   *
   * <p>For example, the function "glob" is available under both a global symbol name {@code glob()}
   * as well as under the native module namepsace {@code native.glob()}. An entry of this map is
   * thus ("glob" : glob function).
   */
  public static final ImmutableMap<String, Object> BINDINGS_FOR_BUILD_FILES = initializeBindings();

  private static ImmutableMap<String, Object> initializeBindings() {
    ImmutableMap.Builder<String, Object> bindings = ImmutableMap.builder();
    Starlark.addMethods(bindings, new SkylarkNativeModule());
    return bindings.build();
  }

  @Override
  public Sequence<?> glob(
      Sequence<?> include,
      Sequence<?> exclude,
      Integer excludeDirs,
      Object allowEmptyArgument,
      StarlarkThread thread)
      throws EvalException, ConversionException, InterruptedException {
    BazelStarlarkContext.from(thread).checkLoadingPhase("native.glob");
    PackageContext context = getContext(thread);

    List<String> includes = Type.STRING_LIST.convert(include, "'glob' argument");
    List<String> excludes = Type.STRING_LIST.convert(exclude, "'glob' argument");

    List<String> matches;
    boolean allowEmpty;
    if (allowEmptyArgument == Starlark.UNBOUND) {
      allowEmpty = !thread.getSemantics().incompatibleDisallowEmptyGlob();
    } else if (allowEmptyArgument instanceof Boolean) {
      allowEmpty = (Boolean) allowEmptyArgument;
    } else {
      throw Starlark.errorf(
          "expected boolean for argument `allow_empty`, got `%s`", allowEmptyArgument);
    }

    try {
      Globber.Token globToken =
          context.globber.runAsync(includes, excludes, excludeDirs != 0, allowEmpty);
      matches = context.globber.fetchUnsorted(globToken);
    } catch (IOException e) {
      String errorMessage =
          String.format(
              "error globbing [%s]%s: %s",
              Joiner.on(", ").join(includes),
              excludes.isEmpty() ? "" : " - [" + Joiner.on(", ").join(excludes) + "]",
              e.getMessage());
      Location loc = thread.getCallerLocation();
      context.eventHandler.handle(Event.error(loc, errorMessage));
      context.pkgBuilder.setIOExceptionAndMessage(e, errorMessage);
      matches = ImmutableList.of();
    } catch (BadGlobException e) {
      throw new EvalException(null, e.getMessage());
    } catch (IllegalArgumentException e) {
      throw new EvalException(null, "illegal argument in call to glob", e);
    }

    ArrayList<String> result = new ArrayList<>(matches.size());
    for (String match : matches) {
      if (match.charAt(0) == '@') {
        // Add explicit colon to disambiguate from external repository.
        match = ":" + match;
      }
      result.add(match);
    }
    result.sort(Comparator.naturalOrder());

    return StarlarkList.copyOf(thread.mutability(), result);
  }

  @Override
  public Object existingRule(String name, StarlarkThread thread)
      throws EvalException, InterruptedException {
    BazelStarlarkContext.from(thread).checkLoadingOrWorkspacePhase("native.existing_rule");
    PackageContext context = getContext(thread);
    Target target = context.pkgBuilder.getTarget(name);
    Dict<String, Object> rule = targetDict(target, thread.mutability());
    return rule != null ? rule : Starlark.NONE;
  }

  /*
    If necessary, we could allow filtering by tag (anytag, alltags), name (regexp?), kind ?
    For now, we ignore this, since users can implement it in Skylark.
  */
  @Override
  public Dict<String, Dict<String, Object>> existingRules(StarlarkThread thread)
      throws EvalException, InterruptedException {
    BazelStarlarkContext.from(thread).checkLoadingOrWorkspacePhase("native.existing_rules");
    PackageContext context = getContext(thread);
    Collection<Target> targets = context.pkgBuilder.getTargets();
    Mutability mu = thread.mutability();
    Dict<String, Dict<String, Object>> rules = Dict.of(mu);
    for (Target t : targets) {
      if (t instanceof Rule) {
        Dict<String, Object> rule = targetDict(t, mu);
        Preconditions.checkNotNull(rule);
        rules.put(t.getName(), rule, (Location) null);
      }
    }

    return rules;
  }

  @Override
  public NoneType packageGroup(
      String name,
      Sequence<?> packagesO,
      Sequence<?> includesO,
      StarlarkThread thread)
      throws EvalException {
    BazelStarlarkContext.from(thread).checkLoadingPhase("native.package_group");
    PackageContext context = getContext(thread);

    List<String> packages =
        Type.STRING_LIST.convert(packagesO, "'package_group.packages argument'");
    List<Label> includes =
        BuildType.LABEL_LIST.convert(
            includesO, "'package_group.includes argument'", context.pkgBuilder.getBuildFileLabel());

    Location loc = thread.getCallerLocation();
    try {
      context.pkgBuilder.addPackageGroup(name, packages, includes, context.eventHandler, loc);
      return Starlark.NONE;
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("package group has invalid name: %s: %s", name, e.getMessage());
    } catch (Package.NameConflictException e) {
      throw new EvalException(null, e.getMessage());
    }
  }

  @Override
  public NoneType exportsFiles(
      Sequence<?> srcs, Object visibilityO, Object licensesO, StarlarkThread thread)
      throws EvalException {
    BazelStarlarkContext.from(thread).checkLoadingPhase("native.exports_files");
    Package.Builder pkgBuilder = getContext(thread).pkgBuilder;
    List<String> files = Type.STRING_LIST.convert(srcs, "'exports_files' operand");

    RuleVisibility visibility =
        EvalUtils.isNullOrNone(visibilityO)
            ? ConstantRuleVisibility.PUBLIC
            : PackageFactory.getVisibility(
                pkgBuilder.getBuildFileLabel(),
                BuildType.LABEL_LIST.convert(
                    visibilityO, "'exports_files' operand", pkgBuilder.getBuildFileLabel()));

    // TODO(bazel-team): is licenses plural or singular?
    License license = BuildType.LICENSE.convertOptional(licensesO, "'exports_files' operand");

    Location loc = thread.getCallerLocation();
    for (String file : files) {
      String errorMessage = LabelValidator.validateTargetName(file);
      if (errorMessage != null) {
        throw Starlark.errorf("%s", errorMessage);
      }
      try {
        InputFile inputFile = pkgBuilder.createInputFile(file, loc);
        if (inputFile.isVisibilitySpecified() && inputFile.getVisibility() != visibility) {
          throw Starlark.errorf(
              "visibility for exported file '%s' declared twice", inputFile.getName());
        }
        if (license != null && inputFile.isLicenseSpecified()) {
          throw Starlark.errorf(
              "licenses for exported file '%s' declared twice", inputFile.getName());
        }

        // See if we should check third-party licenses: first checking for any hard-coded policy,
        // then falling back to user-settable flags.
        boolean checkLicenses;
        if (pkgBuilder.getThirdPartyLicenseExistencePolicy()
            == ThirdPartyLicenseExistencePolicy.ALWAYS_CHECK) {
          checkLicenses = true;
        } else if (pkgBuilder.getThirdPartyLicenseExistencePolicy()
            == ThirdPartyLicenseExistencePolicy.NEVER_CHECK) {
          checkLicenses = false;
        } else {
          checkLicenses = !thread.getSemantics().incompatibleDisableThirdPartyLicenseChecking();
        }

        if (checkLicenses
            && license == null
            && !pkgBuilder.getDefaultLicense().isSpecified()
            && RuleClass.isThirdPartyPackage(pkgBuilder.getPackageIdentifier())) {
          throw Starlark.errorf(
              "third-party file '%s' lacks a license declaration with one of the following types:"
                  + " notice, reciprocal, permissive, restricted, unencumbered, by_exception_only",
              inputFile.getName());
        }

        pkgBuilder.setVisibilityAndLicense(inputFile, visibility, license);
      } catch (Package.Builder.GeneratedLabelConflict e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    }
    return Starlark.NONE;
  }

  @Override
  public String packageName(StarlarkThread thread) throws EvalException {
    BazelStarlarkContext.from(thread).checkLoadingPhase("native.package_name");
    PackageIdentifier packageId =
        PackageFactory.getContext(thread).getBuilder().getPackageIdentifier();
    return packageId.getPackageFragment().getPathString();
  }

  @Override
  public String repositoryName(StarlarkThread thread) throws EvalException {
    BazelStarlarkContext.from(thread).checkLoadingPhase("native.repository_name");
    PackageIdentifier packageId =
        PackageFactory.getContext(thread).getBuilder().getPackageIdentifier();
    return packageId.getRepository().toString();
  }

  @Nullable
  private static Dict<String, Object> targetDict(Target target, Mutability mu)
      throws EvalException {
    if (!(target instanceof Rule)) {
      return null;
    }
    Dict<String, Object> values = Dict.of(mu);

    Rule rule = (Rule) target;
    AttributeContainer cont = rule.getAttributeContainer();
    for (Attribute attr : rule.getAttributes()) {
      if (!Character.isAlphabetic(attr.getName().charAt(0))) {
        continue;
      }

      if (attr.getName().equals("distribs")) {
        // attribute distribs: cannot represent type class java.util.Collections$SingletonSet
        // in Skylark: [INTERNAL].
        continue;
      }

      try {
        Object val = skylarkifyValue(mu, cont.getAttr(attr.getName()), target.getPackage());
        if (val == null) {
          continue;
        }
        values.put(attr.getName(), val, (Location) null);
      } catch (NotRepresentableException e) {
        throw new NotRepresentableException(
            String.format(
                "target %s, attribute %s: %s", target.getName(), attr.getName(), e.getMessage()));
      }
    }

    values.put("name", rule.getName(), (Location) null);
    values.put("kind", rule.getRuleClass(), (Location) null);
    return values;
  }

  /**
   * Converts a target attribute value to a Starlark value for return in {@code
   * native.existing_rule()} or {@code native.existing_rules()}.
   *
   * <p>Any dict values in the result have mutability {@code mu}.
   *
   * @return the value, or null if we don't want to export it to the user.
   * @throws NotRepresentableException if an unknown type is encountered.
   */
  @Nullable
  private static Object skylarkifyValue(Mutability mu, Object val, Package pkg)
      throws NotRepresentableException {
    if (val == null) {
      return null;
    }
    if (val instanceof Boolean) {
      return val;
    }
    if (val instanceof Integer) {
      return val;
    }
    if (val instanceof String) {
      return val;
    }

    if (val instanceof TriState) {
      switch ((TriState) val) {
        case AUTO:
          return -1;
        case YES:
          return 1;
        case NO:
          return 0;
      }
    }

    if (val instanceof Label) {
      Label l = (Label) val;
      if (l.getPackageName().equals(pkg.getName())) {
        return ":" + l.getName();
      }
      return l.getCanonicalForm();
    }

    if (val instanceof List) {
      List<Object> l = new ArrayList<>();
      for (Object o : (List) val) {
        Object elt = skylarkifyValue(mu, o, pkg);
        if (elt == null) {
          continue;
        }

        l.add(elt);
      }

      return Tuple.copyOf(l);
    }
    if (val instanceof Map) {
      Map<Object, Object> m = new TreeMap<>();
      for (Map.Entry<?, ?> e : ((Map<?, ?>) val).entrySet()) {
        Object key = skylarkifyValue(mu, e.getKey(), pkg);
        Object mapVal = skylarkifyValue(mu, e.getValue(), pkg);

        if (key == null || mapVal == null) {
          continue;
        }

        m.put(key, mapVal);
      }
      return Starlark.fromJava(m, mu);
    }
    if (val.getClass().isAnonymousClass()) {
      // Computed defaults. They will be represented as
      // "deprecation": com.google.devtools.build.lib.analysis.BaseRuleClasses$2@6960884a,
      // Filter them until we invent something more clever.
      return null;
    }

    if (val instanceof License) {
      // License is deprecated as a Starlark type, so omit this type from Starlark values
      // to avoid exposing these objects, even though they are technically StarlarkValue.
      return null;
    }

    if (val instanceof StarlarkValue) {
      return val;
    }

    if (val instanceof BuildType.SelectorList) {
      // This is terrible:
      //  1) this value is opaque, and not a BUILD value, so it cannot be used in rule arguments
      //  2) its representation has a pointer address, so it breaks hermeticity.
      //
      // Even though this is clearly imperfect, we return this value because otherwise
      // native.rules() fails if there is any rule using a select() in the BUILD file.
      //
      // To remedy this, we should return a SelectorList. To do so, we have to
      // 1) recurse into the Selector contents of SelectorList, so those values are skylarkified too
      // 2) get the right Class<?> value. We could probably get at that by looking at
      //    ((SelectorList)val).getSelectors().first().getEntries().first().getClass().

      return val;
    }

    // We are explicit about types we don't understand so we minimize changes to existing callers
    // if we add more types that we can represent.
    throw new NotRepresentableException(
        String.format("cannot represent %s (%s) in Starlark", val, val.getClass()));
  }
}
