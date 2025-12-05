package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAttrModuleApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;

/**
 * Fake implementation of {@link StarlarkAttrModuleApi}.
 */
public class FakeStarlarkAttrModuleApi implements StarlarkAttrModuleApi {

  @Override
  public Descriptor licenseAttribute(
      Object defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    // LICENSE type doesn't exist, use STRING instead
    return new FakeDescriptor(AttributeType.STRING, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor stringListDictAttribute(
      Boolean mandatory,
      Object allowEmpty,
      Dict<?, ?> defaultValue,
      Object doc,
      Boolean allowFiles,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.STRING_LIST_DICT, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor intAttribute(
      Object configurable,
      StarlarkInt defaultValue,
      Object doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.INT, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor stringAttribute(
      Object configurable,
      Object defaultValue,
      Object doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    String defaultString = defaultValue instanceof String ? (String) defaultValue : null;
    return new FakeDescriptor(
        AttributeType.STRING,
        docString,
        mandatory,
        ImmutableList.of(),
        defaultString != null ? "\"" + defaultString + "\"" : null);
  }

  @Override
  public Descriptor dormantLabelAttribute(
      Object defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.LABEL, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor labelAttribute(
      Object configurable,
      Object defaultValue,
      Object materializer,
      Object doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      Boolean skipValidations,
      Sequence<?> providers,
      Object forDependencyResolution,
      Object allowRules,
      Object cfg,
      Sequence<?> aspects,
      Sequence<?> flags,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(AttributeType.LABEL, docString, mandatory, allNameGroups, defaultValue);
  }

  @Override
  public Descriptor stringListAttribute(
      Boolean mandatory,
      Boolean allowEmpty,
      Object configurable,
      Object defaultValue,
      Object doc,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(
        AttributeType.STRING_LIST, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor intListAttribute(
      Boolean mandatory,
      Boolean allowEmpty,
      Object configurable,
      Sequence<?> defaultValue,
      Object doc,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(
        AttributeType.INT_LIST, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object configurable,
      Object defaultValue,
      Object materializer,
      Object doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Object forDependencyResolution,
      Sequence<?> flags,
      Boolean mandatory,
      Boolean skipValidations,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(AttributeType.LABEL_LIST, docString, mandatory, allNameGroups, defaultValue);
  }

  @Override
  public Descriptor stringKeyedLabelDictAttribute(
      Boolean allowEmpty,
      Object configurable,
      Object defaultValue,
      Object doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Object forDependencyResolution,
      Sequence<?> flags,
      Boolean mandatory,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(
        AttributeType.LABEL_STRING_DICT, docString, mandatory, allNameGroups, defaultValue);
  }

  @Override
  public Descriptor dormantLabelListAttribute(
      Boolean allowEmpty,
      Object defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.LABEL_LIST, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object configurable,
      Object defaultValue,
      Object doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Object forDependencyResolution,
      Sequence<?> flags,
      Boolean mandatory,
      Boolean skipValidations,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(
        AttributeType.LABEL_STRING_DICT, docString, mandatory, allNameGroups, defaultValue);
  }

  @Override
  public Descriptor boolAttribute(
      Object configurable,
      Boolean defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread) throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(
        AttributeType.BOOLEAN,
        docString,
        mandatory,
        ImmutableList.of(),
        Boolean.TRUE.equals(defaultValue) ? "True" : "False");
  }

  @Override
  public Descriptor outputAttribute(Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.OUTPUT, docString, mandatory, ImmutableList.of(), "");
  }

  @Override
  public Descriptor outputListAttribute(
      Boolean allowEmpty,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.OUTPUT_LIST, docString, mandatory, ImmutableList.of(), "");
  }

  @Override
  public Descriptor stringDictAttribute(
      Boolean allowEmpty,
      Object configurable,
      Dict<?, ?> defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.STRING_DICT, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public Descriptor labelListDictAttribute(
      Boolean allowEmpty,
      Object configurable,
      Dict<?, ?> defaultValue,
      Object doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Object forDependencyResolution,
      Sequence<?> flags,
      Boolean mandatory,
      Boolean skipValidations,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    return new FakeDescriptor(AttributeType.LABEL_LIST_DICT, docString, mandatory, ImmutableList.of(), defaultValue);
  }

  @Override
  public void repr(Printer printer) {
  }

  /**
   * Returns a list of provider name groups, given the value of a Starlark
   * attribute's "providers"
   * argument.
   *
   * <p>
   * {@code providers} can either be a list of providers or a list of lists of
   * providers, where
   * each provider is represented by a ProviderApi or by a String. In the case of
   * a single-level
   * list, the whole list is considered a single group, while in the case of a
   * double-level list,
   * each of the inner lists is a separate group.
   */
  private static List<List<String>> allProviderNameGroups(
      Sequence<?> providers, StarlarkThread thread) {

    List<List<String>> allNameGroups = new ArrayList<>();
    for (Object object : providers) {
      List<String> providerNameGroup;
      if (object instanceof Sequence) {
        Sequence<?> group = (Sequence<?>) object;
        providerNameGroup = parseProviderGroup(group, thread);
        allNameGroups.add(providerNameGroup);
      } else {
        providerNameGroup = parseProviderGroup(providers, thread);
        allNameGroups.add(providerNameGroup);
        break;
      }
    }
    return allNameGroups;
  }

  /**
   * Returns the names of the providers in the given group.
   *
   * <p>
   * Each item in the group may be either a {@link ProviderApi} or a
   * {@code String} (representing
   * a legacy provider).
   */
  private static List<String> parseProviderGroup(Sequence<?> group, StarlarkThread thread) {
    List<String> providerNameGroup = new ArrayList<>();
    for (Object object : group) {
      if (object instanceof ProviderApi) {
        ProviderApi provider = (ProviderApi) object;
        String providerName = providerName(provider, thread);
        providerNameGroup.add(providerName);
      } else if (object instanceof String) {
        String legacyProvider = (String) object;
        providerNameGroup.add(legacyProvider);
      }
    }
    return providerNameGroup;
  }

  // Returns the name of the provider using fragile heuristics.
  private static String providerName(ProviderApi provider, StarlarkThread thread) {
    Module bzl = Module.ofInnermostEnclosingStarlarkFunction(thread);

    // Generic fake provider? (e.g. Starlark-defined, or trivial fake)
    // Return name set at construction, or by "export" operation, if any.
    if (provider instanceof FakeProviderApi) {
      return ((FakeProviderApi) provider).getName(); // may be "Unexported Provider"
    }

    // Specialized fake provider? (e.g. DefaultInfo)
    // Return name under which FakeApi.addPredeclared added it to environment.
    // (This only works for top-level names such as DefaultInfo, but not for
    // nested ones such as cc_common.XyzInfo, but that has always been broken;
    // it is not part of the regression that is b/175703093.)
    for (Map.Entry<String, Object> e : bzl.getPredeclaredBindings().entrySet()) {
      if (provider.equals(e.getValue())) {
        return e.getKey();
      }
    }

    return "Unknown Provider";
  }
}
