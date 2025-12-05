package build.stack.devtools.build.constellate.fakebuildapi.repository;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryModuleApi;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryModuleApi.TagClassApi;
import build.stack.devtools.build.constellate.fakebuildapi.FakeDescriptor;
import build.stack.devtools.build.constellate.fakebuildapi.FakeStarlarkRuleFunctionsApi.AttributeNameComparator;
import build.stack.devtools.build.constellate.fakebuildapi.PostAssignHookAssignableIdentifier;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import build.stack.devtools.build.constellate.rendering.RuleInfoWrapper;
import java.util.List;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Fake implementation of {@link RepositoryModuleApi}.
 */
public class FakeRepositoryModule implements RepositoryModuleApi {
  private static final FakeDescriptor IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR = new FakeDescriptor(
      AttributeType.NAME, "A unique name for this repository.", true, ImmutableList.of(), "");

  private static final FakeDescriptor IMPLICIT_REPO_MAPPING_ATTRIBUTE_DESCRIPTOR = new FakeDescriptor(
      AttributeType.STRING_DICT,
      "A dictionary from local repository name to global repository name. "
          + "This allows controls over workspace dependency resolution for dependencies of "
          + "this repository."
          + "<p>For example, an entry `\"@foo\": \"@bar\"` declares that, for any time "
          + "this repository depends on `@foo` (such as a dependency on "
          + "`@foo//some:target`, it should actually resolve that dependency within "
          + "globally-declared `@bar` (`@bar//some:target`).",
      true,
      ImmutableList.of(),
      "");

  private final List<RuleInfoWrapper> ruleInfoList;

  public FakeRepositoryModule(List<RuleInfoWrapper> ruleInfoList) {
    this.ruleInfoList = ruleInfoList;
  }

  @Override
  public StarlarkCallable repositoryRule(
      StarlarkCallable implementation,
      Object attrs,
      Boolean local,
      Sequence<?> environ, // <String> expected
      Boolean configure,
      Boolean remotable,
      Object doc,
      StarlarkThread thread)
      throws EvalException {
    String docString = doc instanceof String ? (String) doc : "";
    List<AttributeInfo> attrInfos;
    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && attrs != Starlark.NONE) {
      attrsMapBuilder.putAll(Dict.cast(attrs, String.class, FakeDescriptor.class, "attrs"));
    }

    attrsMapBuilder.put("name", IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR);
    attrsMapBuilder.put("repo_mapping", IMPLICIT_REPO_MAPPING_ATTRIBUTE_DESCRIPTOR);
    attrInfos = attrsMapBuilder.build().entrySet().stream()
        .filter(entry -> !entry.getKey().startsWith("_"))
        .map(entry -> entry.getValue().asAttributeInfo(entry.getKey()))
        .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    RepositoryRuleDefinitionIdentifier functionIdentifier = new RepositoryRuleDefinitionIdentifier();

    // Only the Builder is passed to RuleInfoWrapper as the rule name is not yet
    // available.
    RuleInfo.Builder ruleInfo = RuleInfo.newBuilder().setDocString(docString).addAllAttribute(attrInfos);

    Location loc = thread.getCallerLocation();
    ruleInfoList.add(new RuleInfoWrapper(functionIdentifier, loc, ruleInfo));
    return functionIdentifier;
  }

  /**
   * A fake {@link StarlarkCallable} implementation which serves as an identifier
   * for a rule
   * definition. A Starlark invocation of 'rule()' should spawn a unique instance
   * of this class and
   * return it. Thus, Starlark code such as 'foo = rule()' will result in 'foo'
   * being assigned to a
   * unique identifier, which can later be matched to a registered rule()
   * invocation saved by the
   * fake build API implementation.
   */
  private static class RepositoryRuleDefinitionIdentifier
      implements StarlarkCallable, PostAssignHookAssignableIdentifier {

    private static int idCounter = 0;
    private final String name = "RepositoryRuleDefinitionIdentifier" + idCounter++;
    private String assignedName = "<unassigned>";

    @Override
    public void setAssignedName(String assignedName) {
      this.assignedName = assignedName;
    }

    @Override
    public String getAssignedName() {
      return assignedName;
    }

    @Override
    public String getName() {
      return name;
    }
  }

  @Override
  public void failWithIncompatibleUseCcConfigureFromRulesCc(StarlarkThread thread)
      throws EvalException {
    // Noop until --incompatible_use_cc_configure_from_rules_cc is implemented.
  }

  @Override
  public TagClassApi tagClass(Dict<?, ?> attrs, Object doc) throws EvalException {
    // Stub implementation - return a fake tag class
    return new FakeTagClass();
  }

  private static class FakeTagClass implements TagClassApi {
    // Stub implementation
  }

  @Override
  public Object moduleExtension(
      StarlarkCallable implementation,
      Dict<?, ?> tagClasses,
      Object doc,
      Sequence<?> environ,
      boolean osDependent,
      boolean archDependent,
      StarlarkThread thread)
      throws EvalException {
    // Stub implementation
    return new Object();
  }
}
