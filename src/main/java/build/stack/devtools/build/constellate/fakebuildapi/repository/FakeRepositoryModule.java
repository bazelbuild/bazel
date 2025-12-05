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
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionTagClassInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import build.stack.devtools.build.constellate.rendering.ModuleExtensionInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RepositoryRuleInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RuleInfoWrapper;
import build.stack.devtools.build.constellate.rendering.TagClassInfoWrapper;
import java.util.List;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
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
  private final List<RepositoryRuleInfoWrapper> repositoryRuleInfoList;
  private final List<ModuleExtensionInfoWrapper> moduleExtensionInfoList;

  public FakeRepositoryModule(List<RuleInfoWrapper> ruleInfoList) {
    this(ruleInfoList, null, null);
  }

  public FakeRepositoryModule(List<RuleInfoWrapper> ruleInfoList,
      List<RepositoryRuleInfoWrapper> repositoryRuleInfoList) {
    this(ruleInfoList, repositoryRuleInfoList, null);
  }

  public FakeRepositoryModule(List<RuleInfoWrapper> ruleInfoList,
      List<RepositoryRuleInfoWrapper> repositoryRuleInfoList,
      List<ModuleExtensionInfoWrapper> moduleExtensionInfoList) {
    this.ruleInfoList = ruleInfoList;
    this.repositoryRuleInfoList = repositoryRuleInfoList;
    this.moduleExtensionInfoList = moduleExtensionInfoList;
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

    Location loc = thread.getCallerLocation();

    // If repositoryRuleInfoList is provided, use RepositoryRuleInfo proto
    if (repositoryRuleInfoList != null) {
      RepositoryRuleInfo.Builder repositoryRuleInfo = RepositoryRuleInfo.newBuilder()
          .setDocString(docString)
          .addAllAttribute(attrInfos);

      // Add environment variables if provided
      if (environ != null && !Starlark.isNullOrNone(environ)) {
        for (Object envVar : environ) {
          if (envVar instanceof String) {
            repositoryRuleInfo.addEnviron((String) envVar);
          }
        }
      }

      repositoryRuleInfoList.add(new RepositoryRuleInfoWrapper(functionIdentifier, loc, repositoryRuleInfo));
    } else {
      // Fallback: store as RuleInfo for backwards compatibility
      RuleInfo.Builder ruleInfo = RuleInfo.newBuilder().setDocString(docString).addAllAttribute(attrInfos);
      ruleInfoList.add(new RuleInfoWrapper(functionIdentifier, loc, ruleInfo));
    }

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
    String docString = doc instanceof String ? (String) doc : "";

    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && !Starlark.isNullOrNone(attrs)) {
      attrsMapBuilder.putAll(Dict.cast(attrs, String.class, FakeDescriptor.class, "attrs"));
    }

    List<AttributeInfo> attrInfos = attrsMapBuilder.build().entrySet().stream()
        .filter(entry -> !entry.getKey().startsWith("_"))
        .map(entry -> entry.getValue().asAttributeInfo(entry.getKey()))
        .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    // Return a FakeTagClass that stores the tag class info
    return new FakeTagClass(docString, attrInfos);
  }

  private static class FakeTagClass implements TagClassApi {
    private final String docString;
    private final List<AttributeInfo> attributes;

    FakeTagClass(String docString, List<AttributeInfo> attributes) {
      this.docString = docString;
      this.attributes = attributes;
    }

    public String getDocString() {
      return docString;
    }

    public List<AttributeInfo> getAttributes() {
      return attributes;
    }
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

    if (moduleExtensionInfoList == null) {
      // If no list provided, return stub
      return new Object();
    }

    String docString = doc instanceof String ? (String) doc : "";

    ModuleExtensionInfo.Builder moduleExtensionInfo = ModuleExtensionInfo.newBuilder()
        .setDocString(docString);

    // Process tag classes
    if (tagClasses != null && !Starlark.isNullOrNone(tagClasses)) {
      Dict<String, TagClassApi> tagClassDict = Dict.cast(tagClasses, String.class, TagClassApi.class, "tag_classes");
      for (var entry : tagClassDict.entrySet()) {
        String tagName = entry.getKey();
        TagClassApi tagClassApi = entry.getValue();

        if (tagClassApi instanceof FakeTagClass) {
          FakeTagClass fakeTagClass = (FakeTagClass) tagClassApi;
          ModuleExtensionTagClassInfo tagClassInfo = ModuleExtensionTagClassInfo.newBuilder()
              .setTagName(tagName)
              .setDocString(fakeTagClass.getDocString())
              .addAllAttribute(fakeTagClass.getAttributes())
              .build();
          moduleExtensionInfo.addTagClass(tagClassInfo);
        }
      }
    }

    ModuleExtensionDefinitionIdentifier identifier = new ModuleExtensionDefinitionIdentifier();
    Location loc = thread.getCallerLocation();

    moduleExtensionInfoList.add(new ModuleExtensionInfoWrapper(identifier, loc, moduleExtensionInfo));

    return identifier;
  }

  /**
   * A fake identifier for module extension definitions.
   */
  private static class ModuleExtensionDefinitionIdentifier implements PostAssignHookAssignableIdentifier, StarlarkValue {
    private static int idCounter = 0;
    private final String name = "ModuleExtensionDefinitionIdentifier" + idCounter++;
    private String assignedName = "<unassigned>";

    @Override
    public void setAssignedName(String assignedName) {
      this.assignedName = assignedName;
    }

    @Override
    public String getAssignedName() {
      return assignedName;
    }

    public String getName() {
      return name;
    }
  }
}
