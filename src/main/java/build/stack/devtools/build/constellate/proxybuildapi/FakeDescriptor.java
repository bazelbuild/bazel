package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAttrModuleApi.Descriptor;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderNameGroup;
import java.util.List;
import net.starlark.java.eval.Printer;

/**
 * Fake implementation of {@link Descriptor}.
 */
public class FakeDescriptor implements Descriptor {
  private final AttributeType type;
  private final String docString;
  private final boolean mandatory;
  private final List<List<String>> providerNameGroups;
  private final String defaultRepresentation;

  public FakeDescriptor(
      AttributeType type,
      String docString,
      boolean mandatory,
      List<List<String>> providerNameGroups,
      Object defaultObject) {
    this.type = type;
    this.docString = docString;
    this.mandatory = mandatory;
    this.providerNameGroups = providerNameGroups;
    this.defaultRepresentation = defaultObject.toString();
  }

  @Override
  public void repr(Printer printer) {
  }

  public AttributeInfo asAttributeInfo(String attributeName) {
    AttributeInfo.Builder attrInfo = AttributeInfo.newBuilder()
        .setName(attributeName)
        .setDocString(docString)
        .setType(type)
        .setMandatory(mandatory)
        .setDefaultValue(mandatory ? "" : defaultRepresentation);

    if (!providerNameGroups.isEmpty()) {
      for (List<String> providerNameGroup : providerNameGroups) {
        ProviderNameGroup.Builder providerNameListBuild = ProviderNameGroup.newBuilder();
        ProviderNameGroup providerNameList = providerNameListBuild.addAllProviderName(providerNameGroup).build();
        attrInfo.addProviderNameGroup(providerNameList);
      }
    }
    return attrInfo.build();
  }
}
