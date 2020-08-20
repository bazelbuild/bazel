package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.*;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.syntax.Location;
import org.junit.Test;
import static org.junit.Assert.*;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class InstrumentationFilterTest {
  @Test
  public void testComputeInstrumentationFilter() {
    List<Target> testTargets =
        Arrays.asList(
            new TestTarget("//foo"),
            new TestTarget("//foo/bar"),
            new TestTarget("//bar"),
            new TestTarget("@repo1//foo/bar"));
    EventHandler eventHandler = event -> {};
    String filter =
        InstrumentationFilterSupport.computeInstrumentationFilter(eventHandler, testTargets);
    assertEquals("@repo1//foo/bar[/:],^//bar[/:],^//foo[/:]", filter);

    testTargets =
        Arrays.asList(
            new TestTarget("//"),
            new TestTarget("@repo1//foo"),
            new TestTarget("//foo"),
            new TestTarget("@repo1//"));
    filter = InstrumentationFilterSupport.computeInstrumentationFilter(eventHandler, testTargets);
    assertEquals("@repo1//,^//", filter);
  }
}

class TestTarget implements Target {
  String pkg;

  TestTarget(String pkg) {
    this.pkg = pkg;
  }

  @Override
  public Package getPackage() {
    return null;
  }

  @Override
  public String getTargetKind() {
    return null;
  }

  @Nullable
  @Override
  public Rule getAssociatedRule() {
    return null;
  }

  @Override
  public License getLicense() {
    return null;
  }

  @Override
  public Location getLocation() {
    return null;
  }

  @Override
  public Set<License.DistributionType> getDistributions() {
    return null;
  }

  @Override
  public RuleVisibility getVisibility() {
    return null;
  }

  @Override
  public boolean isConfigurable() {
    return false;
  }

  @Override
  public Label getLabel() {
    try {
      return Label.create(this.pkg, "go_default_test");
    } catch (LabelSyntaxException e) {
      e.printStackTrace();
    }
    return null;
  }

  @Override
  public String getName() {
    return null;
  }
}
