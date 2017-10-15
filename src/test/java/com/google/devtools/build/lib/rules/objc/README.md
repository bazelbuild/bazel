# Objc rule tests
This package contains subclasses of ObjcRuleTestCase.  These test cases use
--experimental_objc_crosstool=all by default, as per
ObjcRuleTestCase#useConfiguration.  This is meant to test the "crosstool"
case.

The "legacy" case (that is, --experimental_objc_crosstool=off) is also tested
in subclasses prefixed with the word "Legacy".  Tests in the superclass, then,
are tested for both crosstool configurations, while tests in the subclass are
only tested for --experimental_objc_crosstool=off.

As the crosstool case is developed, tests will moved up to superclasses.
Eventually, the legacy subclasses will be removed.
