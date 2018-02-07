package com.google.testing.junit.runner.model;

import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.io.StringWriter;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class AntXmlResultWriterTest {
    private static final long NOW = 1;
    private static TestSuiteNode root;
    private static XmlWriter writer;
    private static AntXmlResultWriter resultWriter = new AntXmlResultWriter();
    private static StringWriter stringWriter;

    @BeforeClass
    public static void createDescriptions() {
        stringWriter = new StringWriter();
        writer = XmlWriter.createForTesting(stringWriter);
        root = new TestSuiteNode(Description.createSuiteDescription("root"));
    }

    @Test
    public void testFilteredCasesNotWritten() throws IOException {
        TestSuiteNode parent = createTestSuite();
        TestCaseNode test1 = createTestCase(parent);
        runToCompletion(test1);

        createTestCase(parent); // creates a test case that is FILTERED by default

        resultWriter.writeTestSuites(writer, root.getResult());

        String resultXml = stringWriter.toString();
        assertThat(resultXml)
                .contains("<testcase name='testCase1' classname='com.google.testing.junit.runner.model.TestCaseNodeTest$TestSuite'");
        assertThat(resultXml)
                .doesNotContain("<testcase name='testCase2' classname='com.google.testing.junit.runner.model.TestCaseNodeTest$TestSuite'");
    }

    private void runToCompletion(TestCaseNode test1) {
        test1.started(NOW);
        test1.finished(NOW + 1);
    }

    private TestCaseNode createTestCase(TestSuiteNode parent) {
        int idx = parent.getChildren().size() + 1;
        TestCaseNode testCase = new TestCaseNode(Description.createSuiteDescription("testCase" + idx), parent);
        parent.addTestCase(testCase);
        return testCase;
    }

    private TestSuiteNode createTestSuite() {
        Description suite = Description.createSuiteDescription(TestCaseNodeTest.TestSuite.class);
        TestSuiteNode parent = new TestSuiteNode(suite);
        root.addTestSuite(parent);
        return parent;
    }
}
