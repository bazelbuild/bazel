package com.google.testing.junit.runner.model;

import org.junit.Before;
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
    private static AntXmlResultWriter resultWriter;
    private static StringWriter stringWriter;

    @Before
    public void before() {
        stringWriter = new StringWriter();
        writer = XmlWriter.createForTesting(stringWriter);
        resultWriter = new AntXmlResultWriter();
        root = new TestSuiteNode(Description.createSuiteDescription("root"));
    }

    @Test
    public void allPassingTestCasesWritten() throws IOException {
        TestSuiteNode parent = createTestSuite();
        TestCaseNode test1 = createTestCase(parent);
        TestCaseNode test2 = createTestCase(parent);
        runToCompletion(test1);
        runToCompletion(test2);

        resultWriter.writeTestSuites(writer, root.getResult());
        String resultXml = stringWriter.toString();
        assertThat(resultXml).contains("<testcase name='testCase1'");
        assertThat(resultXml).contains("<testcase name='testCase2'");
    }

    @Test
    public void testFilteredCasesNotWritten() throws IOException {
        TestSuiteNode parent = createTestSuite();
        TestCaseNode test1 = createTestCase(parent);
        runToCompletion(test1);

        createTestCase(parent); // creates a test case that is FILTERED by default

        resultWriter.writeTestSuites(writer, root.getResult());

        String resultXml = stringWriter.toString();
        assertThat(resultXml).contains("<testcase name='testCase1'");
        assertThat(resultXml).doesNotContain("<testcase name='testCase2'");
    }

    private void runToCompletion(TestCaseNode test) {
        test.started(NOW);
        test.finished(NOW + 1);
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
