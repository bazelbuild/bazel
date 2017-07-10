package joptsimple.examples;

import java.util.List;

import static java.util.Arrays.*;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class OptionSynonymTest {
    @Test
    public void supportsOptionSynonyms() {
        OptionParser parser = new OptionParser();
        List<String> synonyms = asList( "message", "blurb", "greeting" );
        parser.acceptsAll( synonyms ).withRequiredArg();
        String expectedMessage = "Hello";

        OptionSet options = parser.parse( "--message", expectedMessage );

        for ( String each : synonyms ) {
            assertTrue( each, options.has( each ) );
            assertTrue( each, options.hasArgument( each ) );
            assertEquals( each, expectedMessage, options.valueOf( each ) );
            assertEquals( each, asList( expectedMessage ), options.valuesOf( each ) );
        }
    }
}
