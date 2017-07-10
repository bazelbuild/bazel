package joptsimple.examples;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class AlternativeLongOptionsTest {
    @Test
    public void handlesAlternativeLongOptions() {
        OptionParser parser = new OptionParser( "W;" );
        parser.recognizeAlternativeLongOptions( true );  // same effect as above
        parser.accepts( "level" ).withRequiredArg();

        OptionSet options = parser.parse( "-W", "level=5" );

        assertTrue( options.has( "level" ) );
        assertTrue( options.hasArgument( "level" ) );
        assertEquals( "5", options.valueOf( "level" ) );
    }
}
