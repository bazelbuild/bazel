package joptsimple.examples;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class LongOptionsWithArgumentPositioningTest {
    @Test
    public void allowsDifferentFormsOfPairingArgumentWithOption() {
        OptionParser parser = new OptionParser();
        parser.accepts( "count" ).withRequiredArg();
        parser.accepts( "level" ).withOptionalArg();

        OptionSet options = parser.parse( "--count", "4", "--level=3" );

        assertTrue( options.has( "count" ) );
        assertTrue( options.hasArgument( "count" ) );
        assertEquals( "4", options.valueOf( "count" ) );

        assertTrue( options.has( "level" ) );
        assertTrue( options.hasArgument( "level" ) );
        assertEquals( "3", options.valueOf( "level" ) );
    }
}
