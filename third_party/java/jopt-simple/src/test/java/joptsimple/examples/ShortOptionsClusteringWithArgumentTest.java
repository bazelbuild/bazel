package joptsimple.examples;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class ShortOptionsClusteringWithArgumentTest {
    @Test
    public void allowsClusteringShortOptionsThatAcceptArguments() {
        OptionParser parser = new OptionParser();
        parser.accepts( "a" );
        parser.accepts( "B" );
        parser.accepts( "c" ).withRequiredArg();

        OptionSet options = parser.parse( "-aBcfoo" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.has( "B" ) );
        assertTrue( options.has( "c" ) );
        assertEquals( "foo", options.valueOf( "c" ) );
    }
}
